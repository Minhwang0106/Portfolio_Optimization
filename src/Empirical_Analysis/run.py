"""Run every strategy over the sample period and write the comparison out.

    python -m src.Empirical_Analysis.run                      # all ten
    python -m src.Empirical_Analysis.run --n-workers 5
    python -m src.Empirical_Analysis.run --only epo ppp --verbose
    python -m src.Empirical_Analysis.run --rebuild            # summary only

`--rebuild` is the one to reach for after changing a metric: it replays the
weights a finished run already wrote and recomputes everything downstream of
them, in seconds rather than hours, without importing a single model. See
`rebuild`.

Ten strategies: six weighting rules, and four re-solves of the proposed model
that hold it to a comparator's breadth or put its moments through B&H's rule.

Every weighting rule here is long-only and fully invested by default, which is
what makes the return columns comparable at all; `--long-short` restores the
long-short forms the three papers state, and it reaches all of them -- `epo`,
`ppp` and both `proposed_*` runs. It leaves `icc_mvo_ex_post` alone: Bielstein
and Hanauer state a long-only book, so that is the one it runs. The four
matched rows stay long-only too: the breadth they are held to is read off a
long-only book.

One EPO row, not two. Under the default it is the objective solved on the
simplex, which is this repo's addition: eq. (16) is what Proposition 2 (p.13,
proved p.36) derives as the *exact* solution to a robust max-min problem over
an ellipsoidal uncertainty region for expected returns, and that problem has no
sign or budget constraint. `--long-short` is what runs EPO as the paper states
it, and that is the form to quote against the paper. The shrinkage `w` is
selected against whichever book is built, never across the two -- see
`EPO.Config`.

* `equal_weight` -- the benchmark `PPP` tilts away from, on the same universe.
* `epo` -- Enhanced Portfolio Optimization, monthly. Long-only and summing to
  one by default, so it sits on the same footing as `ppp` and the proposed
  model; eq. (16) unconstrained, scaled to one unit of gross exposure, under
  `--long-short`.
* `ppp` -- the parametric portfolio policy, monthly, shorts truncated inside
  the theta fit.
* `icc_mvo_ex_post` -- Bielstein & Hanauer's (2019) construction: maximum
  Sharpe ratio on the Gebhardt-Lee-Swaminathan implied cost of capital plus
  rescaled momentum, Ledoit-Wolf covariance, long-only with a 5% cap.
  Quarterly by default, on the proposed model's calendar; `--icc-annual`
  rebalances each June, as B&H do. **Lookahead; not a strategy.** Its eleven
  explicit earnings years are the realised ones -- the same future window
  `proposed_ex_post` elicits its moments from -- so it is a point-estimate
  benchmark under perfect earnings foresight, read against the ex post rows
  and never against the tradeable ones.
* `proposed_historical` -- the residual income model, quarterly, moments
  elicited from the training window. **This is the tradeable one.**
* `proposed_ex_post` -- the same model with `ex_post=True`, i.e. moments
  elicited from the realised future window. **Lookahead; not a strategy.** It
  exists to split the model's error into the part that is the simulation
  machinery and the part that is not knowing the future moments -- the gap
  between it and `proposed_historical` is what perfect moment forecasting would
  be worth. Read it as a ceiling, never as a result.

The four matched rows re-solve the proposed model from the implied-return dumps
(`strategy.replay_weight`), on the quarterly calendar, and never simulate:

* `proposed_historical_epo_n`, `proposed_historical_ppp_n` -- CRRA, held at
  each rebalance date to EPO's or PPP's effective N at that date.
* `proposed_ex_post_icc_n` -- CRRA, held to `icc_mvo_ex_post`'s effective N.
* `proposed_ex_post_msr_icc_n` -- maximum Sharpe on the simulation's mean and
  covariance, held to the same breadth: B&H's rule on this model's moments.

Each needs its comparator's books, from this run or from `weights_<comparator>.csv`
in `--out`, which is how the rows are added to a finished run:
`--only proposed_historical_epo_n proposed_historical_ppp_n`, then `--rebuild`
to put every saved strategy back in the summary.

Each strategy is one job, and the jobs run in their own processes because the
class-level panels (`RIM_PortOp.general_data`, `EPO.vol`, `PPP.characteristics`)
do not cross a process boundary -- so a worker configures whatever its own
strategy needs and nothing else. That also means the two proposed variants each
fit their own copulas rather than sharing one fit; they differ only in the
elicitation window, so this duplicates real work, but sharing it would mean
threading a cache through `RIM_PortOp.sampling` and the two are the long poles
anyway, running side by side.

The `__main__` guard at the bottom is load bearing on Windows, not decoration:
under the spawn start method every worker re-imports this module, and without
the guard that re-import runs the whole backtest again in each of them.

Reproducibility has two separate seeds because the randomness sits in two
places. `RIM_PortOp` takes a seed as an argument, varied per formation date --
one fixed seed would draw the *same* copula sample at every date, correlating
the errors across the backtest and understating its noise. `PPP.find_theta`
instead seeds its optimiser's starting guess off the global numpy state, so that
one is fixed once, inside the worker that runs it.
"""
import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any
from constant import (
    testing_period, RAW_BACKTEST_DIR, backtest_cost_bps, ppp_estimation_month,
    risk_aversion, IMPLIED_RETURN_DIR, icc_weight_cap
)
from .data import universe, monthly_return, risk_free
from .engine import backtest, quarter_ends, BacktestResult, WeightFn
from .metrics import summarise, cumulative_wealth

STRATEGIES: tuple[str, ...] = ('equal_weight', 'epo', 'ppp', 'icc_mvo_ex_post',
                               'proposed_historical', 'proposed_ex_post',
                               'proposed_historical_epo_n',
                               'proposed_historical_ppp_n',
                               'proposed_ex_post_msr_icc_n',
                               'proposed_ex_post_icc_n')

# The two that call `RIM_PortOp`, and the elicitation window each one uses.
_PROPOSED: dict[str, bool] = {'proposed_historical': False,
                              'proposed_ex_post': True}

# The ones re-solved from the dumps: (dump variant, objective, comparator whose
# effective N the book is held to at each date). See `strategy.replay_weight`.
_REPLAY: dict[str, tuple[str, str, str]] = {
    'proposed_historical_epo_n': ('historical', 'crra', 'epo'),
    'proposed_historical_ppp_n': ('historical', 'crra', 'ppp'),
    'proposed_ex_post_msr_icc_n': ('ex_post', 'max_sharpe', 'icc_mvo_ex_post'),
    'proposed_ex_post_icc_n': ('ex_post', 'crra', 'icc_mvo_ex_post')}


def _dated_seed (base: int, date: pd.Timestamp)-> int:
    """A distinct but reproducible seed per formation date.

    `base` alone would hand every date the same copula draw; `date.toordinal()`
    alone would not respond to the seed at all. Masked into 31 bits because that
    is what numpy's legacy seeding accepts.
    """
    return (base*1_000_003+date.toordinal()) % (2**31-1)


def effective_n_target (weights: pd.DataFrame, dates: list[pd.Timestamp]
                        )-> pd.Series:
    """A comparator's effective N in force at each date: what a matched book is held to.

    Read off the comparator's own formation books -- the last one at or before
    each date -- so the target at `t` uses nothing the comparator had not
    already decided at `t`. Its full-sample average would be known only at the
    end of the sample, which makes it lookahead in a tradeable row.

    Args:
        weights (pd.DataFrame): `BacktestResult.weights`, or a saved
            `weights_<strategy>.csv` read with dates parsed: formation dates
            down the rows, one column per ticker, NaN where not held.
        dates (list[pd.Timestamp]): Dates to read the target at.

    Returns:
        pd.Series: `gross^2 / sum(w^2)` per date, on `dates`; NaN before the
            comparator's first book.

    Example:
        >>> effective_n_target(results['epo'].weights, quarter_ends(dates))
        date
        2015-03-31    68.1...
    """
    held: pd.DataFrame = weights.sort_index().fillna(0.0)
    breadth: pd.Series = held.abs().sum(axis=1)**2/(held**2).sum(axis=1)
    return breadth.reindex(pd.DatetimeIndex(dates),
                           method='ffill').rename('effective_n')


def _breadth_targets (labels: list[str], collected: dict[str, BacktestResult],
                      result_dir: Path|None, dates: list[pd.Timestamp]
                      )-> dict[str, pd.Series]:
    """Each matched row's effective-N floor per date, read off its comparator.

    From this call's own run of the comparator where there is one, otherwise
    from the `weights_<comparator>.csv` a finished run left in `result_dir` --
    which is what lets the matched rows be added without re-running EPO, PPP or
    B&H.

    Raises:
        FileNotFoundError: If a comparator is neither in this run nor saved in
            `result_dir`.
    """
    targets: dict[str, pd.Series] = {}
    for label in labels:
        comparator: str = _REPLAY[label][2]
        if comparator in collected:
            weights: pd.DataFrame = collected[comparator].weights
        else:
            path: Path|None = (None if result_dir is None
                               else Path(result_dir)/f'weights_{comparator}.csv')
            if path is None or not path.exists():
                raise FileNotFoundError(
                    f'{label} is held to the breadth of {comparator}, which is '
                    f'neither in this run nor saved in {result_dir}; run it '
                    'first, or name it in `only` alongside')
            weights = pd.read_csv(path, index_col=0, parse_dates=True)
        targets[label] = effective_n_target(weights, dates)
    return targets


def universe_from_books (path: Path)-> dict[pd.Timestamp, list[str]]:
    """The candidate universe a finished run used, read back off its equal-weight books.

    `equal_weight` holds every candidate at every date, so its saved books are
    the universe of the run that wrote them -- the only record of it once
    `applicable_ticker.csv` has been rebuilt from newer panels. Running a new
    strategy on it keeps the "same universe" control against that run intact.

    Args:
        path (Path): A `weights_equal_weight.csv` that `save` wrote.

    Returns:
        dict[pd.Timestamp, list[str]]: As `data.universe`.

    Example:
        >>> uni = universe_from_books(RAW_BACKTEST_DIR/'weights_equal_weight.csv')
        >>> len(uni[pd.Timestamp('2015-01-31')])
        169
    """
    books: pd.DataFrame = pd.read_csv(path, index_col=0, parse_dates=True)
    return {date: row.dropna().index.tolist() for date, row in books.iterrows()}


def _proposed_at (date: pd.Timestamp, tickers: list[str], base_seed: int,
                  n_samples: int, n_lags: int, ex_post: bool, common_theta: bool,
                  long_only: bool, dump_dir: Path|None = None)-> pd.Series:
    """`strategy.proposed_weight` with the seed varied by date. See `_dated_seed`."""
    from .strategy import proposed_weight
    return proposed_weight(date, tickers, n_samples=n_samples, n_lags=n_lags,
                           ex_post=ex_post, common_theta=common_theta,
                           seed=_dated_seed(base_seed, date),
                           long_only=long_only, dump_dir=dump_dir)


def _build (label: str, params: dict[str, Any]
            )-> tuple[WeightFn, list[pd.Timestamp]|None]:
    """Configure one strategy's model and return its weighting rule and schedule.

    Called *inside* the worker that will run the strategy, never in the parent:
    every model here keeps its panels as class attributes, and class attributes
    set in the parent do not survive the fork/spawn to a child. Configuring per
    worker also means a run that skips a model never pays that model's setup --
    `EPO.Config` alone sweeps all of `constant.testing_period`.

    Args:
        label (str): One of `STRATEGIES`.
        params (dict[str, Any]): The run-level settings; see `run_all`.

    Returns:
        tuple[WeightFn, list[pd.Timestamp] | None]: The `(date, tickers) ->
            Series` callable, and its rebalance schedule (None meaning every
            formation date).

    Raises:
        ValueError: If `label` is not a known strategy.
    """
    # Imported here rather than at module scope so that a worker running
    # `equal_weight` does not import pyvinecopulib and parse the accounting
    # panel on the way in.
    from .strategy import epo_weight, ppp_weight, equal_weight

    if label == 'equal_weight':
        return equal_weight, None

    if label == 'epo':
        from ..EPO.model import EPO
        # Its own inner pool. Only one of the five jobs is EPO, so the nesting
        # is one level deep in one worker rather than five ways of
        # oversubscription. `ProcessPoolExecutor` children are non-daemonic,
        # which is what makes a nested pool legal at all.
        #
        # The run-level `long_only` reaches EPO exactly as it reaches PPP and
        # the proposed model: one book is built, one shrinkage is selected
        # against it, and one EPO row comes out. Under the default that is the
        # long-only solve, the expensive half of the sweep -- one SLSQP per
        # date per grid point, which is why this job wants `--epo-workers`.
        EPO.Config(long_only=params['long_only'],
                   n_workers=params['epo_workers'])
        return (partial(epo_weight, how=params['epo_scale']), None)

    if label == 'ppp':
        from ..PPP.model import PPP
        # `find_theta` seeds its optimiser's starting guess with
        # `np.random.random(3)` off the global state, so this is the only way to
        # make the fit reproducible without editing the model. Set in the worker
        # because the worker has its own interpreter and its own global state.
        np.random.seed(params['seed'])
        PPP.Config()
        # One instance across every date, not one per date: it accumulates
        # `theta` and `weight` per formation date and those are worth keeping.
        return (partial(ppp_weight, PPP(), n_month=params['ppp_n_month'],
                        long_only=params['long_only']),
                None)

    if label == 'icc_mvo_ex_post':
        from ..ICC_MVO.model import Icc_Mvo
        from .strategy import icc_mvo_weight
        Icc_Mvo.Config()
        # Quarterly by default, on the proposed model's calendar, so that the
        # two ex post rows differ in the model and not in how often they
        # trade. B&H themselves rebalance once a year, at the end of June.
        # Long-only whatever `params['long_only']` says: that is B&H's book.
        schedule: list[pd.Timestamp] = (
            [d for d in params['dates'] if d.month == 6]
            if params['icc_annual'] else quarter_ends(params['dates']))
        return partial(icc_mvo_weight, cap=params['icc_cap']), schedule

    if label in _PROPOSED:
        from ..Proposed_Model.model import RIM_PortOp
        RIM_PortOp.Config()
        dates: list[pd.Timestamp] = params['dates']
        return (partial(_proposed_at, base_seed=params['seed'],
                        n_samples=params['n_samples'],
                        n_lags=params['n_lags'],
                        ex_post=_PROPOSED[label],
                        common_theta=params['common_theta'],
                        long_only=params['long_only'],
                        dump_dir=params['dump_dir']),
                quarter_ends(dates) if params['proposed_quarterly'] else None)

    if label in _REPLAY:
        from .strategy import replay_weight
        variant, objective, _ = _REPLAY[label]
        # Quarterly whatever `proposed_quarterly` says: the dumps were written
        # on that calendar and there is nothing to read on any other date.
        return (partial(replay_weight, variant=variant, objective=objective,
                        n_eff=params['targets'][label], rf=params['rf'],
                        dump_dir=params['replay_dir']),
                quarter_ends(params['dates']))

    raise ValueError(f'unknown strategy {label!r}; pick from {STRATEGIES}')


def _run_one (job: tuple[str, dict[str, Any]])-> tuple[str, BacktestResult]:
    """Configure and backtest one strategy. The unit of work the pool maps over.

    Takes a `(label, params)` pair rather than a prepared callable because a
    prepared callable would have to be pickled to reach the worker, and the
    models' state lives on their classes -- a pickled `PPP` instance arrives
    with `PPP.characteristics` unset. The panels are re-read here for the same
    reason, which costs a parse per worker and saves shipping them.

    Args:
        job (tuple[str, dict[str, Any]]): `(label, params)`; see `run_all`.

    Returns:
        tuple[str, BacktestResult]: The label back, so results can be keyed even
            though a pool does not guarantee completion order.
    """
    label, params = job
    fn, rebal = _build(label, params)
    return_df: pd.DataFrame = monthly_return()
    uni: dict[pd.Timestamp, list[str]] = (universe() if params['universe'] is None
                                          else params['universe'])
    with warnings.catch_warnings():
        # The models warn per ticker per date -- thin characteristics, an
        # unfittable conditional law. Informative once, deafening over 132
        # dates, and anything that actually cost a date is in `failures`.
        warnings.simplefilter('once', RuntimeWarning)
        result: BacktestResult = backtest(
            fn, params['dates'], return_df, uni, rebalance=rebal,
            cost_bps=params['cost_bps'], name=label,
            verbose=params['verbose'])
    return label, result


def _map_jobs (jobs: list[tuple[str, dict[str, Any]]], n_workers: int
               )-> dict[str, BacktestResult]:
    """`_run_one` over `jobs`: in a pool, unless there is one job or one worker."""
    collected: dict[str, BacktestResult] = {}
    if not jobs:
        return collected
    if n_workers <= 1 or len(jobs) == 1:
        for job in jobs:
            label, res = _run_one(job)
            collected[label] = res
    else:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(jobs))) as ex:
            for label, res in ex.map(_run_one, jobs):
                collected[label] = res
    return collected


def run_all (dates=testing_period, only: tuple[str, ...]|None = None,
             skip: tuple[str, ...] = (),
             n_workers: int = 5,
             cost_bps: float = backtest_cost_bps,
             epo_scale: str = 'gross', epo_workers: int|None = None,
             ppp_n_month: int = ppp_estimation_month,
             proposed_quarterly: bool = True,
             icc_cap: float = icc_weight_cap, icc_annual: bool = False,
             n_samples: int = 10000, n_lags: int = 4,
             common_theta: bool = True, long_only: bool = True,
             seed: int = 0, verbose: bool = False,
             sr_benchmark: str|None = 'equal_weight', n_boot: int = 4999,
             sr_block: int|str = 5,
             result_dir: Path|None = RAW_BACKTEST_DIR,
             dump_dir: Path|None = None,
             replay_dir: Path = IMPLIED_RETURN_DIR,
             universe_from: Path|None = None
             )-> tuple[dict[str, BacktestResult], pd.DataFrame]:
    """Backtest every strategy on one universe and one calendar, in parallel.

    Every strategy is handed the same candidate universe from `data.universe` at
    the same formation dates, so a difference between them is a difference in
    the weighting rule and not in what they were allowed to hold. Each still
    screens that list its own way -- PPP for a complete characteristic record,
    `RIM_PortOp` for `constant.min_char_obs` usable quarters, `Icc_Mvo` for a
    solvable implied cost of capital -- and the surviving book's breadth is
    reported as `avg_effective_n`.

    Args:
        dates: Formation dates. Defaults to `constant.testing_period` (month
            ends, 2015-01 to 2025-12). The last produces no return, there being
            no following month to earn.
        only (tuple[str, ...] | None): Run just these. None runs all of
            `STRATEGIES` less `skip`.
        skip (tuple[str, ...]): Names to leave out. Ignored when `only` is given.
        n_workers (int): Processes to spread the strategies over. One job per
            strategy, so anything above the strategy count is idle; 1 runs them
            serially in this process, which is what a debugger wants. Defaults
            to 5.
        cost_bps (float): One-way transaction cost in basis points of traded
            notional. Defaults to `constant.backtest_cost_bps`.
        epo_scale (str): How to normalise EPO's raw weights -- 'gross', 'net' or
            'none', passed to `strategy.normalise`. Defaults to 'gross'. Under
            the long-only default this barely bites, that book already summing
            to one; it is what makes EPO readable at all under `--long-short`,
            since unconstrained EPO returns `Sigma_w^-1 @ signal / gamma`, whose
            leverage is whatever the covariance matrix implies rather than a
            decision the model made, and 'gross' reads it as a self-financing
            long-short book.
        epo_workers (int | None): Processes for `EPO.Config`'s own inner sweep.
            None takes EPO's default of `cpu_count - 1`. Drop it if the outer
            pool is already saturating the machine. Defaults to None.
        ppp_n_month (int): PPP's estimation window in months. Defaults to
            `constant.ppp_estimation_month`.
        proposed_quarterly (bool): Rebalance both proposed variants on quarter
            ends only, holding through the intervening months. Defaults to True;
            see `engine.quarter_ends`. EPO and PPP always rebalance monthly.
        icc_cap (float): Largest weight `icc_mvo_ex_post` puts on one name.
            Defaults to `constant.icc_weight_cap`, Bielstein & Hanauer's 5%;
            1 removes the cap.
        icc_annual (bool): Rebalance `icc_mvo_ex_post` once a year at the end
            of June, as B&H do, instead of on the proposed model's quarter
            ends. Defaults to False, which keeps the two ex post rows on one
            calendar.
        n_samples (int): Simulated paths per ticker in the proposed model.
            Defaults to 10000. Lower it to iterate.
        n_lags (int): Lags the proposed model's persistence fit reads. Defaults
            to 4.
        common_theta (bool): Pool the proposed model's decay estimate across the
            universe. Defaults to True.
        long_only (bool): Constrain every model to a non-negative book summing
            to one, each in its own idiom -- bounded SLSQP for the proposed
            model, SLSQP on the simplex for `epo`, and BSV's
            truncate-and-renormalise for PPP, applied inside the theta fit
            rather than after it. Defaults to True.

            False restores the long-short forms those papers state, which are
            not comparable to each other on a return basis: PPP runs at a
            median 9.9x gross, 32.9x at its worst date, while EPO returns
            `Sigma_w^-1 @ signal / gamma` and is rescaled by `epo_scale`. It is
            the form to quote EPO against its paper, eq. (16) being the exact
            solution to Proposition 2's unconstrained robust problem.

            For EPO it also changes which book the shrinkage `w` is selected
            on, since `EPO.Config` builds one book and ranks `w` against that
            one; the two are close to uncorrelated, so this is not a cosmetic
            difference.
        seed (int): Base seed. Fixes PPP's optimiser start and seeds the
            proposed model's copula draw per date. Defaults to 0.
        sr_benchmark (str | None): Strategy whose Sharpe ratio every other one
            is tested against, by Ledoit-Wolf (2008); see
            `Empirical_Analysis.sharpe_inference`. Defaults to `'equal_weight'`.
            None reports the Sharpe ratios without testing them.
        n_boot (int): Bootstrap resamples behind `sr_diff_pval_boot`. Defaults
            to 4999. 0 leaves only the HAC p-value, which is the liberal one.
        sr_block (int | str): Block length for that bootstrap, or `'calibrate'`
            to run the paper's Algorithm 3.1 per strategy. Defaults to 5; the
            calibration is cheap but does not identify a block size on these
            series, so it is not the default. See `sharpe_inference`.
        verbose (bool): Print each formation date as it is reached. Interleaved
            across workers when `n_workers > 1`, so the lines arrive out of
            order; the label on each says which strategy it belongs to.
            Defaults to False.
        result_dir (Path | None): Where to write the CSVs. None writes nothing
            and only returns, which is what a notebook wants. Defaults to
            `constant.RAW_BACKTEST_DIR`. Also where a matched row looks for its
            comparator's `weights_<comparator>.csv` when the comparator is not
            in this run.
        dump_dir (Path | None): Write the proposed model's simulated implied
            returns here, one file per formation date per variant. Defaults to
            None, i.e. no files.
        replay_dir (Path): Where the matched rows read those files back from.
            Defaults to `constant.IMPLIED_RETURN_DIR`.
        universe_from (Path | None): A `weights_equal_weight.csv` whose books
            give the candidate universe at each date, instead of the current
            `applicable_ticker.csv` -- see `universe_from_books`. For adding a
            strategy to a run whose panels have since been refetched. Defaults
            to None, today's universe.

    Returns:
        tuple[dict[str, BacktestResult], pd.DataFrame]: The per-strategy runs
            keyed by label in `STRATEGIES` order, and the after-cost summary
            table from `metrics.summarise`.

    Raises:
        ValueError: If nothing is left to run, or if `only`/`skip` names a
            strategy that does not exist.
        FileNotFoundError: If a matched row's comparator is neither in this run
            nor saved in `result_dir`.

    Note:
        `proposed_ex_post` is a lookahead diagnostic, not a strategy -- its
        moments are elicited from the realised future window. See the module
        docstring and `strategy.proposed_weight`. It is in the summary table
        because the gap to `proposed_historical` is the number worth reading;
        quoting its Sharpe ratio on its own would be quoting a forecast of the
        past. `icc_mvo_ex_post` is the same kind of row: its explicit earnings
        years are realised, not forecast.

    Example:
        >>> results, table = run_all(only=('equal_weight', 'ppp'),
        ...                          result_dir=None)
        >>> table.loc[['ann_return', 'sharpe']].round(3)
                    equal_weight    ppp
        ann_return         0.116  0.134
        sharpe             0.635  0.661
    """
    known: set[str] = set(STRATEGIES)
    if only is not None:
        unknown: set[str] = set(only)-known
        if unknown:
            raise ValueError(f'unknown strategy {sorted(unknown)}; '
                             f'pick from {STRATEGIES}')
        wanted: list[str] = [s for s in STRATEGIES if s in set(only)]
    else:
        unknown = set(skip)-known
        if unknown:
            raise ValueError(f'unknown strategy {sorted(unknown)}; '
                             f'pick from {STRATEGIES}')
        wanted = [s for s in STRATEGIES if s not in set(skip)]
    if not wanted:
        raise ValueError(f'nothing left to run; pick from {STRATEGIES}')

    params: dict[str, Any] = {
        'dates': [pd.Timestamp(d) for d in dates],
        'cost_bps': cost_bps, 'epo_scale': epo_scale,
        'epo_workers': epo_workers, 'ppp_n_month': ppp_n_month,
        'proposed_quarterly': proposed_quarterly,
        'icc_cap': icc_cap, 'icc_annual': icc_annual, 'n_samples': n_samples,
        'n_lags': n_lags, 'common_theta': common_theta,
        'long_only': long_only, 'seed': seed, 'verbose': verbose,
        'dump_dir': dump_dir, 'replay_dir': replay_dir,
        'universe': (None if universe_from is None
                     else universe_from_books(Path(universe_from))),
    }
    replay: list[str] = [s for s in wanted if s in _REPLAY]
    collected: dict[str, BacktestResult] = _map_jobs(
        [(s, params) for s in wanted if s not in _REPLAY], n_workers)
    if replay:
        # Second, because each matched row is held to a comparator's breadth at
        # every date, and that comparator may be one this call has just run.
        matched: dict[str, Any] = {
            **params, 'rf': risk_free(),
            'targets': _breadth_targets(replay, collected, result_dir,
                                        params['dates'])}
        collected.update(_map_jobs([(s, matched) for s in replay], n_workers))

    # `STRATEGIES` order, not completion order -- a pool returns whenever each
    # job finishes, and a summary table whose columns reorder between runs is
    # unreadable side by side.
    results: dict[str, BacktestResult] = {s: collected[s] for s in wanted}
    rf: pd.Series = risk_free()
    # Seeded off the run's own seed so the bootstrap p-values reproduce with
    # everything else, and benchmarked on `equal_weight` -- which is why a run
    # that drops it gets a warning rather than a silently untested table.
    table: pd.DataFrame = summarise(results, rf, net=True, gamma=risk_aversion,
                                    benchmark=sr_benchmark, n_boot=n_boot,
                                    block_size=sr_block, seed=seed)

    if result_dir is not None:
        save(results, table, rf, result_dir)
    return results, table


def save (results: dict[str, BacktestResult], table: pd.DataFrame,
          rf: pd.Series, result_dir: Path = RAW_BACKTEST_DIR)-> Path:
    """Write the run to CSV: returns, wealth, summary, diagnostics, weights.

    Args:
        results (dict[str, BacktestResult]): Per-strategy runs from `run_all`.
        table (pd.DataFrame): The summary table from `metrics.summarise`.
        rf (pd.Series): Monthly risk-free rate, saved alongside so the Sharpe
            ratios can be recomputed from the returns file alone.
        result_dir (Path): Output directory, created if absent. Defaults to
            `constant.RAW_BACKTEST_DIR`.

    Returns:
        Path: The directory written to.

    Files:
        * `summary.csv` -- the comparison table, strategies across the columns.
        * `monthly_returns.csv` -- gross and net monthly returns per strategy,
          plus `rf`.
        * `cumulative_wealth.csv` -- one unit of capital compounded, after cost.
        * `diagnostics_<strategy>.csv` -- exposures, turnover, holding counts.
        * `weights_<strategy>.csv` -- target weights at each rebalance date.
        * `failures.csv` -- formation dates whose weighting rule raised, and the
          exception. Absent when nothing failed.

    Example:
        >>> save(results, table, risk_free())
        WindowsPath('.../Data/Result/raw_backtest')
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    table.to_csv(result_dir/'summary.csv')

    frames: dict[str, pd.Series] = {}
    for label, res in results.items():
        frames[f'{label}_gross'] = res.returns
        frames[f'{label}_net'] = res.net_returns
    returns_df: pd.DataFrame = pd.DataFrame(frames).sort_index()
    returns_df['rf'] = rf.reindex(returns_df.index)
    returns_df.to_csv(result_dir/'monthly_returns.csv')

    cumulative_wealth(results, net=True).to_csv(
        result_dir/'cumulative_wealth.csv')

    for label, res in results.items():
        res.diagnostics.to_csv(result_dir/f'diagnostics_{label}.csv')
        if not res.weights.empty:
            res.weights.to_csv(result_dir/f'weights_{label}.csv')

    failed: list[dict] = [{'strategy': label, 'date': date, 'error': msg}
                          for label, res in results.items()
                          for date, msg in res.failures.items()]
    if failed:
        pd.DataFrame(failed).to_csv(result_dir/'failures.csv', index=False)
    return result_dir


def rebuild (result_dir: Path = RAW_BACKTEST_DIR, dates=testing_period,
             cost_bps: float = backtest_cost_bps,
             sr_benchmark: str|None = 'equal_weight', n_boot: int = 4999,
             sr_block: int|str = 5, seed: int = 0, verify: bool = True
             )-> tuple[dict[str, BacktestResult], pd.DataFrame]:
    """Redo the summary from a finished run's saved weights, without the models.

    Changing a metric should not cost another backtest. The expensive half of a
    run is the weighting rules, and their output is already in
    `weights_<strategy>.csv`, so this replays those weights through the same
    `engine.backtest` -- same drift, same turnover, same costs -- and
    re-summarises. `RIM_PortOp`, `EPO` and `PPP` are never imported.

    That replay is the reason this is not simply re-reading `summary.csv`: any
    statistic of the *held* book, `effective_n` among them, is a per-month quantity
    that the saved aggregates cannot reconstruct. Replaying regenerates the
    per-month diagnostics too.

    Args:
        result_dir (Path): A directory `save` has written. Defaults to
            `constant.RAW_BACKTEST_DIR`.
        dates: Formation dates, which must be the ones the run used. Defaults to
            `constant.testing_period`.
        cost_bps (float): Transaction cost, likewise. Defaults to
            `constant.backtest_cost_bps`.
        sr_benchmark (str | None): As `run_all`.
        n_boot (int): As `run_all`.
        sr_block (int | str): As `run_all`.
        seed (int): As `run_all`.
        verify (bool): Check the replayed net returns against the saved
            `monthly_returns.csv` and raise if they differ. Defaults to True.
            Set it False to replay under a *different* `cost_bps`, which is the
            one case where the returns are meant to move.

    Returns:
        tuple[dict[str, BacktestResult], pd.DataFrame]: As `run_all`, and
            written to `result_dir` the same way.

    Raises:
        FileNotFoundError: If no `weights_*.csv` is there to replay.
        ValueError: If `verify` and the replay does not reproduce the saved
            returns -- which means `dates` or `cost_bps` is not what the run
            used, or the price panel has been refetched since, and every number
            downstream would be quietly wrong.

    Note:
        Each saved book is replayed exactly as it was held. It is not screened
        again against the current `applicable_ticker.csv`, which changes
        whenever the panels are rebuilt; a universe that has moved on since the
        run changes nothing here, since the books already record who was held.

    Example:
        >>> _, table = rebuild()
        >>> table.loc['sharpe']
        equal_weight    0.6322
        ...
    """
    saved: Path = result_dir/'monthly_returns.csv'
    files: dict[str, Path] = {p.stem.removeprefix('weights_'): p
                              for p in sorted(result_dir.glob('weights_*.csv'))}
    if not files:
        raise FileNotFoundError(f'no weights_*.csv in {result_dir}; '
                                'there is no run here to rebuild from')
    # `STRATEGIES` order where they are known, so the table's columns do not
    # reorder just because the directory listing did.
    labels: list[str] = ([s for s in STRATEGIES if s in files]
                         +[s for s in files if s not in STRATEGIES])

    # Failures cannot be rediscovered -- a lookup rule never raises -- so they
    # are carried across from the run that did fail.
    failed: dict[str, dict[pd.Timestamp, str]] = {}
    if (result_dir/'failures.csv').exists():
        for _, row in pd.read_csv(result_dir/'failures.csv').iterrows():
            failed.setdefault(str(row['strategy']), {})[
                pd.Timestamp(row['date'])] = str(row['error'])

    return_df: pd.DataFrame = monthly_return()
    uni: dict[pd.Timestamp, list[str]] = universe()
    formation: list[pd.Timestamp] = [pd.Timestamp(d) for d in dates]
    prior: pd.DataFrame|None = (pd.read_csv(saved, index_col=0,
                                            parse_dates=True)
                                if saved.exists() else None)

    results: dict[str, BacktestResult] = {}
    for label in labels:
        weights: pd.DataFrame = pd.read_csv(files[label], index_col=0,
                                            parse_dates=True)

        # The saved book exactly as it was held, not screened again against
        # today's universe. `applicable_ticker.csv` is rebuilt whenever the
        # panels are, and dropping a name it no longer lists leaves the book
        # short of fully invested instead of reproducing it -- which is what
        # the refetch of 2026-09-11 did to VZ at 51 dates.
        def rule (date: pd.Timestamp, tickers: list[str],
                  _w: pd.DataFrame = weights)-> pd.Series:
            return _w.loc[date].dropna()

        res: BacktestResult = backtest(
            rule, formation, return_df, uni, rebalance=list(weights.index),
            cost_bps=cost_bps, name=label)
        res.failures.update(failed.get(label, {}))

        if verify and prior is not None and f'{label}_net' in prior:
            gap: float = float((res.net_returns
                                -prior[f'{label}_net'].dropna()).abs().max())
            if gap > 1e-10:
                raise ValueError(
                    f'{label}: replay does not reproduce the saved run '
                    f'(max return gap {gap:.2e}). `dates` or `cost_bps` is not '
                    f'what produced {saved.name}, or the price panel has been '
                    f'refetched since; pass verify=False only if the returns '
                    f'are meant to change.')
        results[label] = res

    rf: pd.Series = risk_free()
    table: pd.DataFrame = summarise(results, rf, net=True, gamma=risk_aversion,
                                    benchmark=sr_benchmark, n_boot=n_boot,
                                    block_size=sr_block, seed=seed)
    save(results, table, rf, result_dir)
    return results, table


def _cli ()-> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Backtest the proposed model (historical and ex post), '
                    'ICC-MVO, EPO and PPP on one universe.')
    parser.add_argument('--rebuild', action='store_true',
                        help='recompute the summary from the saved weights in '
                             '--out, without running any model. Use after '
                             'changing a metric.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--only', nargs='+', choices=STRATEGIES,
                       help='run just these strategies')
    group.add_argument('--skip', nargs='+', default=[], choices=STRATEGIES,
                       help='strategies to leave out')
    parser.add_argument('--n-workers', type=int, default=5,
                        help='processes to spread the strategies over')
    parser.add_argument('--epo-workers', type=int, default=None,
                        help="processes for EPO.Config's own inner sweep")
    parser.add_argument('--cost-bps', type=float, default=backtest_cost_bps,
                        help='one-way transaction cost, bp of traded notional')
    parser.add_argument('--epo-scale', default='gross',
                        choices=['gross', 'net', 'none'],
                        help="how to normalise EPO's raw weight vector")
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='simulated paths per ticker in the proposed model')
    parser.add_argument('--monthly-proposed', action='store_true',
                        help='rebalance the proposed model monthly (very slow)')
    parser.add_argument('--icc-cap', type=float, default=icc_weight_cap,
                        help='largest weight icc_mvo_ex_post puts on one name '
                             f'(default {icc_weight_cap:g}, Bielstein and '
                             "Hanauer's; 1 removes the cap)")
    parser.add_argument('--icc-annual', action='store_true',
                        help='rebalance icc_mvo_ex_post each June, as '
                             'Bielstein and Hanauer do, instead of quarterly')
    parser.add_argument('--long-short', action='store_true',
                        help='let EPO, PPP and the proposed model go short, '
                             'i.e. run those papers as stated. For EPO this '
                             'also reselects the shrinkage against the '
                             'unconstrained book')
    parser.add_argument('--sr-benchmark', default='equal_weight',
                        choices=[*STRATEGIES, 'none'],
                        help='strategy whose Sharpe ratio the others are '
                             'tested against; "none" skips the test')
    parser.add_argument('--n-boot', type=int, default=4999,
                        help='bootstrap resamples for the Sharpe ratio test')
    parser.add_argument('--sr-block', default='5',
                        help='block length for that bootstrap, or "calibrate" '
                             'to run Algorithm 3.1 (default 5)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--out', type=Path, default=RAW_BACKTEST_DIR)
    parser.add_argument('--dump-implied-return', type=Path, nargs='?',
                        const=IMPLIED_RETURN_DIR, default=None,
                        metavar='DIR',
                        help='also write the simulated implied returns the '
                             'proposed model optimises over, one CSV per '
                             'formation date per '
                             'variant (rows simulations, columns tickers). '
                             f'Defaults to {IMPLIED_RETURN_DIR} when the flag '
                             'is given without a path. ~30 MB per file, '
                             '~2.7 GB over a full run.')
    parser.add_argument('--replay-dir', type=Path, default=IMPLIED_RETURN_DIR,
                        metavar='DIR',
                        help='where the matched rows read those files back '
                             f'from (default {IMPLIED_RETURN_DIR})')
    parser.add_argument('--universe-from', type=Path, default=None,
                        metavar='CSV',
                        help="run on the universe a finished run used, read off "
                             "its weights_equal_weight.csv, instead of today's "
                             'applicable_ticker.csv')
    return parser.parse_args()


def _say (message: str)-> None:
    """Print a message that may contain a path this console cannot encode.

    Windows hands a bare `python` a cp1252 stdout, and this repository lives
    under a path with Vietnamese diacritics -- so printing the output directory
    raises `UnicodeEncodeError` *after* every file has been written, turning a
    finished run into a traceback. Replacing the unencodable characters is the
    right trade: the point of the line is to confirm the write happened.
    """
    encoding: str = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    print(message.encode(encoding, errors='replace').decode(encoding))


if __name__ == '__main__':
    args = _cli()
    sr_block_arg = ('calibrate' if args.sr_block == 'calibrate'
                    else int(args.sr_block))
    sr_benchmark_arg = (None if args.sr_benchmark == 'none'
                        else args.sr_benchmark)
    if args.rebuild:
        _, summary_table = rebuild(
            result_dir=args.out, cost_bps=args.cost_bps,
            sr_benchmark=sr_benchmark_arg, n_boot=args.n_boot,
            sr_block=sr_block_arg, seed=args.seed)
        with pd.option_context('display.width', 160,
                               'display.float_format', '{:,.4f}'.format):
            print()
            print(summary_table)
        _say(f'\nrebuilt from saved weights in {args.out}')
        raise SystemExit(0)

    _, summary_table = run_all(
        only=tuple(args.only) if args.only else None,
        skip=tuple(args.skip), n_workers=args.n_workers,
        cost_bps=args.cost_bps, epo_scale=args.epo_scale,
        epo_workers=args.epo_workers, n_samples=args.n_samples,
        proposed_quarterly=not args.monthly_proposed,
        icc_cap=args.icc_cap, icc_annual=args.icc_annual,
        long_only=not args.long_short, seed=args.seed, verbose=args.verbose,
        sr_benchmark=sr_benchmark_arg, n_boot=args.n_boot,
        sr_block=sr_block_arg, result_dir=args.out,
        dump_dir=args.dump_implied_return, replay_dir=args.replay_dir,
        universe_from=args.universe_from)
    with pd.option_context('display.width', 160,
                           'display.float_format', '{:,.4f}'.format):
        print()
        print(summary_table)
    _say(f'\nwritten to {args.out}')
"""Run every strategy over the sample period and write the comparison out.

    python -m src.Empirical_Analysis.run                      # all five
    python -m src.Empirical_Analysis.run --n-workers 5
    python -m src.Empirical_Analysis.run --only epo ppp --verbose

Five strategies, of which two are the same model:

All five are long-only and fully invested by default, which is what makes their
return columns comparable at all; `--long-short` restores the long-short forms
the three papers state.

* `equal_weight` -- the benchmark `PPP` tilts away from, on the same universe.
* `epo` -- Enhanced Portfolio Optimization, monthly, its mean-variance
  objective solved on the long-only simplex.
* `ppp` -- the parametric portfolio policy, monthly, shorts truncated inside
  the theta fit.
* `proposed_historical` -- the residual income model, quarterly, moments
  elicited from the training window. **This is the tradeable one.**
* `proposed_forward` -- the same model with `forward=True`, i.e. moments
  elicited from the realised future window. **Lookahead; not a strategy.** It
  exists to split the model's error into the part that is the simulation
  machinery and the part that is not knowing the future moments -- the gap
  between it and `proposed_historical` is what perfect moment forecasting would
  be worth. Read it as a ceiling, never as a result.

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
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any
from constant import (
    testing_period, RESULT_DIR, backtest_cost_bps, ppp_estimation_month,
    risk_aversion
)
from .data import universe, monthly_return, risk_free
from .engine import backtest, quarter_ends, BacktestResult, WeightFn
from .metrics import summarise, cumulative_wealth

STRATEGIES: tuple[str, ...] = ('equal_weight', 'epo', 'ppp',
                               'proposed_historical', 'proposed_forward')

# The two that call `RIM_PortOp`, and the elicitation window each one uses.
_PROPOSED: dict[str, bool] = {'proposed_historical': False,
                              'proposed_forward': True}


def _dated_seed (base: int, date: pd.Timestamp)-> int:
    """A distinct but reproducible seed per formation date.

    `base` alone would hand every date the same copula draw; `date.toordinal()`
    alone would not respond to the seed at all. Masked into 31 bits because that
    is what numpy's legacy seeding accepts.
    """
    return (base*1_000_003+date.toordinal()) % (2**31-1)


def _proposed_at (date: pd.Timestamp, tickers: list[str], base_seed: int,
                  n_samples: int, n_lags: int, forward: bool, common_theta: bool,
                  long_only: bool)-> pd.Series:
    """`strategy.proposed_weight` with the seed varied by date. See `_dated_seed`."""
    from .strategy import proposed_weight
    return proposed_weight(date, tickers, n_samples=n_samples, n_lags=n_lags,
                           forward=forward, common_theta=common_theta,
                           seed=_dated_seed(base_seed, date),
                           long_only=long_only)


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
        EPO.Config(n_workers=params['epo_workers'])
        return (partial(epo_weight, how=params['epo_scale'],
                        long_only=params['long_only']), None)

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

    if label in _PROPOSED:
        from ..Proposed_Model.model import RIM_PortOp
        RIM_PortOp.Config()
        dates: list[pd.Timestamp] = params['dates']
        return (partial(_proposed_at, base_seed=params['seed'],
                        n_samples=params['n_samples'],
                        n_lags=params['n_lags'],
                        forward=_PROPOSED[label],
                        common_theta=params['common_theta'],
                        long_only=params['long_only']),
                quarter_ends(dates) if params['proposed_quarterly'] else None)

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
    uni: dict[pd.Timestamp, list[str]] = universe()
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


def run_all (dates=testing_period, only: tuple[str, ...]|None = None,
             skip: tuple[str, ...] = (),
             n_workers: int = 5,
             cost_bps: float = backtest_cost_bps,
             epo_scale: str = 'gross', epo_workers: int|None = None,
             ppp_n_month: int = ppp_estimation_month,
             proposed_quarterly: bool = True,
             n_samples: int = 10000, n_lags: int = 4,
             common_theta: bool = True, long_only: bool = True,
             seed: int = 0, verbose: bool = False,
             sr_benchmark: str|None = 'equal_weight', n_boot: int = 4999,
             result_dir: Path|None = RESULT_DIR
             )-> tuple[dict[str, BacktestResult], pd.DataFrame]:
    """Backtest every strategy on one universe and one calendar, in parallel.

    Every strategy is handed the same candidate universe from `data.universe` at
    the same formation dates, so a difference between them is a difference in
    the weighting rule and not in what they were allowed to hold. Each still
    screens that list its own way -- PPP for a complete characteristic record,
    `RIM_PortOp` for `constant.min_char_obs` usable quarters -- and the
    surviving count is reported as `avg_n_holdings`.

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
            the default `long_only` this barely bites, EPO's book already
            summing to one; it matters when `long_only` is off, where EPO
            returns `Sigma^-1 @ signal / gamma`, whose leverage is whatever the
            covariance matrix implies rather than a decision the model made, and
            'gross' reads it as a self-financing long-short book.
        epo_workers (int | None): Processes for `EPO.Config`'s own inner sweep.
            None takes EPO's default of `cpu_count - 1`. Drop it if the outer
            pool is already saturating the machine. Defaults to None.
        ppp_n_month (int): PPP's estimation window in months. Defaults to
            `constant.ppp_estimation_month`.
        proposed_quarterly (bool): Rebalance both proposed variants on quarter
            ends only, holding through the intervening months. Defaults to True;
            see `engine.quarter_ends`. EPO and PPP always rebalance monthly.
        n_samples (int): Simulated paths per ticker in the proposed model.
            Defaults to 10000. Lower it to iterate.
        n_lags (int): Lags the proposed model's persistence fit reads. Defaults
            to 4.
        common_theta (bool): Pool the proposed model's decay estimate across the
            universe. Defaults to True.
        long_only (bool): Constrain all three models to a non-negative book
            summing to one, each in its own idiom -- bounded SLSQP for the
            proposed model, a constrained mean-variance solve for EPO, and BSV's
            truncate-and-renormalise for PPP, applied inside the theta fit
            rather than after it. Defaults to True. False restores the
            long-short forms, which are the ones the three papers state but
            which are not comparable to each other on a return basis: EPO comes
            out near market neutral and PPP at ~3.6x gross.
        seed (int): Base seed. Fixes PPP's optimiser start and seeds the
            proposed model's copula draw per date. Defaults to 0.
        sr_benchmark (str | None): Strategy whose Sharpe ratio every other one
            is tested against, by Ledoit-Wolf (2008); see
            `Empirical_Analysis.sharpe_inference`. Defaults to `'equal_weight'`.
            None reports the Sharpe ratios without testing them.
        n_boot (int): Bootstrap resamples behind `sr_diff_pval_boot`. Defaults
            to 4999. 0 leaves only the HAC p-value, which is the liberal one.
        verbose (bool): Print each formation date as it is reached. Interleaved
            across workers when `n_workers > 1`, so the lines arrive out of
            order; the label on each says which strategy it belongs to.
            Defaults to False.
        result_dir (Path | None): Where to write the CSVs. None writes nothing
            and only returns, which is what a notebook wants. Defaults to
            `constant.RESULT_DIR`.

    Returns:
        tuple[dict[str, BacktestResult], pd.DataFrame]: The per-strategy runs
            keyed by label in `STRATEGIES` order, and the after-cost summary
            table from `metrics.summarise`.

    Raises:
        ValueError: If nothing is left to run, or if `only`/`skip` names a
            strategy that does not exist.

    Note:
        `proposed_forward` is a lookahead diagnostic, not a strategy -- its
        moments are elicited from the realised future window. See the module
        docstring and `strategy.proposed_weight`. It is in the summary table
        because the gap to `proposed_historical` is the number worth reading;
        quoting its Sharpe ratio on its own would be quoting a forecast of the
        past.

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
        'proposed_quarterly': proposed_quarterly, 'n_samples': n_samples,
        'n_lags': n_lags, 'common_theta': common_theta,
        'long_only': long_only, 'seed': seed, 'verbose': verbose,
    }
    jobs: list[tuple[str, dict[str, Any]]] = [(s, params) for s in wanted]

    collected: dict[str, BacktestResult] = {}
    if n_workers <= 1 or len(jobs) == 1:
        for job in jobs:
            label, res = _run_one(job)
            collected[label] = res
    else:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(jobs))) as ex:
            for label, res in ex.map(_run_one, jobs):
                collected[label] = res

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
                                    seed=seed)

    if result_dir is not None:
        save(results, table, rf, result_dir)
    return results, table


def save (results: dict[str, BacktestResult], table: pd.DataFrame,
          rf: pd.Series, result_dir: Path = RESULT_DIR)-> Path:
    """Write the run to CSV: returns, wealth, summary, diagnostics, weights.

    Args:
        results (dict[str, BacktestResult]): Per-strategy runs from `run_all`.
        table (pd.DataFrame): The summary table from `metrics.summarise`.
        rf (pd.Series): Monthly risk-free rate, saved alongside so the Sharpe
            ratios can be recomputed from the returns file alone.
        result_dir (Path): Output directory, created if absent. Defaults to
            `constant.RESULT_DIR`.

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
        WindowsPath('.../Result')
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


def _cli ()-> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Backtest the proposed model (historical and forward), '
                    'EPO and PPP on one universe.')
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
    parser.add_argument('--long-short', action='store_true',
                        help='let EPO, PPP and the proposed model go short, '
                             'i.e. run each paper as stated')
    parser.add_argument('--sr-benchmark', default='equal_weight',
                        choices=[*STRATEGIES, 'none'],
                        help='strategy whose Sharpe ratio the others are '
                             'tested against; "none" skips the test')
    parser.add_argument('--n-boot', type=int, default=4999,
                        help='bootstrap resamples for the Sharpe ratio test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--out', type=Path, default=RESULT_DIR)
    return parser.parse_args()


if __name__ == '__main__':
    args = _cli()
    _, summary_table = run_all(
        only=tuple(args.only) if args.only else None,
        skip=tuple(args.skip), n_workers=args.n_workers,
        cost_bps=args.cost_bps, epo_scale=args.epo_scale,
        epo_workers=args.epo_workers, n_samples=args.n_samples,
        proposed_quarterly=not args.monthly_proposed,
        long_only=not args.long_short, seed=args.seed, verbose=args.verbose,
        sr_benchmark=None if args.sr_benchmark == 'none' else args.sr_benchmark,
        n_boot=args.n_boot, result_dir=args.out)
    with pd.option_context('display.width', 160,
                           'display.float_format', '{:,.4f}'.format):
        print()
        print(summary_table)
    print(f'\nwritten to {args.out}')
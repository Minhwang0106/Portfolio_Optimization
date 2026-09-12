"""One calling convention for three models that do not share one.

`engine.backtest` needs a single callable `(date, tickers) -> pd.Series` indexed
by ticker. None of the three models offers that:

* `EPO.portfolio_weight` takes no universe at all -- it looks its own up by date
  -- and returns a bare array ordered by `EPO.correl[date].columns`. It is
  unnormalised by default, and sums to one under `long_only=True`.
* `PPP.portfolio_weight` returns a `(1, n_t)` array ordered by `self.ticker`,
  which is *narrower* than the list passed in and is only readable off the
  instance afterwards.
* `RIM_PortOp.weight` and `Icc_Mvo.weight` already return a labelled Series,
  and are the only ones that do.

The adapters here put all three on the Series convention and nothing else. They
do not change any model's weighting rule; the one thing they do decide is EPO's
scale, which the model genuinely leaves open -- see `normalise`.

`replay_weight` is the exception, and says so: it re-solves the proposed model
from a saved simulation under a breadth floor or a different objective, which is
how the matched-breadth rows are built without simulating again.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from constant import icc_weight_cap, IMPLIED_RETURN_DIR
from ..EPO.model import EPO
from ..PPP.model import PPP
from ..ICC_MVO.model import Icc_Mvo
from ..Proposed_Model.model import RIM_PortOp
from ..Proposed_Model.utils.port_optimize import port_weight, moment_port_weight

# What `replay_weight` can re-solve a dump under.
REPLAY_OBJECTIVES: tuple[str, ...] = ('crra', 'max_sharpe')
MONTHS_PER_YEAR: int = 12

# Below this the normalising denominator is treated as zero rather than divided
# by. A gross exposure this small means the solver returned essentially no
# position, and dividing by it turns rounding noise into a levered book.
_SCALE_FLOOR: float = 1e-12


def normalise (weight: pd.Series, how: str = 'gross')-> pd.Series:
    """Put a weight vector on a stated scale.

    Only EPO needs this, and by default EPO always does. `PPP` and `RIM_PortOp`
    both already sum to one -- the first because the standardized tilts cancel,
    the second because `port_weight` constrains them to -- as does EPO under
    `long_only=True`. Unconstrained, which is now EPO's default, it returns
    `Sigma_w^-1 @ signal / gamma`, whose magnitude is whatever the covariance
    matrix and the signal scale happen to imply. That number is not a portfolio
    weight in the same sense as the other two, and comparing a Sharpe ratio
    computed on it against theirs compares two different amounts of capital.

    Args:
        weight (pd.Series): Raw weights, indexed by ticker.
        how (str): One of

            * 'gross' -- scale to `sum(|w|) == 1`, i.e. one unit of capital
              deployed across the book. The right choice for a long-short
              vector, whose net exposure is not a meaningful denominator.
            * 'net' -- scale to `sum(w) == 1`, i.e. fully invested. Only
              meaningful when the net exposure is reliably away from zero.
            * 'none' -- return the weights untouched.

    Returns:
        pd.Series: Scaled weights. All zeros if the denominator is below
            `_SCALE_FLOOR`, which is a flat book rather than a levered one.

    Raises:
        ValueError: If `how` is not one of the three above.

    Example:
        >>> w = pd.Series({'A': 2.0, 'B': -1.0})
        >>> normalise(w, 'gross')
        A    0.666667
        B   -0.333333
        dtype: float64
    """
    if how == 'none':
        return weight.astype(float)
    if how == 'gross':
        scale: float = float(weight.abs().sum())
    elif how == 'net':
        scale = float(weight.sum())
    else:
        raise ValueError(f"how must be 'gross', 'net' or 'none', got {how!r}")
    if abs(scale) < _SCALE_FLOOR:
        warnings.warn(f'normalise({how!r}): denominator {scale:.3e} is at the '
                      'floor; returning a flat book', RuntimeWarning)
        return pd.Series(0.0, index=weight.index)
    return weight/scale


def epo_weight (date: pd.Timestamp, tickers: list[str]|None = None,
                how: str = 'gross')-> pd.Series:
    """EPO weights at one formation date, labelled and scaled.

    `EPO.Config()` must have run first; it precomputes the per-date vol,
    correlation and signal inputs over the whole of `constant.testing_period` in
    a process pool, and `EPO.portfolio_weight` raises rather than doing it
    lazily. Whether the book is long-only was fixed there, by
    `Config(long_only=...)`, so there is nothing to pass per date.

    Args:
        date (pd.Timestamp): Formation date. Keyed into `EPO.correl` as
            `str(date)`, which is the form `EPO.Config` stored -- so it has to
            be a date `Config` covered, i.e. one of `constant.testing_period`.
        tickers (list[str] | None): Accepted for interface uniformity and
            intersected with EPO's own universe if given. EPO reads
            `applicable_ticker.csv` itself in `EPO.utils.compute_date`, so under
            the default backtest configuration this is the same list and the
            intersection is a no-op.
        how (str): Scale, passed to `normalise`. Defaults to 'gross', which is
            what makes an unconstrained book one unit of deployed capital.
            Under `EPO.long_only` the model already returns a book summing to
            one, so all three settings agree up to the renormalisation that
            follows the intersection with `tickers`.

    Returns:
        pd.Series: One weight per ticker EPO priced at `date`.

    Raises:
        RuntimeError: If `EPO.Config()` has not been called.
        KeyError: If `Config` did not cover `date`.
        ValueError: If the long-only solve does not converge.

    Example:
        >>> EPO.Config()
        >>> w = epo_weight(pd.Timestamp('2020-03-31'))
        >>> float(w.abs().sum())
        1.0
    """
    key: str = str(date)
    raw: np.ndarray = EPO(key).portfolio_weight()
    # The column order of the correlation matrix is the only thing that says
    # which ticker each element belongs to; `portfolio_weight` returns a bare
    # array, so reading it against any other list scrambles the cross-section.
    weight: pd.Series = pd.Series(raw, index=EPO.correl[key].columns,
                                  dtype=float)
    weight.index.name = 'ticker'
    if tickers is not None:
        keep: list[str] = [t for t in weight.index if t in set(tickers)]
        weight = weight.loc[keep]
    return normalise(weight, how)


def ppp_weight (model: PPP, date: pd.Timestamp, tickers: list[str],
                n_month: int = 60, long_only: bool = True)-> pd.Series:
    """Parametric-policy weights at one formation date, labelled.

    Args:
        model (PPP): A live `PPP` instance. Reused across dates rather than
            rebuilt, since it also accumulates `theta` and `weight` per date and
            those are worth keeping for inspection.
        date (pd.Timestamp): Formation date; a month end.
        tickers (list[str]): Candidate universe. `find_theta` narrows it to the
            names with a complete characteristic record over the whole
            estimation window, so the result is generally shorter.
        n_month (int): Estimation window length in months. Defaults to 60.
        long_only (bool): Truncate the short leg and renormalise, fitting theta
            under the same constraint. Defaults to True.

    Returns:
        pd.Series: One weight per surviving ticker, summing to one.
            Non-negative under `long_only`; otherwise individual weights may be
            negative, the policy being unconstrained.

    Example:
        >>> PPP.Config()
        >>> w = ppp_weight(PPP(), pd.Timestamp('2020-03-31'), ['AAPL', 'MSFT'])
        >>> float(w.sum())
        1.0
    """
    raw: np.ndarray = model.portfolio_weight(date, tickers, n_month,
                                             long_only=long_only)
    # `(1, n_t)`, not `(n_t,)`, and ordered by `model.ticker` -- which is set as
    # a side effect of the call above and is not the list passed in.
    return pd.Series(np.asarray(raw).reshape(-1), index=model.ticker,
                     dtype=float).rename_axis('ticker')


def icc_mvo_weight (date: pd.Timestamp, tickers: list[str],
                    cap: float = icc_weight_cap, long_only: bool = True
                    )-> pd.Series:
    """Bielstein-Hanauer ICC-MVO weights at one formation date.

    A pass-through, like `proposed_weight`: `Icc_Mvo.weight` already returns a
    ticker-indexed Series summing to one. `Icc_Mvo.Config()` should have run
    first, so the panels are built once rather than on the first date reached.

    Args:
        date (pd.Timestamp): Formation date.
        tickers (list[str]): Candidate universe, narrowed by `Icc_Mvo` to the
            names with a solvable implied cost of capital, 12-1 momentum and a
            complete `constant.icc_cov_month` return history.
        cap (float): Largest weight on one name. Defaults to
            `constant.icc_weight_cap`, B&H's 5%; 1 removes it.
        long_only (bool): B&H's long-only book. False solves the unconstrained
            tangency portfolio instead, which takes no cap. Defaults to True.

    Returns:
        pd.Series: One weight per surviving ticker, summing to one. Names the
            optimiser left under `constant.icc_weight_floor` are exactly 0.

    Note:
        **Lookahead.** The `constant.icc_explicit_years` (11) explicit earnings
        years the implied cost of capital is solved on are the *realised* ones,
        so the book formed at `t` has seen EPS as far ahead as the panel runs --
        the same window `proposed_ex_post` elicits its moments from. It is a
        benchmark under perfect earnings foresight, not a strategy -- see
        `Icc_Mvo`.

    Example:
        >>> Icc_Mvo.Config()
        >>> w = icc_mvo_weight(pd.Timestamp('2020-06-30'), tickers)
        >>> float(w.sum()), float(w.max()) <= 0.05
        (1.0, True)
    """
    return Icc_Mvo(date, tickers).weight(cap=cap, long_only=long_only)


def proposed_weight (date: pd.Timestamp, tickers: list[str],
                     n_samples: int = 10000, n_lags: int = 4,
                     ex_post: bool = False, common_theta: bool = True,
                     seed: int|None = None, long_only: bool = True,
                     dump_dir: Path|None = None
                     )-> pd.Series:
    """Residual-income model weights at one formation date.

    The only adapter that is close to a pass-through: `RIM_PortOp.weight`
    already returns a ticker-indexed Series summing to one.

    Args:
        date (pd.Timestamp): Formation date.
        tickers (list[str]): Candidate universe, narrowed by `RIM_PortOp` to the
            names with at least `constant.min_char_obs` complete quarters.
        n_samples (int): Simulated paths per ticker. Defaults to 10000.
        n_lags (int): Lags the persistence fit reads. Defaults to 4.
        ex_post (bool): Elicit the unconditional moments from the *realised*
            future window rather than from the training window. Defaults to
            False.

            **This is lookahead and the returns it produces are not
            tradeable.** `take_training_data` hands `RIM_PortOp` a future window
            running `constant.n_quarter_ahead` quarters past the formation date,
            and under `ex_post=True` the mean and variance each characteristic
            is simulated around come from that window -- so the portfolio formed
            at `t` has already seen the accounting figures as far ahead as the
            panel runs: about 45 quarters at the first formation date, 2 at the
            last. Its purpose is to decompose the model's error: run against
            `ex_post=False` it separates how much of the shortfall is the
            simulation machinery and how much is simply not knowing the future
            moments. Read it as an upper bound on what perfect moment
            forecasting would buy, never as a strategy.

            Named `ex_post`, not `forward`, to match the manuscript. "Forward"
            reads in finance as a forward-*looking* estimate built from
            information available now, which is implementable -- the opposite
            of what this flag does.

        common_theta (bool): Pool the decay estimate across the universe.
            Defaults to True.
        seed (int | None): Seed for the copula draw. Pass one for a reproducible
            backtest; `run` derives a per-date seed from it so that two
            formation dates do not draw the same paths. Defaults to None.
        long_only (bool): Bound weights to [0, 1]. Defaults to True.
        dump_dir (Path | None): Write this date's simulated implied returns
            here before optimising, as
            `{historical|ex_post}_{YYYY-MM-DD}.csv` -- rows the simulation
            index, columns the ticker, which is the transpose of what
            `RIM_PortOp.joint_return` returns and the orientation a
            column-masking subsample test wants. Defaults to None, i.e. no
            file.

            Costs no extra simulation. `RIM_PortOp.weight` computes the same
            frame internally and discards it; this path keeps a reference and
            calls `port_weight` on it directly, so the weights are identical
            to the undumped call given the same seed.

    Returns:
        pd.Series: One weight per ticker, summing to one. A ticker whose
            simulations were all non-finite gets exactly 0.

    Example:
        >>> RIM_PortOp.Config()
        >>> w = proposed_weight(pd.Timestamp('2020-03-31'), tickers, seed=0)
        >>> float(w.sum())
        1.0
    """
    model: RIM_PortOp = RIM_PortOp(date, tickers)
    if dump_dir is None:
        return model.weight(n_samples=n_samples, n_lags=n_lags, ex_post=ex_post,
                            common_theta=common_theta, seed=seed,
                            long_only=long_only).astype(float)

    # `weight` is `joint_return` then `port_weight`. Inlining the two is what
    # makes the dump free -- calling `joint_return` separately and then
    # `weight` would simulate the whole universe twice.
    return_df: pd.DataFrame = model.joint_return(
        n_samples=n_samples, n_lags=n_lags, ex_post=ex_post,
        common_theta=common_theta, seed=seed)
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    variant: str = 'ex_post' if ex_post else 'historical'
    path: Path = dump_dir / f'{variant}_{pd.Timestamp(date).date()}.csv'
    # Transposed on the way out only. `port_weight` reads the frame in
    # `joint_return`'s own (ticker, simulation) orientation and would silently
    # read a transposed one as a universe of 10,000 names.
    return_df.T.to_csv(path, index_label='simulation')
    return port_weight(return_df, long_only=long_only).astype(float)


def replay_weight (date: pd.Timestamp, tickers: list[str], variant: str,
                   objective: str = 'crra', n_eff: pd.Series|None = None,
                   rf: pd.Series|None = None,
                   dump_dir: Path = IMPLIED_RETURN_DIR)-> pd.Series:
    """The proposed model's weights, re-solved from a saved simulation.

    `proposed_weight(dump_dir=...)` writes each formation date's simulated
    implied returns; this reads them back and solves again under a breadth
    floor or a different objective. Rows built this way share their draws with
    each other and with the run that wrote the dump, so they differ in nothing
    but the constraint or the objective. No copula is fitted and nothing is
    simulated -- seconds a date rather than minutes.

    Args:
        date (pd.Timestamp): Formation date. A dump must exist for it.
        tickers (list[str]): Unused. The dump already holds the universe the
            model screened at `date`, and screening it again here would change
            the problem the rows are meant to share. Present for the engine's
            signature.
        variant (str): `'historical'` or `'ex_post'`, the dump's prefix.
        objective (str): `'crra'` solves `port_weight`, the model as specified;
            `'max_sharpe'` solves `moment_port_weight`, B&H's rule on the
            simulation's mean and covariance. Defaults to `'crra'`.
        n_eff (pd.Series | None): Effective-N floor per formation date, from
            `run.effective_n_target`. None leaves breadth free. Defaults to
            None.
        rf (pd.Series | None): Monthly risk-free rate from `data.risk_free`.
            Under `'max_sharpe'` the formation month's rate is compounded to a
            year and subtracted from the annual implied returns, as `Icc_Mvo`
            does for B&H. Required there, unused otherwise. Defaults to None.
        dump_dir (Path): Where the dumps are. Defaults to
            `constant.IMPLIED_RETURN_DIR`.

    Returns:
        pd.Series: One weight per ticker in the dump, summing to one.
            Long-only whatever the run's `long_only` says: the floor is defined
            on a long-only book, and the comparators it is read off are one.

    Raises:
        FileNotFoundError: If there is no dump for `date`.
        ValueError: If `objective` is unknown, if `n_eff` holds no target for
            `date` (the comparator had not formed a book yet), if
            `'max_sharpe'` is asked for without `rf`, or if the solve fails.
            `engine.backtest` records any of these and holds the previous book.

    Example:
        >>> w = replay_weight(pd.Timestamp('2020-06-30'), [], 'historical',
        ...                   n_eff=pd.Series({pd.Timestamp('2020-06-30'): 70.0}))
        >>> round(1/float((w**2).sum()), 1) >= 70.0
        True
    """
    if objective not in REPLAY_OBJECTIVES:
        raise ValueError(f'objective must be one of {REPLAY_OBJECTIVES}; '
                         f'got {objective!r}')
    date = pd.Timestamp(date)
    # (simulation, ticker) on disk; both solvers read (ticker, simulation).
    draws: pd.DataFrame = pd.read_csv(
        Path(dump_dir)/f'{variant}_{date.date()}.csv', index_col=0)
    target: float|None = None
    if n_eff is not None:
        target = float(n_eff.get(date, np.nan))
        if not np.isfinite(target):
            raise ValueError(f'{date.date()}: no effective-N target -- the '
                             'comparator had not formed a book yet')
    if objective == 'crra':
        return port_weight(draws.T, n_eff=target).astype(float)
    if rf is None:
        raise ValueError("objective 'max_sharpe' needs the risk-free rate")
    rf_year: float = (1.0+float(rf.loc[:date].iloc[-1]))**MONTHS_PER_YEAR-1.0
    return moment_port_weight(draws.T, rf=rf_year, n_eff=target).astype(float)


def equal_weight (date: pd.Timestamp, tickers: list[str])-> pd.Series:
    """Equal weights over the candidate universe: the benchmark the tilts start from.

    Worth running alongside the three models rather than quoting from an index
    provider, because it is the same universe on the same dates -- so the
    difference between it and a model is the weighting rule alone, with the
    stock selection screen held fixed. It is also literally `PPP`'s benchmark
    `w_bar`, which makes PPP's contribution readable as the gap between the two.

    Args:
        date (pd.Timestamp): Formation date. Unused; present so the signature
            matches the other adapters.
        tickers (list[str]): Universe to spread the capital over.

    Returns:
        pd.Series: `1/N` on every ticker.

    Raises:
        ValueError: If `tickers` is empty.

    Example:
        >>> equal_weight(pd.Timestamp('2020-03-31'), ['A', 'B'])
        ticker
        A    0.5
        B    0.5
        dtype: float64
    """
    if not tickers:
        raise ValueError(f'{date.date()}: empty universe')
    return pd.Series(1.0/len(tickers), index=pd.Index(tickers, name='ticker'))
import pandas as pd
import numpy as np
from constant import (monthly_FF5_path,
                       daily_price_path,
                       monthly_price_path,
                       applicable_ticker_path,
                       risk_aversion,
                       trading_day_per_year,
                       epo_theta,
                       epo_shrinkage_grid)
from .functions import (cal_return, cal_excess_return, cov_window,
                         cross_sectional_momentum, _long_only_weight)

# populated once per worker process by init_worker, so each process builds
# daily_r/monthly_er/applicable_ticker only once instead of per-date
_worker_data: dict = {}

# Fewest surviving names `compute_date` will build a covariance matrix from. A
# date this thin is a data problem rather than a small universe -- the screen
# below demands a complete window, so a handful of survivors means the panel is
# short, not that the market was.
_MIN_UNIVERSE: int = 10


def init_worker(n_day, day_per_name, n_month, long_only: bool = True) -> None:
    """Load and cache shared data once per worker process for `compute_date`.

    Populates the module-level `_worker_data` dict with raw daily returns,
    monthly excess returns, applicable tickers, and the window parameters, so
    `compute_date` can reuse them across many dates without re-loading from
    disk each call. Intended as the `initializer` for a `ProcessPoolExecutor`
    (or called directly for serial execution).

    Daily returns are raw and monthly ones are excess, per §2 of
    `Instruction/epo_equity4_implementation.md`: the risk model does not care
    about a rate common to every name, while the signal and the realised P&L do.

    Args:
        n_day (int): Floor for the trailing daily covariance window.
        day_per_name (int): Daily observations required per name; the window is
            `max(n_day, day_per_name * n_t)`. See `functions.cov_window`.
        n_month (int): Trailing window (months) for the momentum signal.
        long_only (bool): Which book `compute_date` solves at every grid point,
            and therefore which one `EPO._select_w` scores the shrinkage
            against. True (the default) solves `x >= 0, sum(x) == 1` by SLSQP;
            False takes the unconstrained closed form. One or the other, never
            both -- a run holds one EPO book and the shrinkage has to be ranked
            on the book actually held.

    Returns:
        None. Mutates the module-level `_worker_data` dict.

    Example:
        >>> init_worker(n_day=120, day_per_name=2, n_month=12)
        >>> 'daily_r' in _worker_data
        True
    """
    daily_r: pd.DataFrame = cal_return(daily_price_path)
    monthly_er: pd.DataFrame = cal_excess_return(monthly_price_path, monthly_FF5_path)
    applicable_ticker: pd.DataFrame = pd.read_csv(applicable_ticker_path)
    applicable_ticker['date'] = pd.to_datetime(applicable_ticker['date'])
    applicable_ticker.set_index('date', inplace=True)
    _worker_data.update(daily_r=daily_r, monthly_er=monthly_er,
                         applicable_ticker=applicable_ticker, n_day=n_day,
                         day_per_name=day_per_name, n_month=n_month,
                         long_only=long_only)


def compute_date(date):
    """Compute volatility, correlation, the XSMOM signal, and w-selection
    inputs for one date.

    The paper's Equity risk model and signal (Table 1, p.44; §III.A pp.18-21),
    spelled out step by step in `Instruction/epo_equity4_implementation.md`.
    Uses data cached by `init_worker` in `_worker_data`.

    Starts from the tickers applicable on `date`, drops any without a complete
    estimation window, then computes: an equal-weighted covariance of the
    trailing `cov_window(n_t)` daily returns annualized by
    `constant.trading_day_per_year`, the volatilities and correlations it
    factors into, the signal `sigma * XSMOM` over the trailing `n_month`-month
    cumulative excess return, the EPO weight at every candidate shrinkage in
    `constant.epo_shrinkage_grid`, and next month's realised excess return.

    Which weight that is comes from `init_worker`'s `long_only`: the long-only
    solve on the simplex, or the unconstrained closed form. Exactly one book is
    built, because exactly one is held -- see `EPO.Config`.

    The candidates and the realised return feed `EPO._select_w`'s out-of-sample
    search for the shrinkage actually used (`constant.epo_shrinkage`'s
    docstring); they are computed here, once per date per grid point, rather
    than inside that search, because the risk model and signal they need are
    already in hand -- and because the search is `O(dates^2 * grid)` in
    *scoring* but only `O(dates * grid)` in *solving*, so caching the solve
    here is what keeps the long-only variant affordable.

    The correlation matrix returned is the *raw* one. `constant.epo_theta`'s 5%
    pre-shrink is applied to the candidates below and again in
    `EPO.portfolio_weight`, so callers reading `EPO.correl` see the estimate
    rather than the shrunk matrix the solve actually uses.

    Args:
        date: A date present in the applicable-ticker table's index.

    Returns:
        tuple[str, pd.Series, pd.DataFrame, pd.Series, dict[float, pd.Series], pd.Series | None] | None:
            (date_str, std, correl, signal, candidates, realized), or None if
            the date has no applicable ticker list or fewer than
            `_MIN_UNIVERSE` names survive the completeness screen. `candidates`
            maps each grid point to the weight vector it implies under
            `init_worker`'s `long_only`, indexed like `signal`. `realized` is
            next month's excess return over the same tickers, or None at the
            last formation date, where there is no next month to score against.

    Example:
        >>> init_worker(n_day=120, day_per_name=2, n_month=12)
        >>> result = compute_date('2020-03-31')
        >>> date_str, std, correl, signal, cand, realized = result
    """
    daily_r = _worker_data['daily_r']
    monthly_er = _worker_data['monthly_er']
    applicable_ticker = _worker_data['applicable_ticker']
    n_day = _worker_data['n_day']
    day_per_name = _worker_data['day_per_name']
    n_month = _worker_data['n_month']
    long_only = _worker_data['long_only']

    ticker = applicable_ticker.loc[date, 'tickers']
    if not isinstance(ticker, str):
        return None
    ticker = ticker.split(',')

    # The window is sized off the candidate pool, before the screen below
    # narrows it -- see `cov_window` for why that ordering is what makes
    # `K >= day_per_name * n_t` hold without iterating.
    n_obs: int = cov_window(len(ticker), n_day, day_per_name)
    daily_win: pd.DataFrame = daily_r.loc[:date, ticker].tail(n_obs)
    mom_win: pd.DataFrame = monthly_er.loc[:date, ticker].tail(n_month)
    if len(daily_win) < n_obs or len(mom_win) < n_month:
        return None

    # A name survives only with a complete estimation window and a return to be
    # scored on next month. Dropping the incomplete ones is not tidiness: a
    # single NaN column makes `cov` pairwise-complete, which silently returns a
    # matrix that need not be positive semi-definite at all.
    future: pd.Index = monthly_er.index[monthly_er.index > date]
    complete: pd.Series = (daily_win.notna().all() & mom_win.notna().all())
    if len(future):
        complete &= monthly_er.loc[future.min(), ticker].notna()
    cols: list[str] = [t for t in ticker if bool(complete[t])]
    if len(cols) < _MIN_UNIVERSE:
        return None
    daily_win, mom_win = daily_win[cols], mom_win[cols]

    # Equal-weighted covariance, annualized. `DataFrame.cov` is ddof=1, which is
    # exactly the 1/(K-1) of fn.17 (p.20) -- do not "fix" it.
    cov: np.ndarray = daily_win.cov().to_numpy()*trading_day_per_year
    sigma: np.ndarray = np.sqrt(np.diag(cov))
    std: pd.Series = pd.Series(sigma, index=cols)
    correl: pd.DataFrame = pd.DataFrame(cov/np.outer(sigma, sigma),
                                         index=cols, columns=cols)

    # Equity 4's signal: cross-sectional momentum scaled by volatility. No
    # skip-month -- the paper writes r[t-12,t] straight through (§7).
    cum_return: pd.Series = (1+mom_win).prod(axis=0)-1
    signal: pd.Series = std*cross_sectional_momentum(cum_return)

    # The weight at every candidate shrinkage, computed once per date so
    # `EPO._select_w` only ever does arithmetic on cached vectors rather than
    # re-solving anything. Under `long_only` that is the SLSQP solve on the
    # simplex; otherwise eq. (16) of the paper, `Sigma_w^-1 @ signal / gamma`.
    # Only the configured book is built: the other one is never held, and a `w`
    # ranked on it says close to nothing about this one (the two correlate ~0
    # at every `w` below 0.75).
    n_stock: int = len(cols)
    identity: np.ndarray = np.identity(n_stock)
    vol_arr: np.ndarray = np.diag(sigma)
    shrink_correl: np.ndarray = correl.to_numpy()*epo_theta+(1-epo_theta)*identity
    signal_arr: np.ndarray = signal.to_numpy()
    candidates: dict[float, pd.Series] = {}
    for w in epo_shrinkage_grid:
        sigma_w: np.ndarray = vol_arr@((1-w)*shrink_correl+w*identity)@vol_arr
        x_w: np.ndarray = (
            _long_only_weight(sigma_w, signal_arr, risk_aversion) if long_only
            else np.linalg.solve(sigma_w, signal_arr)/risk_aversion)
        candidates[w] = pd.Series(x_w, index=cols)

    # Next month's realised excess return for this date's tickers -- what
    # holding any candidate `x_w` from `date` into the following month would
    # actually have earned.
    realized: pd.Series|None = (
        monthly_er.loc[future.min(), cols] if len(future) else None)

    return str(date), std, correl, signal, candidates, realized

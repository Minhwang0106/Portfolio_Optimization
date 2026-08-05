import pandas as pd
import numpy as np
from math import sqrt
from constant import (daily_FF5_path,
                       monthly_FF5_path,
                       daily_price_path,
                       monthly_price_path,
                       applicable_ticker_path,
                       risk_rolling,
                       risk_aversion,
                       epo_theta,
                       epo_shrinkage_grid)
from .excess_return import cal_excess_return

# populated once per worker process by init_worker, so each process builds
# daily_er/monthly_er/overlap_r/applicable_ticker only once instead of per-date
_worker_data: dict = {}


def init_worker(com, correl_com, n_day, n_month) -> None:
    """Load and cache shared data once per worker process for `compute_date`.

    Populates the module-level `_worker_data` dict with daily/monthly excess
    returns, rolling overlapping daily returns, applicable tickers, and the
    EWM/window parameters, so `compute_date` can reuse them across many
    dates without re-loading from disk each call. Intended as the
    `initializer` for a `ProcessPoolExecutor` (or called directly for serial
    execution).

    Args:
        com: EWM center-of-mass used for the volatility calculation.
        correl_com: EWM center-of-mass used for the correlation calculation.
        n_day (int): Trailing window (days) for vol/correlation.
        n_month (int): Trailing window (months) for the TSMOM signal.

    Returns:
        None. Mutates the module-level `_worker_data` dict.

    Example:
        >>> init_worker(com=60, correl_com=60, n_day=252, n_month=12)
        >>> 'daily_er' in _worker_data
        True
    """
    daily_er: pd.DataFrame = cal_excess_return(daily_price_path, daily_FF5_path)
    monthly_er: pd.DataFrame = cal_excess_return(monthly_price_path, monthly_FF5_path)
    overlap_r: pd.DataFrame = daily_er.rolling(risk_rolling).sum()
    applicable_ticker: pd.DataFrame = pd.read_csv(applicable_ticker_path)
    applicable_ticker['date'] = pd.to_datetime(applicable_ticker['date'])
    applicable_ticker.set_index('date', inplace=True)
    _worker_data.update(daily_er=daily_er, monthly_er=monthly_er, overlap_r=overlap_r,
                         applicable_ticker=applicable_ticker, com=com,
                         correl_com=correl_com, n_day=n_day, n_month=n_month)


def compute_date(date):
    """Compute volatility, correlation, TSMOM signal, and w-selection inputs
    for one date.

    Uses data cached by `init_worker` in `_worker_data`. Restricts the
    universe to the tickers applicable on `date` (per the applicable-ticker
    table), then computes: annualized EWM volatility over the trailing
    `n_day` days, an EWM correlation matrix from overlapping rolling
    returns, a time-series-momentum (TSMOM) signal (sign of trailing
    `n_month`-month compounded return, scaled by volatility and a fixed 10%
    target), the unconstrained closed-form EPO weight at every candidate
    shrinkage in `constant.epo_shrinkage_grid`, and next month's realised
    excess return. The last two feed `EPO._select_w`'s out-of-sample search
    for the shrinkage actually used (`constant.epo_shrinkage`'s docstring);
    they are computed here, once per date, rather than inside that search,
    because the risk model and signal it needs are already in hand.

    Args:
        date: A date present in the applicable-ticker table's index.

    Returns:
        tuple[str, pd.Series, pd.DataFrame, pd.Series, dict[float, pd.Series], pd.Series | None] | None:
            (date_str, std, correl, tsmom, candidates, realized) if the date
            has an applicable ticker list, else None if no tickers are
            available for that date. `candidates` maps each grid point to the
            unconstrained weight vector it implies, indexed like `tsmom`.
            `realized` is next month's excess return over the same tickers,
            or None at the last formation date, where there is no next month
            to score against.

    Raises:
        ValueError: If the computed `std` or `tsmom` isn't a pandas Series
            (indicates a data-shape problem upstream).

    Example:
        >>> init_worker(com=60, correl_com=60, n_day=252, n_month=12)
        >>> result = compute_date('2020-03-31')
        >>> date_str, std, correl, tsmom, candidates, realized = result
    """
    daily_er = _worker_data['daily_er']
    monthly_er = _worker_data['monthly_er']
    overlap_r = _worker_data['overlap_r']
    applicable_ticker = _worker_data['applicable_ticker']
    com = _worker_data['com']
    correl_com = _worker_data['correl_com']
    n_day = _worker_data['n_day']
    n_month = _worker_data['n_month']

    ticker = applicable_ticker.loc[date, 'tickers']
    if not isinstance(ticker, str):
        return None
    ticker = ticker.split(',')

    std = daily_er.loc[:date, ticker].tail(n_day).ewm(
        com=com, adjust=False).std().asof(date) * sqrt(n_day)
    if not isinstance(std, pd.Series):
        raise ValueError('std should be pandas Series type')

    cov_all = overlap_r.loc[:date, ticker].tail(n=n_day).ewm(
        com=correl_com, adjust=False).cov()
    cov = cov_all.loc[cov_all.index.get_level_values(0).max()]
    sigma = np.sqrt(np.diag(cov))
    D_inv = np.diag(1 / sigma)
    correl = pd.DataFrame(D_inv @ cov.to_numpy() @ D_inv,
                           index=cov.index, columns=cov.columns)

    growth: pd.DataFrame = 1 + monthly_er
    sign = np.sign(growth.loc[:date, ticker].tail(n_month).prod(axis=0) - 1)
    tsmom = sign * std * 0.1
    if not isinstance(tsmom, pd.Series):
        raise ValueError('TSMOM should be pandas Series')

    # Unconstrained closed-form weight at every candidate shrinkage -- eq. (16)
    # of the paper, `Sigma_w^-1 @ signal / gamma`, computed once per date so
    # `EPO._select_w` only ever does arithmetic on cached vectors rather than
    # re-solving anything.
    n_stock: int = len(ticker)
    identity: np.ndarray = np.identity(n_stock)
    correl_arr: np.ndarray = correl.to_numpy()
    vol_arr: np.ndarray = np.diag(std.to_numpy())
    shrink_correl: np.ndarray = correl_arr*epo_theta+(1-epo_theta)*identity
    candidates: dict[float, pd.Series] = {}
    for w in epo_shrinkage_grid:
        sigma_w: np.ndarray = vol_arr@((1-w)*shrink_correl+w*identity)@vol_arr
        x_w: np.ndarray = np.linalg.solve(sigma_w, tsmom.to_numpy())/risk_aversion
        candidates[w] = pd.Series(x_w, index=ticker)

    # Next month's realised excess return for this date's tickers -- what
    # holding any candidate `x_w` from `date` into the following month would
    # actually have earned.
    future: pd.Index = monthly_er.index[monthly_er.index > date]
    realized: pd.Series|None = (
        monthly_er.loc[future.min(), ticker] if len(future) else None)

    return str(date), std, correl, tsmom, candidates, realized

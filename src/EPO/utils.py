import pandas as pd
import numpy as np
from math import sqrt
from constant import (daily_FF5_path,
                       monthly_FF5_path,
                       daily_price_path,
                       monthly_price_path,
                       applicable_ticker_path,
                       risk_rolling)
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
    """Compute volatility, correlation, and TSMOM signal for one date.

    Uses data cached by `init_worker` in `_worker_data`. Restricts the
    universe to the tickers applicable on `date` (per the applicable-ticker
    table), then computes: annualized EWM volatility over the trailing
    `n_day` days, an EWM correlation matrix from overlapping rolling
    returns, and a time-series-momentum (TSMOM) signal (sign of trailing
    `n_month`-month compounded return, scaled by volatility and a fixed 10%
    target).

    Args:
        date: A date present in the applicable-ticker table's index.

    Returns:
        tuple[str, pd.Series, pd.DataFrame, pd.Series] | None:
            (date_str, std, correl, tsmom) if the date has an applicable
            ticker list, else None if no tickers are available for that date.

    Raises:
        ValueError: If the computed `std` or `tsmom` isn't a pandas Series
            (indicates a data-shape problem upstream).

    Example:
        >>> init_worker(com=60, correl_com=60, n_day=252, n_month=12)
        >>> result = compute_date('2020-03-31')
        >>> date_str, std, correl, tsmom = result
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

    return str(date), std, correl, tsmom

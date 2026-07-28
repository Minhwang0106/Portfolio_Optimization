import pandas as pd
import numpy as np
from pathlib import Path
from constant import(
    monthly_FF5_path, monthly_price_path
)
from ...PPP.input_generator import read_csv_file

# Column names as Ken French publishes them, mapped to the namespaced names
# used everywhere downstream. The prefix is not cosmetic: 'RF' is also the
# ticker of Regions Financial, so unprefixed factor columns make `data['RF']`
# return two columns and the risk-free rate silently becomes a 2-D array.
FF5_RENAME: dict[str, str] = {
    'Mkt-RF': 'ff5_mkt_rf', 'SMB': 'ff5_smb', 'HML': 'ff5_hml',
    'RMW': 'ff5_rmw', 'CMA': 'ff5_cma', 'RF': 'ff5_rf',
}
FF5_FACTOR: list[str] = ['ff5_mkt_rf', 'ff5_smb', 'ff5_hml', 'ff5_rmw', 'ff5_cma']
RF: str = 'ff5_rf'
FF5_COMPONENT: list[str] = FF5_FACTOR + [RF]
MONTH_PER_QUARTER: int = 3
MIN_OBS: int = len(FF5_COMPONENT) + 12

def generator (ff5_path:Path=monthly_FF5_path,
               price_path:Path=monthly_price_path)->pd.DataFrame:
    """Join the FF5 factor file and monthly stock returns on one date index.

    Both sides are put in *decimal* units: Ken French publishes the factors and
    RF in percent (-0.39 means -0.39%), so they are divided by 100, while
    `adj_close.pct_change()` is already decimal and is left alone.

    Returns are computed within each ticker. A plain `pct_change()` on the
    (ticker, date) panel would divide the first price of one ticker by the last
    price of the one before it, which contaminates 566 of the 567 tickers in
    `monthly_price.csv`.

    Args:
        ff5_path (Path): Fama-French 5-factor CSV with a YYYYMM 'date' column
            and the columns of `FF5_RENAME`, in percent. Defaults to
            `constant.monthly_FF5_path`.
        price_path (Path): Monthly price panel CSV, indexed (ticker, date), with
            an 'adj_close' column. Defaults to `constant.monthly_price_path`.

    Returns:
        pd.DataFrame: Indexed by month-end date, with the six `FF5_COMPONENT`
            columns (renamed per `FF5_RENAME`) followed by one column of simple
            returns per ticker. Dates covered by only one of the two files are
            kept, with NaN on the other side; `re_cal` drops them per ticker.
    """
    ff5_df: pd.DataFrame = pd.read_csv(ff5_path)
    ff5_df['date'] = ff5_df['date'].apply(lambda x: str(x))
    ff5_df['date'] = ff5_df['date'].apply(lambda x: f'{x[:4]}-{x[4::]}-01')
    ff5_df['date'] = pd.to_datetime(ff5_df['date'])+pd.offsets.MonthEnd()
    ff5_df.set_index('date',inplace=True)
    ff5_df /= 100
    ff5_df.rename(columns=FF5_RENAME,inplace=True)

    price_df: pd.DataFrame = read_csv_file(price_path)
    return_df: pd.DataFrame = price_df['adj_close'].groupby(
        level='ticker').pct_change().unstack(level='ticker')

    df: pd.DataFrame = pd.concat([ff5_df,return_df],axis=1).sort_index()
    return df

def re_cal (data:pd.DataFrame, tickers:list[str], expectation:str='mean',
            factor_window:int|None=None, scale:int=MONTH_PER_QUARTER,
            min_obs:int=MIN_OBS)->pd.Series:
    """Cost of equity per ticker from an FF5 regression, scaled to a quarter.

    For each ticker the monthly excess return is regressed on the five factors
    over that ticker's own sample::

        r_it - rf_t = a_i + b_i' f_t + e_it

    and the cost of equity for the next period is the risk-free rate at the
    formation date plus the priced risk::

        re_i = rf_T + b_i' E[f]

    The intercept is dropped: alpha is the part of the return the factor model
    does not price, so it has no place in a required return. `rf_T` is the last
    observed risk-free rate under either `expectation` -- it is observable at
    the formation date, so there is nothing to estimate; only the premia `E[f]`
    are estimated.

    Each ticker is fitted on its own non-missing rows rather than on rows where
    every requested ticker is present. A joint `dropna()` over the 567-ticker
    panel leaves 16 of 775 months, because the sample is then bounded by the
    youngest listing.

    Args:
        data (pd.DataFrame): Output of `generator`, i.e. indexed by month-end
            date with the `FF5_COMPONENT` columns and one column per ticker, all
            in decimal units. Must be truncated at the formation date by the
            caller -- this function reads its last row, so any later row leaks.
        tickers (list[str]): Tickers to price. Every one must be a column of
            `data`.
        expectation (str): How `E[f]` is formed. 'mean' averages the factor
            rows, which is the premium the regression prices and what a cost of
            capital is normally built on. 'last' uses the single most recent
            factor row instead; monthly factor returns are near serially
            uncorrelated, so that reads one month's noise as next quarter's
            premium and turns the whole cross-section negative after any down
            month (-57% per quarter on average at 2020-03-31, +48% one month
            later). Defaults to 'mean'.
        factor_window (int | None): Trailing months to average when
            `expectation='mean'`, or None for the full factor history back to
            1963. A short window buys responsiveness at the cost of the same
            noise 'last' suffers from. Ignored when `expectation='last'`.
            Defaults to None.
        scale (int): Months per output period, applied as a simple multiple.
            Defaults to 3, i.e. a monthly rate stated per quarter. Compounding
            instead, `(1 + re) ** scale - 1`, differs by less than 10bp at these
            magnitudes.
        min_obs (int): Months a ticker needs before it is fitted. Defaults to
            `MIN_OBS` (17), five factors plus an intercept plus a dozen degrees
            of freedom.

    Returns:
        pd.Series: Cost of equity per period of `scale` months, indexed by
            `tickers` in the order given. NaN for a ticker with fewer than
            `min_obs` months of return history.

    Raises:
        KeyError: If a column of `FF5_COMPONENT` or a name in `tickers` is
            missing from `data`.
        ValueError: If `expectation` is neither 'mean' nor 'last'.

    Example:
        >>> df = generator()
        >>> re_cal(df.loc[:'2015-12-31'], ['AAPL', 'KO']).round(4)
        AAPL    0.0062
        KO      0.0185
        dtype: float64
    """
    if expectation not in ('mean','last'):
        raise ValueError(f"expectation must be 'mean' or 'last', got {expectation!r}")
    try:
        ff5_df: pd.DataFrame = data[FF5_COMPONENT].dropna()
    except KeyError:
        raise KeyError('There is at least 1 component of FF5 that is not in Data')

    try:
        return_df: pd.DataFrame = data[tickers]
    except KeyError:
        raise KeyError('There is at least 1 ticker that is not in Data ')

    return_df = return_df.reindex(ff5_df.index)
    ff5_arr: np.ndarray = ff5_df[FF5_FACTOR].to_numpy()
    rf_arr: np.ndarray = ff5_df[RF].to_numpy()
    return_arr: np.ndarray = return_df.to_numpy()

    # The premium the betas are paid, and the rate observed at the formation date.
    if expectation=='last':
        f_exp: np.ndarray = ff5_arr[-1]
    else:
        f_exp = ff5_arr.mean(axis=0) if factor_window is None \
            else ff5_arr[-factor_window::].mean(axis=0)
    rf_last: float = rf_arr[-1]

    X: np.ndarray = np.c_[np.ones(ff5_arr.shape[0]),ff5_arr]
    excess_return: np.ndarray = return_arr-rf_arr[:,None]

    re: np.ndarray = np.full(len(tickers),np.nan)
    for i in range(len(tickers)):
        valid: np.ndarray = ~np.isnan(excess_return[:,i])
        if valid.sum() < min_obs:
            continue
        coef, *_ = np.linalg.lstsq(X[valid],excess_return[valid,i],rcond=None)
        beta: np.ndarray = coef[1::]
        re[i] = rf_last+beta@f_exp

    return pd.Series(re*scale,index=tickers)

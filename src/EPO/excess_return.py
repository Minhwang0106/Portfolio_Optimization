import pandas as pd
import numpy as np
from pathlib import Path

def cal_excess_return(price_path: Path, rf_path: Path, start_date: str = "2000-01-01") -> pd.DataFrame:
    """Compute daily excess returns (simple return minus risk-free rate) per ticker.

    price_path is expected to hold long-format columns ['ticker', 'date', 'adj_close'].
    rf_path is expected to hold Kenneth French daily factor data with a 'date' column
    in YYYYMMDD format and an 'RF' column given in percent (e.g. 0.01 == 0.01%).

    Args:
        price_path: CSV of daily adjusted close prices, long format.
        rf_path: CSV of Fama-French daily factors, must contain 'date' and 'RF'.
        start_date: Inclusive lower bound (YYYY-MM-DD) applied to the result's date index.

    Returns:
        DataFrame indexed by date, columns are tickers, values are excess returns
        (decimal, not percent).
    """
    df_p: pd.DataFrame = pd.read_csv(price_path)
    df_p['date'] = pd.to_datetime(df_p['date'])
    df_p.set_index(['ticker', 'date'], inplace=True)
    return_df: pd.DataFrame = df_p['adj_close'].unstack(level='ticker').pct_change()

    df_rf: pd.DataFrame = pd.read_csv(rf_path)
    if 'date' not in df_rf.columns:
        df_rf.rename(columns={df_rf.columns[0]: 'date'}, inplace=True)
    date_str = df_rf['date'].astype(str)
    if date_str.str.len().eq(6).all():
        # monthly FF5 file: YYYYMM, no day-of-month -> snap to month end to match price dates
        df_rf['date'] = pd.to_datetime(date_str, format='%Y%m') + pd.offsets.MonthEnd(0)
    else:
        df_rf['date'] = pd.to_datetime(date_str, format='%Y%m%d')
    df_rf.set_index('date', inplace=True)
    rf_rate: pd.Series = (df_rf['RF'] / 100).rename('RF_rate')  # RF is a stock ticker too, avoid name collision

    return_df = pd.concat([return_df, rf_rate], axis=1)
    return_df = return_df[return_df.index >= start_date]
    return_df = return_df[return_df.index <= rf_rate.index.max()]  # RF often lags price data; trim to avoid NaN rows

    tickers: list = [c for c in return_df.columns if c != 'RF_rate']
    return_arr: np.ndarray = return_df[tickers].to_numpy()
    rf_arr: np.ndarray = return_df[['RF_rate']].to_numpy()  # kept 2D (n,1) to broadcast against (n, n_tickers)

    excess_return: pd.DataFrame = pd.DataFrame(
        return_arr - rf_arr, columns=tickers, index=return_df.index
    )
    excess_return.index.name = 'date'
    excess_return.columns.name = 'ticker'
    return excess_return
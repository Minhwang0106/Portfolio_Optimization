import pandas as pd
import numpy as np
from constant import (
    accounting_path, monthly_price_path
)
from pathlib import Path
from ..panel import read_csv_file, book_equity

def generator (accout_path: Path = accounting_path,
               price_path: Path = monthly_price_path)->pd.DataFrame:
    """Join accounting and price data into the raw inputs the PPP model needs.

    Produces the three quantities `PPP.Config` turns into characteristics:

    * `book_value` -- common-equity book value, i.e. total equity net of
      minority interest and preferred stock (clean-surplus BE). Lagged one row
      per the `.shift(1)`, so a formation date never sees a figure that was
      still unfiled.
    * `market_cap` -- `close * shares`, in price units.
    * `adj_close` -- split/dividend-adjusted close, the return input.

    Accounting data is quarterly and price data monthly, so the concat unions
    the two calendars and the per-ticker forward fill carries each quarterly
    book value across the months that follow it. Months before a ticker's first
    filing stay NaN (ffill never back-fills) and are dropped later by the
    completeness screen in `PPP.find_theta`.

    Args:
        accout_path (Path): Accounting panel CSV, indexed (ticker, date), with
            columns including 'book_value', 'minority_interest',
            'preferred_stock'. Defaults to `constant.accounting_path`.
        price_path (Path): Monthly price panel CSV, indexed (ticker, date), with
            columns 'close', 'adj_close', 'shares'. Defaults to
            `constant.monthly_price_path`.

    Returns:
        pd.DataFrame: Indexed by (ticker, date) over the union of both
            calendars, with columns ['book_value', 'market_cap', 'adj_close'].

    Note:
        The `.shift(1)` runs over the whole stacked frame rather than
        per-ticker, so the first row of each ticker inherits the previous
        ticker's last book value instead of NaN.

    Example:
        Given `accounting.csv`::

            ticker,date,...,book_value,...,minority_interest,preferred_stock,shares
            A,2020-03-31,...,4617000000.0,...,0.0,0.0,306000000.0

        and `monthly_price.csv`::

            ticker,date,close,adj_close,shares
            A,2020-03-31,71.40,75.317406,306000000.0

        >>> generator().loc['A'].loc['2020-03-31':'2020-05-31']
                      book_value    market_cap  adj_close
        date
        2020-03-31  4.617000e+09  2.184840e+10  75.317406
        2020-04-30  4.617000e+09  2.407000e+10  82.964447
        2020-05-31  4.617000e+09  2.512000e+10  86.582024
    """
    account_df: pd.DataFrame = read_csv_file(accout_path)
    price_df: pd.DataFrame = read_csv_file(price_path)
    columns_value: list[pd.Series] = []
    col_keys: list[str] = ['book_value', 'market_cap' , 'adj_close']
    
    #book value
    book_value: pd.Series = book_equity(account_df)
    columns_value.append(book_value.shift(1))
    #market_cap
    columns_value.append(price_df['close']*price_df['shares'])
    #adj_close
    columns_value.append(price_df['adj_close'])
    
    df: pd.DataFrame = pd.concat(columns_value,axis=1,keys=col_keys)
    df = df.sort_index().groupby(level=0).ffill()
    return df
    
    
    


    
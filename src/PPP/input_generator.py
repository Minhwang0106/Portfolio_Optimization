import pandas as pd
import numpy as np
from constant import (
    accounting_path, monthly_price_path
)
from pathlib import Path

def read_csv_file (path: Path, date: str|None = 'date',
                   index: list[str]=['ticker','date'])-> pd.DataFrame:
    """Read a raw CSV into a sorted, MultiIndexed panel.

    Parses the date column to datetime, sets the index, and sorts it -- the
    sort matters because every downstream `.loc[idx[ticker, d0:d1], :]` slice
    needs a lexsorted index.

    Args:
        path (Path): CSV to read, e.g. `constant.monthly_price_path`.
        date (str | None): Name of the column to parse as datetime, or None to
            skip parsing (for files with no date column). Defaults to 'date'.
        index (list[str]): Columns to use as the index. Defaults to
            `['ticker', 'date']`.

    Returns:
        pd.DataFrame: The file's remaining columns, indexed by `index` and
            sorted.

    Raises:
        KeyError: If `date` or any name in `index` is not a column of the file.

    Example:
        Given `monthly_price.csv`::

            ticker,date,close,adj_close,shares
            A,1999-11-30,30.177038,25.110935,
            A,1999-12-31,55.302216,46.018112,

        >>> read_csv_file(monthly_price_path).head(2)
                               close  adj_close  shares
        ticker date
        A      1999-11-30  30.177038  25.110935     NaN
               1999-12-31  55.302216  46.018112     NaN
    """
    df: pd.DataFrame = pd.read_csv(path)
    if date is not None:
        try:
            df[date] = pd.to_datetime(df[date])
        except KeyError:
            raise KeyError(f"{date} is not in data columns")
    try:
        df.set_index(index, inplace=True)
        df.sort_index(inplace=True)
    except KeyError as e:
        raise e
    return df

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
    # NCI is already netted out of 'book_value' by the data layer (see
    # constant.ratio); only preferred stock is still inside the equity tags.
    book_value: pd.Series = account_df['book_value']-account_df['preferred_stock']
    columns_value.append(book_value.shift(1))
    #market_cap
    columns_value.append(price_df['close']*price_df['shares'])
    #adj_close
    columns_value.append(price_df['adj_close'])
    
    df: pd.DataFrame = pd.concat(columns_value,axis=1,keys=col_keys)
    df = df.sort_index().groupby(level=0).ffill()
    return df
    
    
    


    
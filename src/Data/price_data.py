import yfinance as yf
import pandas as pd
from .network import ensure_encodable_cacert

def take_splits (ticker: str)-> pd.Series:
    """Fetch the full split history of a ticker from Yahoo Finance.

    Needed because the prices Yahoo returns are already back-adjusted for
    splits -- `auto_adjust=False` only turns off the *dividend* adjustment,
    so `close` and `adj_close` are both stated on today's share basis while
    the share count from SEC filings is point-in-time. Market cap, price-to-
    book and every per-share quantity are wrong before a split until the two
    are put on the same basis, and that needs the split factors explicitly.

    Args:
        ticker (str): Ticker symbol, case-insensitive. Dot forms ('BRK.B')
            are converted to the hyphen form Yahoo expects.

    Returns:
        pd.Series: Split ratios indexed by tz-naive effective date, named
            'split_ratio' with the index named 'date'. Empty for a ticker
            that has never split.

    Example:
        >>> take_splits('AAPL').tail(2)
        date
        2014-06-09    7.0
        2020-08-31    4.0
        Name: split_ratio, dtype: float64
    """
    ensure_encodable_cacert()
    ticker = ticker.upper().replace('.','-')
    splits = yf.Ticker(ticker).splits

    # `.splits` is None on a fetch failure but an empty Series for a genuine
    # non-splitter; collapsing both to empty would turn an outage into a
    # silent "no splits ever" and reintroduce the bug it exists to fix.
    if splits is None:
        raise ValueError(f"{ticker}: split history unavailable")
    if not len(splits):
        return pd.Series(dtype=float, name='split_ratio',
                         index=pd.DatetimeIndex([], name='date'))

    splits.index = pd.to_datetime(splits.index).tz_localize(None).normalize()
    splits.index.name = 'date'
    return splits.rename('split_ratio').astype(float)

def take_price (ticker: str)-> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the full historical daily and monthly close/adjusted close price
    of a ticker from Yahoo Finance.

    Daily data is pulled via yfinance (period='max'). Monthly data is derived
    by resampling to month-end frequency ('ME'), taking the last available
    value of each month. Class-share tickers written with a dot (e.g.
    'BRK.B') are automatically converted to the hyphen form ('BRK-B')
    expected by Yahoo Finance / yfinance.

    Args:
        ticker (str): Ticker symbol of a public company, case-insensitive
            (e.g. 'aapl', 'BRK.B', 'brk-b').

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (daily_price, monthly_price).
            Each has a MultiIndex ('ticker', 'date') and 2 columns:
                - close (float): split-adjusted, dividend-unadjusted close.
                - adj_close (float): close adjusted for dividends and splits.

            Both are back-adjusted for splits and neither is point-in-time:
            Yahoo applies the split adjustment server-side, so AAPL's close
            for 2020-08-28 is 124.81 rather than the 499.23 that actually
            traded. `auto_adjust=False` only declines the *dividend*
            adjustment -- the two columns differ by dividends alone, which is
            why they are identical for AMZN and 32% apart for ED. Pair either
            with `take_splits` before combining with a point-in-time share
            count; see `src.Data.network` for why the fetch needs that
            workaround at all.
            'date' is every trading day for daily_price, and the last
            trading day of each month for monthly_price.

    Raises:
        ValueError: If the ticker doesn't exist or is delisted (no data
            returned from Yahoo Finance).

    Example:
        >>> daily, monthly = take_price('AAPL')
        >>> monthly.head(2)
                              close  adj_close
        ticker date
        AAPL   1980-12-31  0.152344   0.116568
               1981-01-31  0.126116   0.096499
    """
    ensure_encodable_cacert()
    ticker = ticker.upper().replace('.','-')
    tick: yf.ticker.Ticker = yf.Ticker(ticker)
    his_price: pd.DataFrame = tick.history(period='max',auto_adjust=False)
    if not len(his_price):
        raise ValueError(f"{ticker} possibly delisted; data not found")

    his_price.index = pd.to_datetime(his_price.index).tz_localize(None)
    his_price = his_price[['Close','Adj Close']]
    his_price.columns = ['close', 'adj_close']

    monthly_price: pd.DataFrame = his_price.resample('ME').last()
    daily_price: pd.DataFrame = his_price.copy()

    for df in (daily_price, monthly_price):
        df.index = pd.MultiIndex.from_tuples(
            [(ticker, i) for i in df.index], names=['ticker', 'date']
        )

    return daily_price, monthly_price
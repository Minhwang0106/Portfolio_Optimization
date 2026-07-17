import yfinance as yf
import pandas as pd

def take_price (ticker: str)-> pd.DataFrame:
    """Fetch the full historical monthly close and adjusted close price of a ticker
    from Yahoo Finance.

    Daily data is pulled via yfinance (period='max'), then resampled to
    month-end frequency ('ME') by taking the last available value of each
    month. Class-share tickers written with a dot (e.g. 'BRK.B') are
    automatically converted to the hyphen form ('BRK-B') expected by
    Yahoo Finance / yfinance.

    Args:
        ticker (str): Ticker symbol of a public company, case-insensitive
            (e.g. 'aapl', 'BRK.B', 'brk-b').

    Returns:
        pd.DataFrame: DataFrame with a MultiIndex ('ticker', 'date'), where
            'date' is the last trading day of each month, and 2 columns:
                - close (float): unadjusted close price.
                - adj_close (float): close price adjusted for dividends/splits.

    Raises:
        ValueError: If the ticker doesn't exist or is delisted (no data
            returned from Yahoo Finance).

    Example:
        >>> take_price('AAPL').head(2)
                              close  adj_close
        ticker date
        AAPL   1980-12-31  0.152344   0.116568
               1981-01-31  0.126116   0.096499
    """
    ticker = ticker.upper().replace('.','-')
    tick: yf.ticker.Ticker = yf.Ticker(ticker)
    his_price: pd.DataFrame = tick.history(period='max',auto_adjust=False)
    if not len(his_price):
        raise ValueError(f"{ticker} possibly delisted; data not found")
    
    his_price.index = pd.to_datetime(his_price.index).tz_localize(None)
    his_price = his_price.resample('ME').last()
    mul_index = [(ticker, i) for i in list(his_price.index)]
    mul_index = pd.MultiIndex.from_tuples(mul_index)
    his_price.index = mul_index
    his_price.index.names = ['ticker', 'date']
    his_price = his_price[['Close','Adj Close']]
    his_price.columns = ['close', 'adj_close']
    
    return his_price
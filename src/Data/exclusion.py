import pandas as pd
from pandas.tseries.offsets import QuarterEnd
from constant import (
    testing_period, n_quarter, n_month, sp_500_path
)

def applicable_ticker(
    account_data: pd.DataFrame,
    monthly_price: pd.DataFrame,
    test_per=testing_period,
    nq=n_quarter,
    nm=n_month,
    p_path=sp_500_path):

    ticker_overtime: dict[str, set[str]] = {}

    monthly_price = monthly_price['adj_close'].unstack(level=0)
    historical_ticker: pd.DataFrame = pd.read_csv(p_path)
    historical_ticker['date'] = pd.to_datetime(historical_ticker['date'])
    historical_ticker.set_index('date', inplace=True)
    historical_ticker = historical_ticker[~historical_ticker.index.duplicated(keep='last')]
    historical_ticker.sort_index(inplace=True)

    items: list[str] = [i for i in account_data.columns if i != 'shares']
    unstacked_items: dict[str, pd.DataFrame] = {
        item: account_data[item].unstack(level=0) for item in items
    }

    for date in test_per:
        account_begin: pd.Timestamp = date - QuarterEnd(nq)
        price_begin: pd.Timestamp = date - pd.DateOffset(months=nm)
        acc_ticker: set[str] | None = None

        raw_tickers = historical_ticker['tickers'].asof(date)
        if not isinstance(raw_tickers, str):
            continue
        ticker = raw_tickers.split(',')
        tickers: list[str] = [i.upper().replace('.', '-') for i in ticker]
        ticker_set: set[str] = set(tickers)

        temp_price: pd.DataFrame = monthly_price.loc[price_begin:date
                                                     ].dropna(axis=1)
        price_ticker: set[str] = set(temp_price.columns)

        for item in items:
            unstack_acc: pd.DataFrame = unstacked_items[item]
            unstack_acc = unstack_acc.loc[account_begin:date].dropna(axis=1)
            temp_ticker: set[str] = set(unstack_acc.columns)
            if acc_ticker is None:
                acc_ticker = temp_ticker
            else:
                acc_ticker &= temp_ticker

        if acc_ticker is None:
            acc_ticker = set()
        ticker_overtime[str(date)] = acc_ticker & price_ticker & ticker_set
    return ticker_overtime

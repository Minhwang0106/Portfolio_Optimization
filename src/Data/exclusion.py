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
    """Determine, for each test date, which tickers pass data-availability filters.

    A ticker qualifies at a given date if it: (1) was a historical S&P 500
    constituent as of that date (per `p_path`), (2) has complete monthly
    price history over the trailing `nm` months, (3) has complete
    accounting data (every column in `account_data`) over the trailing `nq`
    quarters, and (4) never has negative book equity over those same `nq`
    quarters (PPP_2009.md sec.2 — `btm = log(1 + BE/ME)` is undefined once
    BE/ME <= -1, so the paper drops these firms).

    Args:
        account_data (pd.DataFrame): MultiIndex ('ticker', 'date') accounting
            data; may include a 'shares' column which is excluded from the
            completeness check.
        monthly_price (pd.DataFrame): MultiIndex ('ticker', 'date') price
            data with an 'adj_close' column.
        test_per: Iterable of dates to evaluate.
        nq (int): Number of trailing quarters of accounting data required.
        nm (int): Number of trailing months of price data required.
        p_path: CSV path of historical S&P 500 constituents, with 'date' and
            'tickers' (comma-separated) columns.

    Returns:
        dict[str, set[str]]: Mapping of stringified date to the set of
            qualifying tickers at that date.

    Example:
        >>> result = applicable_ticker(account_data, monthly_price)
        >>> result['2020-03-31']
        {'AAPL', 'MSFT', ...}
    """
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

    # Book equity, built the same way as src/PPP/input_generator.generator:
    # BE = book_value - minority_interest - preferred_stock.
    book_equity: pd.Series = account_data['book_value']
    for deduction in ('minority_interest', 'preferred_stock'):
        if deduction in account_data.columns:
            book_equity = book_equity - account_data[deduction]
    unstacked_be: pd.DataFrame = book_equity.unstack(level=0)

    for date in test_per:
        account_asof: pd.Timestamp = date - QuarterEnd(1)
        account_begin: pd.Timestamp = account_asof - QuarterEnd(nq)
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

        # Keep only tickers with no negative book equity in the window. Tested
        # as `~(... < 0)` rather than `> 0` so that NaN (missing) stays the
        # completeness check's job below instead of being dropped twice.
        be_window: pd.DataFrame = unstacked_be.loc[account_begin:account_asof]
        positive_be_ticker: set[str] = set(
            be_window.columns[~(be_window < 0).any(axis=0)])

        for item in items:
            unstack_acc: pd.DataFrame = unstacked_items[item]
            unstack_acc = unstack_acc.loc[account_begin:account_asof].dropna(axis=1)
            temp_ticker: set[str] = set(unstack_acc.columns)
            if acc_ticker is None:
                acc_ticker = temp_ticker
            else:
                acc_ticker &= temp_ticker

        if acc_ticker is None:
            acc_ticker = set()
        ticker_overtime[str(date)] = (acc_ticker & price_ticker
                                      & ticker_set & positive_be_ticker)
    return ticker_overtime

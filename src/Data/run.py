import pandas as pd
import requests
from .accounting_info import Sec_Data_Restructure
from .price_data import take_price
from .exclusion import applicable_ticker as compute_applicable_ticker
from constant import (
    ticker_data, ratio, optional_ratio_fields,
    monthly_price_path, daily_price_path,
    accounting_path, applicable_ticker_path,
)

def run (ticker=ticker_data, ratio=ratio,
         optional_ratio_fields=optional_ratio_fields,
         monthly_price_path=monthly_price_path,
         daily_price_path=daily_price_path,
         accounting_path=accounting_path,
         applicable_ticker_path=applicable_ticker_path,
         )->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    """Build and save the full dataset: prices, accounting facts, and applicable tickers.

    For each ticker, fetches SEC accounting facts and Yahoo Finance prices,
    merges in shares-outstanding, then concatenates across all tickers and
    computes the set of data-qualifying tickers per date via
    `applicable_ticker`. Results are written to CSV and also returned.
    Tickers that error out (bad data, HTTP failures) are skipped and logged.

    Args:
        ticker: Iterable of ticker symbols to process.
        ratio: Mapping of accounting field name to candidate SEC tags, passed
            to `Sec_Data_Restructure.line_item_restructure`.
        optional_ratio_fields: Field names allowed to be missing/zero-filled.
        monthly_price_path (Path): Output CSV path for monthly prices.
        daily_price_path (Path): Output CSV path for daily prices.
        accounting_path (Path): Output CSV path for accounting data.
        applicable_ticker_path (Path): Output CSV path for the per-date
            applicable-ticker sets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
            (monthly_price, daily_price, accounting, ticker_overtime) for
            all successfully processed tickers combined.

    Raises:
        ValueError: If every ticker fails, so there is nothing to save.

    Example:
        >>> price, daily_price, accounting, applicable = run(ticker=['AAPL'])
    """
    price_listframe: list[pd.DataFrame] = []
    daily_price_listframe: list[pd.DataFrame] = []
    accounting_listframe: list[pd.DataFrame] = []
    errors: list[str] = []
    for tick in ticker:
        try:
           tick_object: Sec_Data_Restructure = Sec_Data_Restructure(tick)
           daily_price, price = take_price(tick)
           shares: pd.DataFrame = tick_object.share_compose()
           accounting: pd.DataFrame = tick_object.line_item_restructure(
               ratio, optional_ratio_fields)

           price = pd.concat([price,shares],axis=1).sort_index(level='date')
           price['shares'] = price['shares'].ffill().bfill()
           price.dropna(inplace=True)

           accounting = pd.concat([accounting,shares],axis=1).sort_index(level='date')
           accounting['shares'] = accounting['shares'].ffill().bfill()
           accounting.dropna(inplace=True)

           price_listframe.append(price)
           daily_price_listframe.append(daily_price)
           accounting_listframe.append(accounting)
        except (ValueError, requests.exceptions.RequestException) as e:
            errors.append(f'{tick}: {e}')

    if errors:
        print(f'{len(errors)} ticker(s) skipped due to errors:')
        for err in errors:
            print(f'  - {err}')

    if not price_listframe or not accounting_listframe:
        raise ValueError('No ticker succeeded; nothing to save.')

    multi_tick_price: pd.DataFrame = pd.concat(price_listframe,axis=0)
    multi_tick_daily_price: pd.DataFrame = pd.concat(daily_price_listframe,axis=0)
    multi_tick_accounting: pd.DataFrame = pd.concat(accounting_listframe,axis=0)

    ticker_overtime: dict[str, set[str]] = compute_applicable_ticker(
        multi_tick_accounting, multi_tick_price
    )

    monthly_price_path.parent.mkdir(parents=True, exist_ok=True)
    daily_price_path.parent.mkdir(parents=True, exist_ok=True)
    accounting_path.parent.mkdir(parents=True, exist_ok=True)
    applicable_ticker_path.parent.mkdir(parents=True, exist_ok=True)
    multi_tick_price.to_csv(monthly_price_path)
    multi_tick_daily_price.to_csv(daily_price_path)
    multi_tick_accounting.to_csv(accounting_path)
    pd.Series(
        {date: ','.join(sorted(tickers)) for date, tickers in ticker_overtime.items()},
        name='tickers',
    ).rename_axis('date').to_csv(applicable_ticker_path)

    return multi_tick_price, multi_tick_daily_price, multi_tick_accounting, ticker_overtime

if __name__ == "__main__":
    run()

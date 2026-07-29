import pandas as pd
import requests
from .accounting_info import Sec_Data_Restructure
from .price_data import take_price, take_splits
from .split_adjust import adjust_shares
from .exclusion import applicable_ticker as compute_applicable_ticker
from ..panel import clear_panel_cache
from constant import (
    ticker_data, ratio, optional_ratio_fields,
    monthly_price_path, daily_price_path,
    accounting_path, applicable_ticker_path, splits_path,
    share_ffill_month, share_ffill_quarter,
)

def run_splits (ticker=ticker_data, splits_path=splits_path)-> pd.DataFrame:
    """Fetch and save the split history of every ticker.

    Split factors are what put the back-adjusted price and the point-in-time
    share count on a common basis; without them market cap is wrong for every
    date preceding a split. Kept separate from `run` so the table can be
    refreshed on its own -- it needs only Yahoo, not SEC, and splits are
    revised far less often than accounting facts.

    Args:
        ticker: Iterable of ticker symbols to process.
        splits_path (Path): Output CSV path, columns ['ticker','date',
            'split_ratio'].

    Returns:
        pd.DataFrame: Long-format split events for all successful tickers.
            Tickers that have never split contribute no rows, which is not
            distinguishable in the output from a ticker that failed -- hence
            the error log.

    Raises:
        ValueError: If every ticker failed, so there is nothing to save.

    Example:
        >>> splits = run_splits(ticker=['AAPL'])
    """
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    succeeded: int = 0
    for tick in ticker:
        try:
            series: pd.Series = take_splits(tick)
            succeeded += 1
            if len(series):
                frames.append(series.reset_index().assign(ticker=tick))
        except (ValueError, requests.exceptions.RequestException) as e:
            errors.append(f'{tick}: {e}')

    if errors:
        print(f'{len(errors)} ticker(s) skipped due to errors:')
        for err in errors:
            print(f'  - {err}')
    if not succeeded:
        raise ValueError('No ticker succeeded; nothing to save.')

    all_splits: pd.DataFrame = (
        pd.concat(frames, axis=0)[['ticker', 'date', 'split_ratio']]
        if frames else
        pd.DataFrame(columns=['ticker', 'date', 'split_ratio'])
    ).sort_values(['ticker', 'date'])

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    all_splits.to_csv(splits_path, index=False)
    return all_splits


def run_share_adjustment (splits_path=splits_path,
                          monthly_price_path=monthly_price_path,
                          accounting_path=accounting_path,
                          ) -> dict[str, int]:
    """Add an `adj_shares` column to every panel carrying a share count.

    `shares` is left exactly as filed. The adjusted count goes in a new column
    because the adjustment is relative to *today's* split basis: the next time
    any ticker splits, every stored `adj_shares` for that ticker becomes wrong
    by the new factor, with nothing in the file to show it. Keeping the raw
    count means this is re-runnable from `splits.csv` rather than needing the
    SEC and Yahoo pulls again -- so re-run it after `run_splits`, always.

    Overwriting `shares` in place would lose that, and would also destroy the
    point-in-time count, which is the honest record of what the filing said.

    Args:
        splits_path (Path): Split events CSV, as written by `run_splits`.
        monthly_price_path (Path): Monthly price CSV to rewrite.
        accounting_path (Path): Accounting CSV to rewrite.

    Returns:
        dict[str, int]: Number of rows whose adjusted count differs from the
            filed one, per file.

    Raises:
        FileNotFoundError: If `splits_path` does not exist yet.

    Example:
        >>> run_share_adjustment()
        {'monthly_price.csv': 21451, 'accounting.csv': 5461}
    """
    if not splits_path.is_file():
        raise FileNotFoundError(
            f'{splits_path} not found; run `run_splits` first.')
    splits: pd.DataFrame = pd.read_csv(splits_path)

    touched: dict[str, int] = {}
    for path in (monthly_price_path, accounting_path):
        frame: pd.DataFrame = pd.read_csv(path)
        frame['date'] = pd.to_datetime(frame['date'])
        indexed: pd.DataFrame = frame.set_index(['ticker', 'date']).sort_index()
        adjusted: pd.Series = adjust_shares(indexed['shares'], splits)
        indexed['adj_shares'] = adjusted
        # Both null counts as unchanged; `!=` would call NaN != NaN a change
        # and report every pre-2009 price row, where no share count exists.
        both: pd.DataFrame = indexed[['shares', 'adj_shares']].dropna()
        touched[path.name] = int(
            (both['adj_shares'] != both['shares']).sum())
        indexed.reset_index().to_csv(path, index=False)

    # The panels are memoised on path, so a live process would otherwise keep
    # serving the frames read before this rewrite.
    clear_panel_cache()
    return touched

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

           # Forward-fill only, and only for a bounded number of periods. An
           # unbounded bfill would invent a share count for every date before
           # the filer's first XBRL filing (~2009), which is most of the price
           # history -- and for re-incorporated filers such as DIS and XRX,
           # whose CIK only carries facts from the reorganisation onward, that
           # fabrication reaches into the sample period. Leaving those dates
           # NaN lets the market-cap consumers drop them instead.
           price = pd.concat([price,shares],axis=1).sort_index(level='date')
           price['shares'] = price['shares'].ffill(limit=share_ffill_month)
           price.dropna(subset=['close','adj_close'],inplace=True)

           accounting = pd.concat([accounting,shares],axis=1).sort_index(level='date')
           accounting['shares'] = accounting['shares'].ffill(
               limit=share_ffill_quarter)
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

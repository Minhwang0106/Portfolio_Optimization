import pandas as pd
import requests
from pathlib import Path
from .accounting_info import Sec_Data_Restructure, sec_header
from .price_data import take_price, take_splits
from .split_adjust import adjust_shares
from .exclusion import applicable_ticker as compute_applicable_ticker
from ..panel import clear_panel_cache, read_csv_file
from constant import (
    ticker_data, ratio, optional_ratio_fields,
    monthly_price_path, daily_price_path,
    accounting_path, applicable_ticker_path, splits_path,
    share_ffill_month, share_ffill_quarter, universe_asof,
)


def _write_applicable_ticker (ticker_overtime: dict[str, set[str]],
                              path=applicable_ticker_path)-> None:
    """Write the per-date applicable-ticker sets out, one comma-joined row each.

    Shared by `run` and `run_applicable_ticker` so the two cannot write the file
    in two different shapes; `Empirical_Analysis.data.universe` parses exactly
    this one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(
        {date: ','.join(sorted(tickers)) for date, tickers in ticker_overtime.items()},
        name='tickers',
    ).rename_axis('date').to_csv(path)

def _panel_tickers (path: Path)-> set[str]:
    """Tickers already written to a panel CSV; empty set if it does not exist.

    Reads the one column, not the file -- `daily_price.csv` is ~200MB and the
    only question being asked of it is which symbols it already covers.
    """
    if not Path(path).is_file():
        return set()
    return set(pd.read_csv(path, usecols=['ticker'])['ticker'].astype(str))


def _flush_panels (listframes: list[list[pd.DataFrame]], paths: list[Path],
                   truncate: bool)-> None:
    """Write the accumulated per-ticker frames out and empty the lists.

    Appends rather than rewrites, which is what keeps a resumed collection from
    having to hold the panels already on disk in memory. The lists are cleared
    in place so the caller's accumulators keep only what has not been written
    yet, bounding peak memory by the checkpoint interval instead of by the
    length of the ticker list.

    Args:
        listframes: Accumulators, parallel to `paths`. Cleared on write.
        paths (list[Path]): Destination CSVs.
        truncate (bool): Start the file over rather than appending. Set on the
            first flush of a fresh (non-resumed) collection, so that the
            previous contents go away only once something has actually been
            fetched to replace them.
    """
    for listframe, path in zip(listframes, paths):
        if not listframe:
            continue
        chunk: pd.DataFrame = pd.concat(listframe, axis=0).sort_index()
        path.parent.mkdir(parents=True, exist_ok=True)
        # A file that does not exist yet needs its header even mid-resume.
        fresh: bool = truncate or not path.is_file()
        chunk.to_csv(path, mode='w' if fresh else 'a', header=fresh)
        listframe.clear()


def run_splits (ticker=ticker_data, splits_path=splits_path,
                resume: bool = False)-> pd.DataFrame:
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
        resume (bool): Fetch only tickers with no rows in `splits_path` already
            and append to it, instead of refetching every symbol and rewriting.
            For widening the universe, where most of the list is already there.
            Note that a ticker which has never split leaves no rows, so it is
            indistinguishable from one never fetched and gets fetched again --
            one Yahoo call each, and the alternative is a second file recording
            what was attempted.

    Returns:
        pd.DataFrame: Long-format split events for all successful tickers.
            Tickers that have never split contribute no rows, which is not
            distinguishable in the output from a ticker that failed -- hence
            the error log. Under `resume` this is the whole file re-read, not
            just the new rows, so it stays the same table either way.

    Raises:
        ValueError: If every ticker failed, so there is nothing to save.

    Example:
        >>> splits = run_splits(ticker=['AAPL'])
    """
    if resume:
        done: set[str] = _panel_tickers(splits_path)
        ticker = [t for t in ticker if t not in done]
        if not ticker:
            return pd.read_csv(splits_path)

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
    if resume and splits_path.is_file():
        # Sorted on the combined table, not the appended block: `adjust_shares`
        # walks a ticker's events in order, and appending B's splits after Z's
        # would leave the file grouped by fetch order instead.
        # `parse_dates`, because the fetched block's 'date' is datetime64 and
        # the file's is str; concatenating the two gives an object column that
        # `sort_values` refuses to order.
        existing: pd.DataFrame = pd.read_csv(splits_path, parse_dates=['date'])
        all_splits = pd.concat(
            [existing, all_splits], axis=0
        ).sort_values(['ticker', 'date'])
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

def run_applicable_ticker (accounting_path=accounting_path,
                           monthly_price_path=monthly_price_path,
                           applicable_ticker_path=applicable_ticker_path,
                           asof=universe_asof)-> dict[str, set[str]]:
    """Rebuild the applicable-ticker table from the panels already on disk.

    The universe screen is a pure function of `accounting.csv` and
    `monthly_price.csv`, so changing `constant.universe_asof` -- or `n_quarter`,
    or `n_month` -- needs no SEC or Yahoo traffic at all. `run` recomputes this
    table as its last step, but re-running `run` to get it would re-fetch every
    ticker for a screen the existing files can already answer.

    Args:
        accounting_path (Path): Accounting panel CSV, as written by `run`.
        monthly_price_path (Path): Monthly price panel CSV, as written by `run`.
        applicable_ticker_path (Path): Output CSV path, overwritten.
        asof: Freeze date for the candidate pool, or None to reconstitute it per
            date. Passed to `exclusion.applicable_ticker`; defaults to
            `constant.universe_asof`.

    Returns:
        dict[str, set[str]]: Qualifying tickers per stringified date, the same
            mapping that was written.

    Note:
        Rewriting the file underneath a live process leaves two caches stale.
        This clears `panel`'s; a process that has already called
        `Empirical_Analysis.data.universe` must also call
        `Empirical_Analysis.data.clear_backtest_cache`, which is not imported
        here because `Data` does not depend on `Empirical_Analysis`.

    Example:
        >>> overtime = run_applicable_ticker(asof=None)
        >>> len(overtime['2025-12-31 00:00:00'])
        373
    """
    accounting: pd.DataFrame = read_csv_file(accounting_path)
    monthly_price: pd.DataFrame = read_csv_file(monthly_price_path)
    ticker_overtime: dict[str, set[str]] = compute_applicable_ticker(
        accounting, monthly_price, asof=asof)
    _write_applicable_ticker(ticker_overtime, applicable_ticker_path)
    clear_panel_cache()
    return ticker_overtime


def run (ticker=ticker_data, ratio=ratio,
         optional_ratio_fields=optional_ratio_fields,
         monthly_price_path=monthly_price_path,
         daily_price_path=daily_price_path,
         accounting_path=accounting_path,
         applicable_ticker_path=applicable_ticker_path,
         universe_asof=universe_asof,
         resume: bool = False,
         checkpoint: int = 25,
         user_agent: str|None = None,
         )->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    """Build and save the full dataset: prices, accounting facts, and applicable tickers.

    For each ticker, fetches SEC accounting facts and Yahoo Finance prices,
    merges in shares-outstanding, then concatenates across all tickers and
    computes the set of data-qualifying tickers per date via
    `applicable_ticker`. Results are written to CSV and also returned.
    Tickers that error out (bad data, HTTP failures) are skipped and logged.

    Rows reach disk every `checkpoint` tickers rather than once at the end, so
    an interrupted collection keeps what it had fetched and `resume` picks the
    rest up. This is a network-bound job measured in hours; losing it to a
    dropped connection at ticker 600 is the failure mode worth designing out.

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
        universe_asof: Freeze date for the candidate pool, or None to
            reconstitute it from S&P 500 membership at every date. Passed to
            `applicable_ticker`; defaults to `constant.universe_asof`. Changing
            only this does not need a re-fetch -- use `run_applicable_ticker`.
        resume (bool): Fetch only tickers absent from the panels on disk and
            append them, leaving the rows already there untouched. This is the
            mode for widening the universe -- moving `constant.universe_asof`
            from a freeze date to None takes the fetch list from 464 names to
            707, and refetching the ones that already succeeded would be most
            of the runtime for none of the data. A ticker counts as done only if
            all three panels carry it, so a symbol whose accounting failed
            after its prices were written is retried rather than left half
            collected.
        checkpoint (int): Flush to disk every this many *fetched* tickers.
            Lower survives interruption with less lost; higher writes fewer,
            larger blocks.
        user_agent (str|None): Contact string sent to SEC EDGAR as the
            User-Agent. Falls back to `$SEC_USER_AGENT`; see
            `accounting_info.sec_header` for why there is no default.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
            (monthly_price, daily_price, accounting, ticker_overtime). The
            panels are read back from disk, so under `resume` they are the full
            files -- old rows and new -- not just what this call fetched.

    Raises:
        ValueError: If every ticker fails, so there is nothing to save. Under
            `resume` an empty fetch list is not a failure: there is nothing
            left to collect, and the universe table is rebuilt from what is
            already on disk.
        RuntimeError: If no SEC contact string is available from either
            `user_agent` or `$SEC_USER_AGENT`.

    Example:
        >>> price, daily_price, accounting, applicable = run(ticker=['AAPL'])
    """
    # Resolved once, before any network work: a missing contact string is a
    # misconfiguration, and it should stop the run at second zero rather than
    # raise inside the fetch loop after an hour of successful requests.
    header: dict[str, str] = sec_header(user_agent)
    if resume:
        already: set[str] = (_panel_tickers(monthly_price_path)
                             & _panel_tickers(daily_price_path)
                             & _panel_tickers(accounting_path))
        remaining: list[str] = [t for t in ticker if t not in already]
        print(f'resume: {len(already)} ticker(s) on disk, '
              f'{len(remaining)} to fetch')
        ticker = remaining
    price_listframe: list[pd.DataFrame] = []
    daily_price_listframe: list[pd.DataFrame] = []
    accounting_listframe: list[pd.DataFrame] = []
    errors: list[str] = []
    paths: list[Path] = [monthly_price_path, daily_price_path, accounting_path]
    listframes: list[list[pd.DataFrame]] = [
        price_listframe, daily_price_listframe, accounting_listframe]
    # A fresh collection replaces the panels; a resumed one adds to them. Only
    # the first flush truncates, and only once something has been fetched, so a
    # run in which every ticker fails leaves the previous files intact.
    truncate: bool = not resume
    fetched: int = 0
    for tick in ticker:
        try:
           tick_object: Sec_Data_Restructure = Sec_Data_Restructure(
               tick, header=header)
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
           fetched += 1
        except (ValueError, requests.exceptions.RequestException) as e:
            errors.append(f'{tick}: {e}')
            # `continue`, so the checkpoint below counts fetches and not loop
            # iterations: a run of failures must not flush three empty lists
            # and, on the first of them, truncate the panels to nothing.
            continue

        # The panels are written *before* the universe table is derived from
        # them. They cost an hour of SEC and Yahoo requests; the table costs
        # seconds and is a pure function of the two files. Deriving first meant
        # that any error in the screen -- and the sort bug noted below was one
        # -- discarded the entire fetch, with nothing on disk to resume from.
        if fetched % checkpoint == 0:
            _flush_panels(listframes, paths, truncate)
            truncate = False
            print(f'  ... {fetched} fetched, written through {tick}')

    if errors:
        print(f'{len(errors)} ticker(s) skipped due to errors:')
        for err in errors:
            print(f'  - {err}')

    if not fetched:
        # Under `resume` there is legitimately nothing left to fetch; the
        # panels on disk are the answer and the universe table still needs
        # rebuilding off them. Only a fresh run that collected nothing is an
        # error, since it would have nothing to write.
        if not resume:
            raise ValueError('No ticker succeeded; nothing to save.')
    else:
        _flush_panels(listframes, paths, truncate)

    # Read back rather than returned from memory. The blocks were flushed as
    # they arrived, so no single frame here ever held the whole panel -- which
    # is the point when `daily_price.csv` runs to hundreds of megabytes and a
    # resumed run would otherwise have to hold the old rows as well as the new.
    #
    # `read_csv_file` also sorts, which the files themselves are not: each
    # flushed block is sorted within itself, so a file built over several
    # checkpoints is ordered by fetch order at the block level. `exclusion`
    # unstacks these frames and takes a *date slice* of the result, which pandas
    # refuses on a non-monotonic index, so the sort is required and not a
    # nicety. Every other consumer reads through `read_csv_file` too and so
    # gets it for free.
    clear_panel_cache()
    multi_tick_price: pd.DataFrame = read_csv_file(monthly_price_path)
    multi_tick_daily_price: pd.DataFrame = read_csv_file(daily_price_path)
    multi_tick_accounting: pd.DataFrame = read_csv_file(accounting_path)

    ticker_overtime: dict[str, set[str]] = compute_applicable_ticker(
        multi_tick_accounting, multi_tick_price, asof=universe_asof
    )
    _write_applicable_ticker(ticker_overtime, applicable_ticker_path)

    return multi_tick_price, multi_tick_daily_price, multi_tick_accounting, ticker_overtime

def main (argv: list[str]|None = None)-> None:
    """Drive the collection from the command line, in dependency order.

    The order is not arbitrary and the steps are not independent:

    1. `ticker_list.build_ticker_list` writes `ticker.csv`, and what it writes
       depends on `constant.universe_asof` -- the members at the freeze date,
       or the union of members across `constant.testing_period` if it is None.
       Its return value is used rather than `constant.ticker_data`, which was
       read at import and is stale the moment step 1 writes.
    2. `run` fetches prices and accounting facts for that list.
    3. `run_splits` fetches split factors, which
    4. `run_share_adjustment` needs to put the filed share count on the same
       basis as the back-adjusted price. It must follow both fetches, and must
       be re-run after any later `run_splits`.
    5. `run_applicable_ticker` derives the universe table from the panels.
       `run` already wrote one in step 2, but that was before the share
       adjustment, so this is the one that matches the files as they now stand.

    Steps 2 and 3 are the network-bound hours; 4 and 5 are seconds and read
    only what is already on disk.

    Alongside step 4, `industry.run_industry` maps every ticker to its
    Fama-French 48 industry, which `ICC_MVO` needs for its industry ROE. One
    SEC request per ticker, about a minute; it depends on nothing the other
    steps write, only on the ticker list and the User-Agent. Then
    `industry_pool.run_industry_pool` builds the pool that median is taken
    over: every SEC filer's calendar-year ROE and industry. A minute of XBRL
    frames, then one request per filer not yet in the SIC cache -- most of an
    hour the first time, seconds after that.
    """
    import argparse
    from .ticker_list import build_ticker_list

    parser = argparse.ArgumentParser(
        description='Collect the price, accounting and universe panels.')
    parser.add_argument(
        '--resume', action='store_true',
        help='fetch only tickers missing from the panels and append them, '
             'instead of refetching everything. Use after widening the '
             'universe or to pick up an interrupted collection.')
    parser.add_argument(
        '--checkpoint', type=int, default=25,
        help='flush to disk every N fetched tickers (default: 25)')
    parser.add_argument(
        '--user-agent', default=None, metavar='STRING',
        help='contact string sent to SEC EDGAR as the User-Agent, e.g. '
             '"Your Name you@example.com". Overrides $SEC_USER_AGENT. SEC '
             'identifies and throttles callers by this header, so one is '
             'required: at a terminal you are prompted for it, elsewhere the '
             'run fails without it. Not needed under --skip-fetch, which '
             'touches no network.')
    parser.add_argument(
        '--skip-fetch', action='store_true',
        help='skip steps 1-4 and only rebuild the universe table, which is a '
             'pure function of the panels already collected. This is all that '
             'changing `constant.universe_asof` needs when the panels already '
             'cover the wider list.')
    args = parser.parse_args(argv)

    if not args.skip_fetch:
        # Resolved here, before `build_ticker_list` rewrites ticker.csv: a run
        # that cannot identify itself to SEC should cost nothing at all, not a
        # regenerated file and a fetch list printed to a terminal that is about
        # to show a traceback instead of a collection. Resolved to a *string*,
        # not just checked, so that an interactive prompt is answered once here
        # rather than again inside `run`.
        user_agent: str = sec_header(args.user_agent)['User-Agent']
        tickers: list[str] = build_ticker_list()
        print(f'{len(tickers)} ticker(s) in the fetch list '
              f'(universe_asof={universe_asof})')

        run(ticker=tickers, resume=args.resume, checkpoint=args.checkpoint,
            user_agent=user_agent)
        run_splits(ticker=tickers, resume=args.resume)
        print(f'share adjustment: {run_share_adjustment()}')
        from .industry import run_industry
        from .industry_pool import run_industry_pool
        run_industry(user_agent=user_agent, tickers=tickers)
        run_industry_pool(user_agent=user_agent)

    overtime: dict[str, set[str]] = run_applicable_ticker()
    sizes: list[int] = [len(v) for v in overtime.values()]
    print(f'universe: {len(overtime)} dates, '
          f'{min(sizes)}-{max(sizes)} tickers per date')


if __name__ == "__main__":
    main()

"""Build `ticker.csv`, the list of symbols the collection actually fetches.

Derived from `sp_500_historical_components.csv` and nothing else, so the fetch
list cannot drift from the index history it is supposed to represent. This
replaces a hand-maintained file that carried its own exclusions: names whose
ticker no longer resolves to a CIK, multi-class filers with no
shares-outstanding tag, and a few reused symbols. Those exclusions are not
reproduced here -- a ticker that cannot be fetched now fails in `Data.run`,
which skips and logs it, and the skipped list is the record.

Runs before everything else in the collection, and is the one step that must
not read `constant.ticker_data`: that is the file this writes.
"""
import pandas as pd
from pathlib import Path
from constant import sp_500_path, ticker_path, testing_period, universe_asof


def constituents (date, path: Path|str = sp_500_path)-> set[str]:
    """S&P 500 members in force at `date`.

    Args:
        date: Date to read membership at.
        path (Path | str): Constituent history CSV, columns ['date', 'tickers']
            with the second comma-separated. Defaults to `constant.sp_500_path`.

    Returns:
        set[str]: Tickers, uppercased with '.' mapped to '-' -- the form Yahoo
            and `Data.exclusion` both use. Empty if the file begins after
            `date`.

    Example:
        >>> len(constituents('2015-01-31'))
        464
    """
    sp: pd.DataFrame = pd.read_csv(path)
    sp['date'] = pd.to_datetime(sp['date'])
    sp = sp.set_index('date')
    # `asof` reads the snapshot in force -- the most recent one dated at or
    # before `date` -- not the one nearest it, which for a date between two
    # snapshots would be the later, unknowable one.
    sp = sp[~sp.index.duplicated(keep='last')].sort_index()
    raw = sp['tickers'].asof(pd.Timestamp(date))
    if not isinstance(raw, str):
        return set()
    return {t.upper().replace('.', '-') for t in raw.split(',') if t}


def build_ticker_list (asof=universe_asof, dates=testing_period,
                       path: Path|str = sp_500_path,
                       out_path: Path = ticker_path)-> list[str]:
    """Write `ticker.csv`: every symbol the universe could ever admit.

    Under a frozen universe only the members at the freeze date can ever be
    held, however good a later entrant's data turns out to be, so collecting
    anything else is wasted requests. Under `asof=None` the pool reconstitutes
    at every formation date, so the union across `dates` is what is needed.
    The two differ by a lot -- 464 against 708 -- which is most of the runtime
    of a collection.

    Args:
        asof: Freeze date, or None for the reconstituting universe. Defaults to
            `constant.universe_asof`.
        dates: Formation dates, used only when `asof` is None. Defaults to
            `constant.testing_period`.
        path (Path | str): Constituent history CSV. Defaults to
            `constant.sp_500_path`.
        out_path (Path): Where to write. Defaults to `constant.ticker_path`.

    Returns:
        list[str]: The sorted tickers written. Return them rather than making
            the caller re-import `constant`, whose `ticker_data` was read at
            import time and is stale the moment this writes.

    Example:
        >>> len(build_ticker_list(asof=pd.Timestamp('2015-01-31')))
        464
    """
    if asof is not None:
        tickers: set[str] = constituents(asof, path)
    else:
        tickers = set().union(*(constituents(d, path) for d in dates))

    ordered: list[str] = sorted(tickers)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'ticker': ordered}).to_csv(out_path)
    return ordered


if __name__ == '__main__':
    built: list[str] = build_ticker_list()
    print(f'{len(built)} tickers written to {ticker_path}')
"""The raw panels, read once per process, and the definitions built on them.

Every model package reads the same two CSVs. Parsing them is the single most
expensive thing any of them does at startup, and each package used to pay it
separately -- `accounting.csv` was parsed once by `PPP.Config` and again by
`Proposed_Model.data_generator.generator`. This module holds one cache for all
of them, so the parse happens once no matter how many packages are in play.

The cache is on the *parse*, not on the frame handed back: callers get a copy.
Sharing one mutable DataFrame between packages would make an in-place edit in
one model silently rewrite another's inputs, and that class of bug is invisible
at the call site. A copy costs ~2-7ms against a ~30-145ms parse, so the saving
survives the defensive copy nearly intact.
"""
import pandas as pd
from functools import lru_cache
from pathlib import Path

# Cache key, so it has to be hashable -- hence a tuple default for `index`
# rather than the list `set_index` ultimately wants.
_DEFAULT_INDEX: tuple[str, ...] = ('ticker','date')


@lru_cache(maxsize=None)
def _read_cached (path: str, date: str|None,
                  index: tuple[str, ...])-> pd.DataFrame:
    """Parse one CSV. Cached, so the frame here is the process-wide original.

    Private because it hands back the cached object itself. Everything outside
    this module goes through `read_csv_file`, which copies.
    """
    df: pd.DataFrame = pd.read_csv(path)
    if date is not None:
        try:
            df[date] = pd.to_datetime(df[date])
        except KeyError:
            raise KeyError(f"{date} is not in data columns")
    try:
        df.set_index(list(index), inplace=True)
        df.sort_index(inplace=True)
    except KeyError as e:
        raise e
    return df


def read_csv_file (path: Path|str, date: str|None = 'date',
                   index: tuple[str, ...]|list[str] = _DEFAULT_INDEX
                   )-> pd.DataFrame:
    """Read a raw CSV into a sorted, MultiIndexed panel.

    Parses the date column to datetime, sets the index, and sorts it -- the
    sort matters because every downstream `.loc[idx[ticker, d0:d1], :]` slice
    needs a lexsorted index.

    The parse is cached per `(path, date, index)`, so the second caller for a
    given file pays a copy instead of a re-parse. The returned frame is that
    copy and is yours to mutate; the cached original is never handed out. A file
    changed on disk mid-process will not be noticed -- call `clear_panel_cache`
    if that happens, which in practice means after re-running `Data.run`.

    Args:
        path (Path | str): CSV to read, e.g. `constant.monthly_price_path`.
        date (str | None): Name of the column to parse as datetime, or None to
            skip parsing (for files with no date column). Defaults to 'date'.
        index (tuple[str, ...] | list[str]): Columns to use as the index. A list
            is accepted and normalised, since the cache key must be hashable.
            Defaults to `('ticker', 'date')`.

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
    return _read_cached(str(path),date,tuple(index)).copy()


def clear_panel_cache ()-> None:
    """Drop the parsed panels so the next read goes back to disk.

    Needed only when a CSV changes underneath a live process -- after
    `Data.run` rewrites it, or in a test that writes a fixture to a path an
    earlier test already read.
    """
    _read_cached.cache_clear()


def book_equity (df: pd.DataFrame)-> pd.Series:
    """Common book equity from an accounting panel: BE = book_value - preferred.

    The one definition of book equity in the codebase. It exists as a function
    rather than as a line repeated at each use because the deduction is not
    symmetric and the asymmetry is invisible at the call site: preferred stock
    *is* deducted, non-controlling interests are *not*.

    NCI is already out. `constant.ratio` nets it off when it collects
    'book_value' -- 'StockholdersEquity' is parent-only to begin with, and the
    fallback tag that includes NCI carries a '-MinorityInterest' term. The
    'minority_interest' column in the panel is that same figure kept for
    reference, so subtracting it here would remove NCI a second time. That is
    not hypothetical: it used to flip the sign of book equity on ~200 rows and
    drop those firms from the universe as if they had negative equity. Preferred
    stock gets no such treatment upstream, so it is deducted here.

    Args:
        df (pd.DataFrame): Accounting panel from `read_csv_file`, with a
            'book_value' column and optionally 'preferred_stock'. A frame
            without 'preferred_stock' is read as having none to deduct rather
            than as an error, since `constant.optional_ratio_fields` lets that
            column be absent.

    Returns:
        pd.Series: Book equity on `df`'s own index.

    Raises:
        KeyError: If `df` has no 'book_value' column.

    Example:
        >>> acc = read_csv_file(accounting_path)
        >>> book_equity(acc).loc['A'].head(2)
        date
        2008-09-30    3.283000e+09
        2009-06-30    3.238000e+09
        dtype: float64
    """
    if 'book_value' not in df.columns:
        raise KeyError("'book_value' is not in data columns")
    be: pd.Series = df['book_value']
    if 'preferred_stock' in df.columns:
        be = be - df['preferred_stock']
    return be

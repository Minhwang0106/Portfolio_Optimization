"""The panels every backtest reads, independent of which model is being run.

Three things: the candidate universe per formation date, the realised monthly
return each weight actually earns, and the risk-free rate the Sharpe ratio is
measured against. All three models are handed the *same* universe -- the one
`Data.exclusion.applicable_ticker` wrote -- so a performance difference between
them is a difference in the weighting rule and not in what they were allowed to
hold. Each model then narrows that candidate list its own way (PPP screens for
complete characteristics, `RIM_PortOp` for usable quarters, EPO takes it whole),
which is a property of the model and is reported rather than equalised.

Cached on the parse, like `src.panel`, and handed back as a copy for the same
reason: a model that mutates its inputs must not be able to rewrite another's.
"""
import pandas as pd
from functools import lru_cache
from pathlib import Path
from constant import (
    applicable_ticker_path, monthly_price_path, monthly_FF5_path
)
from ...panel import read_csv_file


@lru_cache(maxsize=None)
def _universe_cached (path: str)-> dict[pd.Timestamp, tuple[str, ...]]:
    """Parse the applicable-ticker table. Cached, so this is the original.

    Private because it hands back the cached object itself; `universe` copies.
    Tuples rather than lists so nothing a caller does to one date's list can
    reach the cache.
    """
    df: pd.DataFrame = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    out: dict[pd.Timestamp, tuple[str, ...]] = {}
    for date, tickers in zip(df['date'], df['tickers']):
        # A date with no qualifying name is written as an empty field, which
        # pandas reads back as NaN rather than ''. Skipped rather than stored
        # empty, so `universe.get(date)` distinguishes "no universe" from a
        # universe that happens to be empty.
        if not isinstance(tickers, str):
            continue
        out[date] = tuple(t for t in tickers.split(',') if t)
    return out


def universe (path: Path|str = applicable_ticker_path
              )-> dict[pd.Timestamp, list[str]]:
    """The candidate universe at each formation date.

    Reads the table `Data.run` writes from `Data.exclusion.applicable_ticker`:
    a candidate pool intersected with complete price history over the trailing
    `constant.n_month` months, complete accounting data over the trailing
    `constant.n_quarter` quarters, and strictly positive book equity and revenue
    across that window.

    Which pool depends on `constant.universe_asof` at the time the table was
    written, and the file does not record which -- it is the same two columns
    either way. `None` means the pool is S&P 500 membership as of each date;
    a date means it was frozen to the names applicable then, so the universe
    only ever shrinks. Check the constant before reading a count off this.

    Args:
        path (Path | str): Applicable-ticker CSV, columns ['date', 'tickers']
            with the second comma-separated. Defaults to
            `constant.applicable_ticker_path`.

    Returns:
        dict[pd.Timestamp, list[str]]: Sorted ticker list per date. Dates the
            table leaves blank are absent from the mapping rather than mapping
            to an empty list.

    Example:
        >>> u = universe()
        >>> len(u[pd.Timestamp('2020-03-31')])
        273
    """
    return {date: list(tickers)
            for date, tickers in _universe_cached(str(path)).items()}


def monthly_return (price_path: Path|str = monthly_price_path)-> pd.DataFrame:
    """Realised total return per ticker per month, wide.

    Off `adj_close`, so dividends and splits are both in it -- this is the
    return a weight held from one month end to the next actually earns, and it
    is the only thing in the backtest that is not a model input. `close` would
    understate it by the dividend yield, which over eleven years is most of the
    difference between the strategies being compared.

    Args:
        price_path (Path | str): Monthly price panel CSV, indexed
            (ticker, date), with an 'adj_close' column. Defaults to
            `constant.monthly_price_path`.

    Returns:
        pd.DataFrame: Indexed by month end with one column per ticker. The row
            at date `t` is the return earned *over* the month ending at `t`, so
            a weight formed at `t` earns row `t + MonthEnd(1)`. NaN wherever the
            ticker was not priced in both months.

    Example:
        >>> r = monthly_return()
        >>> r.loc['2020-03-31', 'AAPL']
        np.float64(-0.06699...)
    """
    price: pd.DataFrame = read_csv_file(price_path)
    wide: pd.DataFrame = price['adj_close'].unstack(level='ticker').sort_index()
    return wide.pct_change()


def risk_free (rf_path: Path|str = monthly_FF5_path)-> pd.Series:
    """Monthly risk-free rate in decimals, indexed by month end.

    Kenneth French's `RF` column, which is a monthly rate quoted in percent on a
    YYYYMM date. Snapped to month end so it aligns with the return panel index
    directly rather than through a merge.

    Args:
        rf_path (Path | str): Monthly Fama-French factor CSV, with a date column
            in YYYYMM and an 'RF' column in percent. Defaults to
            `constant.monthly_FF5_path`.

    Returns:
        pd.Series: Named 'rf', indexed by month end.

    Example:
        >>> risk_free().loc['2020-03-31']
        np.float64(0.0013)
    """
    df: pd.DataFrame = pd.read_csv(rf_path)
    if 'date' not in df.columns:
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    date_str: pd.Series = df['date'].astype(str).str.strip()
    # The monthly file carries no day of month; the daily one does. Handling
    # both means a caller can point this at `daily_FF5_path` for a sanity check
    # without it silently parsing YYYYMMDD as an out-of-range YYYYMM.
    if date_str.str.len().eq(6).all():
        parsed = pd.to_datetime(date_str, format='%Y%m') + pd.offsets.MonthEnd(0)
    else:
        parsed = pd.to_datetime(date_str)
    rf: pd.Series = (df['RF'].astype(float)/100).set_axis(parsed).rename('rf')
    rf.index.name = 'date'
    return rf[~rf.index.duplicated(keep='last')].sort_index()


def clear_backtest_cache ()-> None:
    """Drop the parsed universe table so the next read goes back to disk.

    Needed only when `Data.run` rewrites `applicable_ticker.csv` underneath a
    live process -- which `Data.run.run_applicable_ticker` does every time
    `constant.universe_asof` changes, so call this after it. The price and factor
    panels are not cached here -- the first goes through `src.panel`, so
    `panel.clear_panel_cache` covers it, and the second is re-read on every call.
    """
    _universe_cached.cache_clear()
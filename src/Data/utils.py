import pandas as pd
from collections import defaultdict
from constant import (
    quarter_span_days,
    scale_screen_field, scale_screen_tolerance,
    scale_fix_powers, scale_fix_agreement, scale_fix_rel_error,
)


def dedup_facts(raw_data: list[dict]) -> dict:
    """Collapse a list of SEC fact entries into one value per end-date.

    When multiple entries share the same 'end' date, an amended filing
    ('form' ending in '/A') takes precedence over the original.

    Args:
        raw_data (list[dict]): Fact entries, each with keys 'end', 'val',
            'form' (as returned by the SEC companyfacts API).

    Returns:
        dict: Mapping of end-date string to value.

    Example:
        >>> dedup_facts([
        ...     {'end': '2020-12-31', 'val': 100, 'form': '10-K'},
        ...     {'end': '2020-12-31', 'val': 110, 'form': '10-K/A'},
        ... ])
        {'2020-12-31': 110}
    """
    date_to_val: dict = {}
    for inf in raw_data:
        end_date = inf['end']
        if end_date not in date_to_val or inf['form'].endswith('/A'):
            date_to_val[end_date] = inf['val']
    return date_to_val


def is_quarter_span(start: str, end: str,
                    span: tuple[int, int] = quarter_span_days) -> bool:
    """Test whether two dates are one fiscal quarter apart.

    Args:
        start (str): ISO start date (or the earlier of two end-dates).
        end (str): ISO end date.
        span (tuple[int, int]): Inclusive (min, max) day count that counts
            as a quarter. Defaults to `constant.quarter_span_days`.

    Returns:
        bool: True if `end - start` falls inside `span`.

    Example:
        >>> is_quarter_span('2020-01-01', '2020-03-31')
        True
        >>> is_quarter_span('2020-01-01', '2020-06-30')
        False
    """
    low, high = span
    return low <= (pd.Timestamp(end) - pd.Timestamp(start)).days <= high


def quarterly_facts(raw_data: list[dict]) -> dict:
    """Reduce duration facts to one single-quarter value per end-date.

    Flow concepts (revenue, net income) are reported by the SEC over several
    overlapping periods sharing one 'end' date -- the quarter, the
    year-to-date figure, and in the 10-K the full year. Keying on 'end'
    alone therefore picks an arbitrary horizon, which is what made revenue
    growth compare a year against a quarter.

    Two passes fix that. A fact already spanning one quarter is taken as it
    stands. An end-date covered only by cumulative facts -- the fourth
    quarter, which most filers never tag on its own -- is recovered by
    differencing the year-to-date chain: within one 'start', consecutive
    end-dates a quarter apart give `Q4 = FY - 9M`.

    Args:
        raw_data (list[dict]): Duration fact entries, each with keys
            'start', 'end', 'val', 'form'.

    Returns:
        dict: Mapping of end-date string to that quarter's value. End-dates
            whose quarterly figure can be neither read nor derived are
            omitted rather than guessed.

    Example:
        >>> quarterly_facts([
        ...     {'start': '2020-01-01', 'end': '2020-09-30',
        ...      'val': 90, 'form': '10-Q'},
        ...     {'start': '2020-01-01', 'end': '2020-12-31',
        ...      'val': 130, 'form': '10-K'},
        ... ])
        {'2020-12-31': 40}
    """
    # (start, end) -> val, so overlapping horizons stop colliding. An
    # amended filing still wins over the original it restates.
    by_period: dict = {}
    for inf in raw_data:
        period = (inf['start'], inf['end'])
        if period not in by_period or inf['form'].endswith('/A'):
            by_period[period] = inf['val']

    date_to_val: dict = {}
    for (start, end), val in sorted(by_period.items()):
        if is_quarter_span(start, end):
            date_to_val.setdefault(end, val)

    ends_by_start: dict = defaultdict(set)
    for start, end in by_period:
        ends_by_start[start].add(end)
    for start in sorted(ends_by_start):
        ends: list = sorted(ends_by_start[start])
        for prev_end, end in zip(ends, ends[1:]):
            if end in date_to_val or not is_quarter_span(prev_end, end):
                continue
            date_to_val[end] = by_period[(start, end)] - by_period[
                                                        (start, prev_end)]
    return date_to_val


def facts_to_dates(raw_data: list[dict]) -> dict:
    """Reduce fact entries to one value per end-date, by concept type.

    Dispatches on the shape of the facts themselves rather than on a
    hardcoded list of tags: SEC instant concepts (balance-sheet items such
    as Assets) carry no 'start', duration concepts (income-statement flows)
    do. Instants are deduplicated on their end-date; durations go through
    `quarterly_facts` so every value covers exactly one quarter.

    Args:
        raw_data (list[dict]): Fact entries for a single tag.

    Returns:
        dict: Mapping of end-date string to value.

    Example:
        >>> facts_to_dates([{'end': '2020-12-31', 'val': 100, 'form': '10-K'}])
        {'2020-12-31': 100}
    """
    durations: list[dict] = [inf for inf in raw_data if 'start' in inf]
    if not durations:
        return dedup_facts(raw_data)
    instants: list[dict] = [inf for inf in raw_data if 'start' not in inf]
    date_to_val: dict = quarterly_facts(durations)
    date_to_val.update({date: val for date, val
                        in dedup_facts(instants).items()
                        if date not in date_to_val})
    return date_to_val


def dates_to_df(date_to_val: dict, ticker: str) -> pd.DataFrame:
    """Wrap an end-date -> value mapping in a datetime-indexed DataFrame.

    Args:
        date_to_val (dict): Mapping of ISO end-date string to value.
        ticker (str): Column name to use for the values.

    Returns:
        pd.DataFrame: Indexed by datetime 'date', with one column named
            `ticker`.

    Example:
        >>> dates_to_df({'2020-12-31': 100}, 'AAPL')
                    AAPL
        date
        2020-12-31   100
    """
    df: pd.DataFrame = pd.DataFrame(
        list(date_to_val.values()), columns=[ticker],
        index=pd.Index(list(date_to_val.keys()), name='date'))
    df.index = pd.to_datetime(df.index)
    # Values arrive keyed by a dict, so sort to keep the frame -- and the
    # duplicate-dropping that follows the quarter snap -- deterministic.
    return df.sort_index()


def listdict_to_df(raw_data: list[dict], ticker: str) -> pd.DataFrame:
    """Convert a list of SEC fact entries into a single-column DataFrame.

    Duration facts are reduced to single-quarter values by `facts_to_dates`
    before the frame is built, so the column never mixes reporting horizons.

    Args:
        raw_data (list[dict]): Fact entries with keys 'end', 'val', 'form',
            plus 'start' for duration concepts.
        ticker (str): Column name to use for the values.

    Returns:
        pd.DataFrame: Indexed by datetime 'date', with one column named
            `ticker` holding one value per end-date.

    Example:
        >>> listdict_to_df(
        ...     [{'end': '2020-12-31', 'val': 100, 'form': '10-K'}], 'AAPL')
                    AAPL
        date
        2020-12-31   100
    """
    return dates_to_df(facts_to_dates(raw_data), ticker)


def to_calendar_quarter(date: pd.Timestamp) -> pd.Timestamp:
    """Snap a filing end-date to the nearest standard calendar-quarter end.

    Fiscal periods reported by companies don't always land exactly on
    calendar quarter-ends; this buckets a date into whichever of
    Mar 31 / Jun 30 / Sep 30 / Dec 31 it belongs to (Jan is treated as
    belonging to the prior year's Dec 31).

    Args:
        date (pd.Timestamp): The reported end-date.

    Returns:
        pd.Timestamp: The corresponding calendar quarter-end date.

    Example:
        >>> to_calendar_quarter(pd.Timestamp('2020-02-15'))
        Timestamp('2020-03-31 00:00:00')
    """
    month, year = date.month, date.year
    if month == 1:
        year, month, day = year - 1, 12, 31
    elif month in (11,12):
        month, day = 12, 31
    elif month in (2,3,4):
        month, day = 3, 31
    elif month in (5,6,7):
        month, day = 6, 30
    else:
        month, day = 9, 30
    result = pd.Timestamp(year=year, month=month, day=day)
    assert isinstance(result, pd.Timestamp)
    return result


def fix_misscaled_rows(df: pd.DataFrame, required: list[str],
                       powers: tuple[int, ...] = scale_fix_powers,
                       agreement: float = scale_fix_agreement,
                       rel_error: float = scale_fix_rel_error
                       ) -> pd.DataFrame:
    """Rescale filings tagged in thousands or millions but declared as USD.

    A handful of filers report an entire statement in scaled units while the
    XBRL fact still claims plain USD, which puts every field of that one row
    1e3 or 1e6 below the rest of the ticker's history (LUV's 2011-06-30 10-Q
    reports revenue of 4136 between quarters of 3.10e9 and 4.31e9).

    The row has to clear a deliberately strict test before anything is
    touched: every required field present, all of them below their own
    median by factors agreeing within `agreement`, and the shared factor
    within `rel_error` of an exact power of 1000. Ordinary business
    variation cannot satisfy that -- a bad quarter moves earnings, not the
    balance sheet and the income statement together by the same 1e6.

    Rescaling rather than dropping matters because the quarter that follows
    a dropped row loses its own growth rate too, the predecessor being gone.

    Args:
        df (pd.DataFrame): Per-ticker frame indexed by date, one column per
            line item.
        required (list[str]): Field names that must all be present and
            non-null in a row for it to be considered.
        powers (tuple[int, ...]): Scale factors to recognise. Defaults to
            `constant.scale_fix_powers`.
        agreement (float): Maximum ratio between the largest and smallest
            per-field factor. Defaults to `constant.scale_fix_agreement`.
        rel_error (float): Maximum relative distance from an exact power.
            Defaults to `constant.scale_fix_rel_error`.

    Returns:
        pd.DataFrame: `df` with any such row multiplied back up.

    Example:
        >>> fix_misscaled_rows(df, ['revenue']).loc['2011-06-30', 'revenue']
        4136000000.0
    """
    fields: list[str] = [c for c in required if c in df.columns]
    if len(fields) < 2 or len(df) < 4:
        return df
    values: pd.DataFrame = df[fields].abs()
    median: pd.Series = values.median()
    if (median <= 0).any():
        return df
    factor: pd.DataFrame = median/values.where(values > 0)
    # Every field has to agree, which is what separates a tagging error from
    # a genuinely small quarter.
    usable: pd.Series = factor.notna().all(axis=1)
    spread: pd.Series = factor.max(axis=1)/factor.min(axis=1)
    shared: pd.Series = factor.median(axis=1)
    for power in powers:
        hit: pd.Series = (usable & (spread <= agreement)
                          & ((shared/power - 1).abs() <= rel_error))
        if hit.any():
            df.loc[hit, fields] = df.loc[hit, fields]*power
    return df


def drop_placeholder_rows(df: pd.DataFrame,
                          field: str = scale_screen_field,
                          tolerance: float = scale_screen_tolerance
                          ) -> pd.DataFrame:
    """Drop filings that carry a placeholder balance sheet.

    A newly-registered holding company files its first statements before the
    operating merger closes, reporting the predecessor's income statement
    against a balance sheet of essentially nothing -- ICE files total assets
    of 10, SW 111, TRIP 1000, QRVO and VTRS 0. Left in, they make asset
    turnover and price-to-book explode by six to eight orders of magnitude.

    Detected on the fact that a real balance-sheet item cannot collapse
    `tolerance`-fold from one quarter to the next and recover.

    Args:
        df (pd.DataFrame): Per-ticker frame indexed by date.
        field (str): Balance-sheet column to screen on. Defaults to
            `constant.scale_screen_field`. A frame without it is returned
            untouched.
        tolerance (float): How far below the ticker's median the field has
            to fall. Defaults to `constant.scale_screen_tolerance`.

    Returns:
        pd.DataFrame: `df` without the offending rows.

    Example:
        >>> len(drop_placeholder_rows(ice_df))
        51
    """
    if field not in df.columns or len(df) < 4:
        return df
    values: pd.Series = df[field].abs()
    median: float = values.median()
    if not median > 0:
        return df
    return df[~(values < median/tolerance)]

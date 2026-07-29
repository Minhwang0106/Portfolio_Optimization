"""Put the back-adjusted price and the point-in-time share count on one basis.

Yahoo back-adjusts prices for splits server-side, so `close` at every date is
stated in *today's* shares -- AAPL's 2020-08-28 close is 124.81, not the
499.23 that traded. The share count comes from SEC filings and is
point-in-time. Multiplying them gives a market cap too small by the split
factor for every date preceding a split, which corrupts `pb` (and so
`Data.exclusion`), `bvps`, `rps`, and the price the RIM values against.

The correction is not a matter of applying the split on its official date.
The share count changes basis when a *filing* reports the new count, and that
boundary lands on either side of the split:

    LRCX    split 2024-10-03    shares change basis 2024-09-30   (3 days early)
    GOOGL   split 2022-07-18    shares change basis 2022-06-30   (18 days early)
    AMZN    split 2022-06-06    shares change basis 2022-06-30   (24 days late)
    NVDA    split 2024-06-10    shares change basis 2024-07-31   (7 weeks late)
    AAPL    split 2020-08-31    shares change basis 2020-09-30   (1 month late)

Keying off the official date would therefore double-count LRCX and GOOGL --
their share counts are already post-split before the split happens -- while
handling AAPL and NVDA correctly. So the shift date is detected from the share
series itself, and the split catalogue is used only to confirm that a jump is
a split rather than a buyback or a secondary offering. The property being
restored is that market cap is continuous across the boundary, which is also
what `verify_continuity` tests.
"""
import numpy as np
import pandas as pd
from constant import split_search_days, split_ratio_tolerance


def basis_shift_dates(shares: pd.Series, events: pd.Series,
                      search_days: int = split_search_days,
                      ratio_tolerance: float = split_ratio_tolerance
                      ) -> list[tuple[pd.Timestamp, float, bool]]:
    """Find when a ticker's share count actually changes split basis.

    For each split, the ratio between consecutive share counts is searched
    within `search_days` either side of the official date, and the closest
    match to the split ratio is taken as the shift date. Matching on the
    ratio is what separates a split from a buyback: a 20x jump is never a
    repurchase, and a 1.5x one is resolved by the catalogue saying a 1.5:1
    split happened there.

    Args:
        shares (pd.Series): Share count for one ticker, indexed by date.
        events (pd.Series): Split ratios for that ticker, indexed by official
            effective date.
        search_days (int): Half-width of the search window. Defaults to
            `constant.split_search_days`, wide enough for a two-quarter
            filing lag.
        ratio_tolerance (float): Maximum relative distance between the
            observed jump and the split ratio. Defaults to
            `constant.split_ratio_tolerance`; observed jumps run ~0.5% off
            (LRCX 9.948 for a 10:1) because buybacks move the count too.

    Returns:
        list[tuple[pd.Timestamp, float, bool]]: One (shift_date, ratio,
            matched) per split. `matched` is False when no jump could be
            found -- the official date is used instead, which is the right
            fallback for splits predating the share history entirely.

    Example:
        >>> basis_shift_dates(aapl_shares, aapl_splits)[-1]
        (Timestamp('2020-09-30'), 4.0, True)
    """
    series: pd.Series = shares.dropna().sort_index()
    if len(series) < 2 or not len(events):
        return [(date, ratio, False) for date, ratio in events.items()]

    step: pd.Series = (series/series.shift(1)).dropna()
    shifts: list[tuple[pd.Timestamp, float, bool]] = []
    for split_date, ratio in events.items():
        window: pd.Series = step[
            (step.index > split_date - pd.Timedelta(days=search_days))
            & (step.index <= split_date + pd.Timedelta(days=search_days))]
        if window.empty:
            shifts.append((split_date, float(ratio), False))
            continue
        error: pd.Series = (window/ratio - 1).abs()
        best: pd.Timestamp = error.idxmin()
        if error[best] <= ratio_tolerance:
            shifts.append((best, float(ratio), True))
        else:
            shifts.append((split_date, float(ratio), False))
    return shifts


def split_factor(index: pd.DatetimeIndex,
                 shifts: list[tuple[pd.Timestamp, float, bool]]) -> pd.Series:
    """Cumulative factor restating a point-in-time share count on today's basis.

    A date strictly before a shift is quoted in pre-split shares and needs
    multiplying by the ratio; a date on or after it is already on the new
    basis. Factors compound, so a ticker that has split twice since a date
    carries the product.

    Unmatched events are skipped entirely rather than applied at their official
    date. Yahoo reports spinoffs in the same feed as splits -- Agilent's
    Keysight separation appears as a 1.398 "split" -- and a spinoff leaves the
    share count alone, so no jump exists to match and applying the ratio would
    scale every pre-spinoff count by a factor that never happened. The other
    reason an event goes unmatched is that it predates the share history
    entirely (AAPL's 2005 split, against filings starting 2009), and there
    skipping costs nothing because no row precedes it. The one case skipping
    gets wrong is a genuine split the filings never restated, which is rare
    (RMD 2010, STLD 2008) and indistinguishable from a spinoff on this data.

    Args:
        index (pd.DatetimeIndex): Dates to compute the factor for.
        shifts (list[tuple[pd.Timestamp, float, bool]]): Output of
            `basis_shift_dates`.

    Returns:
        pd.Series: Factor per date, 1.0 where no matched split follows.

    Example:
        >>> split_factor(pd.DatetimeIndex(['2020-08-31', '2020-09-30']),
        ...              [(pd.Timestamp('2020-09-30'), 4.0, True)]).tolist()
        [4.0, 1.0]
    """
    factor: pd.Series = pd.Series(1.0, index=index)
    for shift_date, ratio, matched in shifts:
        if not matched:
            continue
        factor[index < shift_date] *= ratio
    return factor


def adjust_shares(shares: pd.Series, splits: pd.DataFrame,
                  search_days: int = split_search_days,
                  ratio_tolerance: float = split_ratio_tolerance
                  ) -> pd.Series:
    """Restate a whole share panel on the same basis as the adjusted price.

    Args:
        shares (pd.Series): Share count with a ('ticker', 'date') MultiIndex.
        splits (pd.DataFrame): Long-format split events, columns ['ticker',
            'date', 'split_ratio'], as written by `src.Data.run.run_splits`.
        search_days (int): Passed to `basis_shift_dates`.
        ratio_tolerance (float): Passed to `basis_shift_dates`.

    Returns:
        pd.Series: `shares` multiplied by its cumulative split factor,
            aligned to the input index. Tickers with no splits are returned
            unchanged, so this is safe to apply to the whole panel.

    Example:
        >>> adjust_shares(shares, splits).xs('AAPL', level='ticker')[
        ...     '2020-08-31']
        17102552000.0
    """
    if not len(splits):
        return shares.copy()

    events: pd.DataFrame = splits.copy()
    events['date'] = pd.to_datetime(events['date'])
    by_ticker: dict = {
        tick: frame.set_index('date')['split_ratio'].sort_index()
        for tick, frame in events.groupby('ticker')
    }

    pieces: list[pd.Series] = []
    for tick, group in shares.groupby(level='ticker'):
        dates: pd.DatetimeIndex = pd.DatetimeIndex(
            group.index.get_level_values('date'))
        ticker_events: pd.Series | None = by_ticker.get(tick)
        if ticker_events is None:
            pieces.append(group)
            continue
        shifts = basis_shift_dates(
            pd.Series(group.to_numpy(), index=dates), ticker_events,
            search_days, ratio_tolerance)
        factor: pd.Series = split_factor(dates, shifts)
        pieces.append(pd.Series(group.to_numpy()*factor.to_numpy(),
                                index=group.index, name=shares.name))

    return pd.concat(pieces).reindex(shares.index)


def verify_continuity(price: pd.Series, shares: pd.Series,
                      raw_shares: pd.Series, splits: pd.DataFrame,
                      tolerance: float = 0.35) -> pd.DataFrame:
    """Check that market cap no longer jumps at a split's basis boundary.

    The test the adjustment exists to pass: a split changes the share count
    and the price together, so market cap should pass through it smoothly.

    The measurement is the single step across the detected boundary, not the
    largest step near it. Scanning a window instead reports every unrelated
    move that happens to fall within a couple of quarters of a split -- a
    volatile month, an isolated bad share count (IDXX carries 102.3M at
    2015-03-31 between a correct 46.8M and 92.6M), or a spinoff, all of which
    survive a correct split adjustment and none of which this function is
    able to fix.

    Args:
        price (pd.Series): Adjusted close with a ('ticker', 'date')
            MultiIndex.
        shares (pd.Series): Share count on the same index, already passed
            through `adjust_shares` -- the series under test.
        raw_shares (pd.Series): The unadjusted share count. Needed because
            the boundary is located by the jump in the share series, and
            `adjust_shares` is precisely what removes that jump.
        splits (pd.DataFrame): Long-format split events.
        tolerance (float): Maximum absolute log change in market cap across
            the boundary before it is reported. Defaults to 0.35, wide enough
            for a genuinely volatile month and far below the log(2) a missed
            2:1 split would leave.

    Returns:
        pd.DataFrame: One row per boundary still showing a jump, with columns
            ['ticker', 'date', 'split_ratio', 'log_change', 'matched'].
            `matched` False means no share jump was found for that event, so
            it was never adjusted -- usually a spinoff, which Yahoo reports
            in the same feed but which leaves the share count alone.

    Example:
        >>> verify_continuity(price, adjusted, shares, splits).empty
        True
    """
    market_cap: pd.Series = (price*shares).dropna()
    events: pd.DataFrame = splits.copy()
    events['date'] = pd.to_datetime(events['date'])

    rows: list[dict] = []
    for tick, group in market_cap.groupby(level='ticker'):
        ticker_events: pd.DataFrame = events[events['ticker'] == tick]
        if ticker_events.empty:
            continue
        dates: pd.DatetimeIndex = pd.DatetimeIndex(
            group.index.get_level_values('date'))
        series: pd.Series = pd.Series(group.to_numpy(),
                                      index=dates).sort_index()
        with np.errstate(divide='ignore', invalid='ignore'):
            change: pd.Series = np.log(series/series.shift(1)).abs()
        change = change.replace([np.inf, -np.inf], np.nan).dropna()

        ticker_shares: pd.Series = pd.Series(
            raw_shares.xs(tick, level='ticker').to_numpy(),
            index=pd.DatetimeIndex(
                raw_shares.xs(tick, level='ticker').index)).sort_index()
        shifts = basis_shift_dates(
            ticker_shares,
            ticker_events.set_index('date')['split_ratio'].sort_index())

        for shift_date, ratio, matched in shifts:
            if shift_date not in change.index:
                continue
            if change[shift_date] <= tolerance:
                continue
            rows.append({'ticker': tick, 'date': shift_date,
                         'split_ratio': ratio,
                         'log_change': change[shift_date],
                         'matched': matched})
    return pd.DataFrame(rows, columns=['ticker', 'date', 'split_ratio',
                                       'log_change', 'matched'])
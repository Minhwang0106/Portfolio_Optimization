import pandas as pd
import numpy as np
from pathlib import Path
from constant import (
    accounting_path, monthly_price_path, splits_path, n_quarter,
    n_quarter_ahead
)
from ..panel import read_csv_file, book_equity
from ..Data.split_adjust import adjust_shares
from .utils.variable_tranformation import Ros_Trans, Ate_Trans, Ato_Trans

def _adjusted_shares (panel: pd.DataFrame, split_path: Path)-> pd.Series:
    """Split-adjusted share count for a panel, stored column preferred.

    `Data.run.run_share_adjustment` writes `adj_shares` alongside the filed
    `shares`, and reading it keeps every consumer on one definition. Computing
    it here when the column is absent means a panel written before that step
    still values correctly -- the alternative, falling through to `shares`,
    would reintroduce the split mismatch silently, which is the failure this
    whole path exists to prevent.

    Args:
        panel (pd.DataFrame): Panel indexed ('ticker', 'date') carrying a
            'shares' column, and optionally 'adj_shares'.
        split_path (Path): Split events CSV, used only on the fallback path.

    Returns:
        pd.Series: Share count on today's split basis.

    Raises:
        KeyError: If the panel has no 'shares' column at all.
    """
    if 'adj_shares' in panel.columns:
        return panel['adj_shares']
    return adjust_shares(panel['shares'], pd.read_csv(split_path))

def generator (acc_path: Path = accounting_path,
               price_path: Path = monthly_price_path,
               split_path: Path = splits_path
               )->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,
                        pd.DataFrame]:
    """Build the characteristic panel, plus the per-share series RIM values on.

    The first return is the panel the copulas are fitted on -- four DuPont-style
    decompositions of the quarterly accounting panel:

    * `ros` -- return on sales, `net_income / revenue`.
    * `ato` -- asset turnover, `revenue / total_assets`.
    * `ate` -- assets to equity, `total_assets / book_value`, where book value is
      net of minority interest and preferred stock (clean-surplus BE, the same
      definition `PPP.input_generator.generator` uses).
    * `g` -- log revenue growth against the *immediately preceding* quarter.

    Three of the four leave here *transformed*, not on their natural scale:
    `ros`, `ato` and `ate` are pushed onto the whole real line by
    `utils.variable_tranformation`, because the conditional law in
    `utils.sampling_distribution` is Gaussian and would otherwise put mass where
    the ratio cannot go (`ato <= 0`, `ate <= 1`, `ros >= 1`). Anything reading a
    simulated path back as a ratio has to invert them -- `RIM_PortOp.sampling`
    does, through `RIM_PortOp.activate_inverse_func`. `g` is already unbounded
    and is left alone. A value the transform is undefined on becomes NaN, which
    is the same treatment the ratios' own division by zero gets.

    The remaining four returns are not characteristics and are not transformed;
    they are the inputs the residual income valuation needs, each unstacked wide
    on tickers. `close` is monthly, straight off the price panel. The other
    three are quarterly: `bvps` and `rps` because the accounting panel is, and
    `pb` because it needs book value, so it survives only on the quarter-end
    dates the two panels share.

    Args:
        acc_path (Path): Accounting panel CSV, indexed (ticker, date), with
            columns 'net_income', 'revenue', 'total_assets', 'book_value',
            'minority_interest', 'preferred_stock', 'shares'. Defaults to
            `constant.accounting_path`.
        price_path (Path): Monthly price panel CSV, indexed (ticker, date), with
            columns 'close', 'shares'. Defaults to `constant.monthly_price_path`.

    Returns:
        tuple[pd.DataFrame, ...]: Five frames, `(df, pb, bvps, rps, close)`.

            * `df` -- the characteristic panel, indexed (ticker, date) and
              lexsorted, with columns ['ros', 'ato', 'ate', 'g'], the first
              three transformed. This is the only one indexed long; it is what
              `take_training_data` slices.
            * `pb` -- price to book, `close * shares / book_value`, on the
              price panel's share count.
            * `bvps` -- book value per share, `book_value / shares`, on the
              accounting panel's own share count.
            * `rps` -- revenue per share, on that same share count.
            * `close` -- the monthly close.

            Both share counts are restated on today's split basis first, since
            `close` already is; see `src.Data.split_adjust`. Every per-share
            figure here is therefore per *current* share, not per share as it
            stood on the date -- consistent with the price, which is what the
            valuation needs, but not the figure the filing reported.

            The last four are unstacked wide, i.e. indexed by date with one
            column per ticker, and are the only ones shaped that way.

    Note:
        The characteristics sit on the accounting period's *end* date, not on
        the date the figures were filed, so this frame is not point-in-time as
        returned -- a formation date can see a quarter that was still unfiled.
        `take_training_data` is what applies the publication lag; any other
        consumer has to apply its own.

    Note:
        A handful of filers tag an entire filing in thousands or millions while
        still declaring the unit as USD -- LUV's 2011-06-30 10-Q reports revenue
        as 4136 between quarters of 3.10e9 and 4.31e9. Around 0.2% of the
        accounting cells are affected, and because the error is a factor of
        1e3-1e6 it lands squarely in the tails: the largest `|g|` (13.9 = log
        1e6), untransformed `ato` (3.7e7) and `pb` (6.5e8) values all trace back
        to one of these rows rather than to anything economic. Winsorize, or
        screen on a row's scale against the ticker's own median, before fitting.

    Example:
        >>> df, pb, bvps, rps, close = generator()
        >>> df.loc['A'].head(3)                  # ros, ato, ate transformed
                         ros       ato       ate         g
        date
        2008-09-30  0.169574 -1.554192  0.552838       NaN
        2009-06-30 -0.017816 -1.827536  0.495196       NaN
        2009-09-30  0.021655 -1.875290  0.711728  0.099002
        >>> pb['A'].head(2)                      # wide, and on its own scale
        date
        2008-09-30    3.001251
        2009-06-30    3.274468
        Name: A, dtype: float64
    """
    acc_df: pd.DataFrame = read_csv_file(acc_path)
    price_df: pd.DataFrame = read_csv_file(price_path)

    # Yahoo back-adjusts `close` for splits, so it is quoted in today's shares
    # at every date, while both share counts come from filings and are
    # point-in-time. Left alone the two disagree by the split factor for every
    # date preceding a split -- NVDA's market cap at 2024-06-30 reads $304bn
    # against a true $3.04tn -- which understates market cap, hence `pb`, hence
    # the terminal value, for 19% of the panel. Worse than noise: firms split
    # after the price has risen, so the error marks the eventual winners years
    # in advance, and the optimiser concentrates into exactly them.
    #
    # `adj_shares` is written by `Data.run.run_share_adjustment` and is the
    # filed count restated on today's split basis. Falling back to computing it
    # here keeps this callable against a panel written before that step
    # existed, rather than silently valuing against the unadjusted count.
    price_shares: pd.Series = _adjusted_shares(price_df, split_path)
    acc_shares: pd.Series = _adjusted_shares(acc_df, split_path)

    col_val: dict[str, pd.Series] = {}

    # The transforms are numpy ufuncs, so they map over a Series as they stand;
    # `errstate` is what keeps a ratio outside the transform's domain quiet,
    # since those are already the missing values the ratios themselves produce
    # and the `replace` below sweeps up whichever of NaN or -inf comes back.
    with np.errstate(divide='ignore', invalid='ignore'):
        #Return on Sales
        col_val['ros'] = Ros_Trans.transform(
            acc_df['net_income']/acc_df['revenue'])

        #Assets turnover
        col_val['ato'] = Ato_Trans.transform(
            acc_df['revenue']/acc_df['total_assets'])

        #Assets to Book Values
        book_values: pd.Series = book_equity(acc_df)
        col_val['ate'] = Ate_Trans.transform(acc_df['total_assets']/book_values)

    #Revenue Growth
    # shift(1) walks rows, not calendar quarters, and ~580 rows in the panel have
    # a predecessor two or more quarters back. Masking those keeps `g` a
    # one-quarter growth rate everywhere instead of a mislabelled multi-quarter
    # one.
    dates: pd.DatetimeIndex = pd.DatetimeIndex(
                                    acc_df.index.get_level_values('date'))
    prev_date: pd.Series = pd.Series(dates, index=acc_df.index).groupby(
                                                    level='ticker').shift(1)
    is_adjacent: pd.Series = prev_date == pd.Series(
                        dates - pd.offsets.QuarterEnd(1), index=acc_df.index)
    revenue: pd.Series = acc_df['revenue']
    ratio: pd.Series = revenue/revenue.groupby(level='ticker').shift(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        g: pd.Series = pd.Series(np.log(ratio.to_numpy()), index=ratio.index)
    col_val['g'] = g.where(is_adjacent)

    # The RIM inputs. None of these are characteristics -- they are left on
    # their natural scale and unstacked wide, since the valuation reads them by
    # date across the universe rather than one ticker at a time.
    market_cap: pd.Series = price_df['close']*price_shares
    temp = pd.concat([market_cap,book_values],axis=1,keys=['me','be']).dropna()
    pb_df: pd.DataFrame = (temp['me']/temp['be']).unstack(level='ticker')

    # Per-share figures take the *accounting* share count, not the price
    # panel's: numerator and denominator then come off the same filing, so a
    # quarter the two panels disagree on cannot show up as a jump in bvps.
    # Split-adjusted, because the RIM compares these to `close` directly and
    # `close` is on today's share basis -- an unadjusted bvps would value a
    # pre-split quarter's book value per old share against a price per new one.
    bvps_df: pd.DataFrame = (book_values/acc_shares
                             ).unstack(level='ticker')
    rps_df: pd.DataFrame = (acc_df['revenue']/acc_shares
                            ).unstack(level='ticker')
    close_df: pd.DataFrame = price_df['close'].unstack(level='ticker')

    # `pb_df` comes off a concat of two panels with different (ticker, date)
    # indexes, and the union it forms is not ordered -- 2008-06-30, 2008-12-31
    # and 2009-03-31 land after 2026-06-30. Consumers truncate these at a
    # formation date with `.loc[:date]`, which on an unsorted DatetimeIndex is a
    # positional slice to the first occurrence of the label, not a chronological
    # one: it silently drops the stragglers, and raises outright for any date not
    # already in the index (i.e. every formation date that is not a quarter end).
    pb_df = pb_df.sort_index()
    bvps_df = bvps_df.sort_index()
    rps_df = rps_df.sort_index()
    close_df = close_df.sort_index()

    # The same sweep `df` gets below, for the same reason: `me/be` and the two
    # per-share ratios divide by a quantity that reaches zero. KDP's book equity
    # is exactly 0 at 2015-12-31, which puts a lone +inf in `pb`, and
    # `Data.exclusion` does not catch it -- that screen tests `~(be < 0)`, and 0
    # is not negative. One infinite P/B is enough to kill a whole ticker:
    # `RIM_PortOp.joint_return` reads it as `max_pb`, and `terminal_val` then
    # computes (beT + inf)/(be0 + dri + inf) = inf/inf = NaN on every path.
    for frame in (pb_df, bvps_df, rps_df, close_df):
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)

    #Compose
    df: pd.DataFrame = pd.concat(col_val.values(),axis=1,keys=col_val.keys())
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Lexsorted on (ticker, date) -- `take_training_data` slices level 'date'
    # with a range, which pandas rejects on anything less.
    df.sort_index(inplace=True)
    return df, pb_df, bvps_df, rps_df, close_df

def accounting_cutoff (date: str|pd.Timestamp)->pd.Timestamp:
    """The most recent quarter whose filing was already public on `date`.

    The single definition of "what the formation date is allowed to have read",
    so that every consumer of the accounting panel lands on the same quarter.
    `take_training_data` ends its training window here and
    `RIM_PortOp.joint_return` takes its revenue per share and its price-to-book
    extrema from here; when the two computed it separately they disagreed on
    every date that is not a quarter end -- 88 of the 132 in
    `constant.testing_period` -- which both compounded simulated growth off the
    wrong base and read a quarter that may not have been filed.

    Args:
        date (str | pd.Timestamp): Formation date. Parsed with `pd.Timestamp` if
            given as a string.

    Returns:
        pd.Timestamp: The quarter end the formation date may read up to and
            including.

    Note:
        A formation date that is itself a quarter end backs off one quarter; any
        other date backs off two. The asymmetry is deliberate conservatism -- a
        mid-quarter date sits close enough to the 40-45 day filing deadline that
        the just-ended quarter may not be public yet -- but it does mean that at,
        say, 2020-05-15 the cutoff is 2019-12-31 even though Q1 was filed around
        2020-05-10.

    Example:
        >>> accounting_cutoff('2020-06-30').date()   # a quarter end
        datetime.date(2020, 3, 31)
        >>> accounting_cutoff('2020-05-31').date()   # mid-quarter
        datetime.date(2019, 12, 31)
    """
    if isinstance(date,str):
        date = pd.Timestamp(date)
    quarters_back: int = 1 if date.is_quarter_end else 2
    return date - pd.offsets.QuarterEnd(quarters_back)

def take_training_data (df:pd.DataFrame, date: str|pd.Timestamp,
                        ticker: list[str], nq: int = n_quarter,
                        nqf: int = n_quarter_ahead
                        )->tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Split the characteristic panel into a training window and a future window.

    Both windows are anchored to `date`, the formation date. The training window
    ends at the most recent quarter whose filing would already have been public
    on `date`, so the split carries no lookahead; the future window starts at the
    quarter after that and holds what the model is asked to predict. The two
    never overlap.

    Args:
        df (pd.DataFrame): Characteristic panel from `generator`, i.e. its
            *first* return, indexed (ticker, date). Re-sorted here if it is not
            already lexsorted; the caller's frame is never modified.
        date (str | pd.Timestamp): Formation date. Parsed with `pd.Timestamp` if
            given as a string.
        ticker (list[str]): Universe to slice. Names absent from `df` are
            dropped rather than raising, since a time-varying universe routinely
            contains tickers with no rows in a given window.
        nq (int): Number of quarters in the training window, counting both ends.
            Defaults to `constant.n_quarter`.
        nqf (int): Number of quarters in the future window, counting both ends.
            Defaults to `constant.n_quarter_ahead`.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, list[str]]: `(df_train, df_future,
            ticker)`, the first two each a slice of `df` over the requested
            tickers and its window, and the third `ticker` narrowed to the names
            that have at least one row in the *training* window, which the caller
            needs since it is not the list it passed in. `df_future` is on that
            same universe but is not what narrows it -- a name can survive here
            and still have no future rows, which only matters to
            `RIM_PortOp.sampling(forward=True)`.

    Note:
        A formation date that is itself a quarter end backs off one quarter; any
        other date backs off two. The asymmetry is deliberate conservatism -- a
        mid-quarter date sits close enough to the 40-45 day filing deadline that
        the just-ended quarter may not be public yet -- but it does mean that at,
        say, 2020-05-15 the training window stops at 2019-12-31 even though Q1
        was filed around 2020-05-10.

    Example:
        >>> tr, fu, tk = take_training_data(
        ...     generator()[0], '2020-06-30', ['A', 'AAPL'])
        >>> tr.index.get_level_values('date').min().date()   # nq quarters
        datetime.date(2015, 6, 30)
        >>> tr.index.get_level_values('date').max().date()
        datetime.date(2020, 3, 31)
        >>> fu.index.get_level_values('date').min().date()   # nqf quarters
        datetime.date(2020, 6, 30)
        >>> fu.index.get_level_values('date').max().date()
        datetime.date(2023, 3, 31)
    """
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if isinstance(date,str):
        date = pd.Timestamp(date)
    end_traindate: pd.Timestamp = accounting_cutoff(date)
    begin_traindate: pd.Timestamp = end_traindate - pd.offsets.QuarterEnd(nq-1)

    begin_futuredata: pd.Timestamp = date if date.is_quarter_end else (
                                        date - pd.offsets.QuarterEnd(1))
    end_futuredata: pd.Timestamp = begin_futuredata + pd.offsets.QuarterEnd(
                                                                        nqf-1)

    available: set[str] = set(df.index.get_level_values('ticker').unique())
    ticker = [t for t in ticker if t in available]

    idx = pd.IndexSlice
    df_train: pd.DataFrame = df.loc[idx[ticker,begin_traindate:end_traindate],:]

    # Being in the panel is not the same as being in the *window*: a 2024 listing
    # is a column of `df` but has no row between 2015 and 2020, and the slice
    # above drops it silently while `ticker` still names it. Every downstream
    # `.loc[ticker, ...]` then raises `KeyError: not in index` -- one such name in
    # a 40-ticker universe takes the whole formation date down. Narrowed on the
    # training window only: narrowing on the future window as well would pick the
    # universe partly on data the formation date cannot see.
    trained: set[str] = set(df_train.index.get_level_values('ticker'))
    ticker = [t for t in ticker if t in trained]

    df_train = df.loc[idx[ticker,begin_traindate:end_traindate],:]
    df_future: pd.DataFrame = df.loc[idx[ticker,begin_futuredata:end_futuredata],
                                     :]
    return df_train, df_future, ticker

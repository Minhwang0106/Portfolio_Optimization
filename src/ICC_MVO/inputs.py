"""Point-in-time inputs for the ICC-MVO arm, built from the panels already collected.

Bielstein & Hanauer (2019) take their inputs from IBES, Compustat and CRSP.
This repo has none of the three, so every input here is a substitute built from
the SEC accounting panel and the Yahoo price panels -- one function per input,
so that each substitution can be read off the code:

* **Explicit EPS** (`realised_eps`, `complete_eps`). IBES consensus in B&H; here
  the EPS actually realised over the three four-quarter blocks after the
  accounting cutoff. Lookahead by construction -- see `model.Icc_Mvo`.
* **Payout ratio** (`dividend_events`, `payout_ratio`). Compustat dividends in
  GLS; here the cash distributions recovered from Yahoo's close and adjusted
  close, whose ratio steps at every ex-date.
* **Industry target ROE** (`read_roe_pool`, `industry_target_roe`). All of
  Compustat in GLS; here every SEC XBRL filer's firm-years, grouped into FF48
  industries from SEC SIC codes (`Data.industry_pool`). Without that file the
  S&P 500 panel's own firm-years stand in (`annual_roe`, `panel_roe_pool`).
* **Momentum and the covariance window** (`momentum`, `return_window`). CRSP in
  B&H; here the monthly adjusted-close returns every strategy is scored on.

Accounting inputs read nothing after `accounting_cutoff(date)` and prices
nothing after the date itself -- except the realised EPS, which is the point of
the arm.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from constant import (
    icc_explicit_years, icc_payout_ta, industry_roe_years, industry_min_obs,
)

QUARTERS_PER_YEAR: int = 4
MONTHS_PER_YEAR: int = 12

# Smallest step in the adjusted-close factor read as a distribution rather than
# rounding. On this panel the factor's day-to-day jitter stays inside 1e-5 in
# both directions -- 2.3 million steps are within 1e-7, and not one of the
# negative ones reaches -1e-5 -- while a dividend only ever steps it up, and
# almost always by more than 1e-4 (a 0.01% yield).
DIVIDEND_NOISE: float = 1e-5


def realised_eps (eps_q: pd.DataFrame, cutoff: pd.Timestamp,
                  n_years: int = icc_explicit_years)-> pd.DataFrame:
    """EPS realised over the `n_years` four-quarter blocks after `cutoff`.

    Args:
        eps_q (pd.DataFrame): Quarterly EPS, quarter ends down the rows, one
            column per ticker.
        cutoff (pd.Timestamp): The formation date's accounting cutoff, a
            quarter end. Block 1 is the four quarters after it.
        n_years (int): Blocks to sum. Defaults to `constant.icc_explicit_years`.

    Returns:
        pd.DataFrame: One row per column of `eps_q`, columns 'fy1'..'fy<n>'.
            NaN for a block with any quarter missing -- a year with a hole in it
            is not a year's earnings -- which includes every block the panel
            does not reach yet.

    Example:
        >>> realised_eps(eps_q, pd.Timestamp('2020-03-31')).columns.tolist()
        ['fy1', 'fy2', 'fy3']
    """
    blocks: dict[str, pd.Series] = {}
    for year in range(n_years):
        quarters: pd.DatetimeIndex = pd.date_range(
            cutoff+pd.offsets.QuarterEnd(QUARTERS_PER_YEAR*year+1),
            periods=QUARTERS_PER_YEAR, freq='QE')
        blocks[f'fy{year+1}'] = eps_q.reindex(quarters).sum(
            min_count=QUARTERS_PER_YEAR)
    return pd.DataFrame(blocks)


def trailing_eps (eps_q: pd.DataFrame, cutoff: pd.Timestamp)-> pd.Series:
    """E0: EPS over the four quarters ending at `cutoff`, NaN if any is missing.

    Args:
        eps_q (pd.DataFrame): Quarterly EPS, as `realised_eps`.
        cutoff (pd.Timestamp): Accounting cutoff, a quarter end.

    Returns:
        pd.Series: One value per column of `eps_q`.
    """
    quarters: pd.DatetimeIndex = pd.date_range(end=cutoff,
                                               periods=QUARTERS_PER_YEAR,
                                               freq='QE')
    return eps_q.reindex(quarters).sum(min_count=QUARTERS_PER_YEAR)


def complete_eps (realised: pd.DataFrame, e0: pd.Series, growth: pd.Series
                  )-> tuple[pd.DataFrame, pd.Series]:
    """Fill the explicit years the panel does not reach, and count the ones it does.

    A realised year is used only while every year before it is realised too.
    From the first missing one on, each year is the previous one grown at
    `growth` -- from E0 if not even year 1 is there. That is GLS's own device
    for a missing forecast year, growing the last one at the long-term growth
    rate, with the sustainable growth rate `(1 - k) * ROE_industry` standing in
    for the analyst forecast there is none of.

    Most firms' accounting ends at 2026-03-31, so at eleven explicit years the
    last one runs past the panel from the 2015-09-30 formation date on, more of
    them at every date after, and by 2025-12-31 all are extrapolated -- the
    same wall the proposed model's ex post window stops at. A firm acquired or
    delisted before its last explicit year is extrapolated the same way.

    Args:
        realised (pd.DataFrame): From `realised_eps`, one column per year.
        e0 (pd.Series): Trailing EPS at the cutoff, from `trailing_eps`.
        growth (pd.Series): Annual growth rate per ticker.

    Returns:
        tuple[pd.DataFrame, pd.Series]: The completed EPS, shaped like
            `realised`, and `n_oracle_years` -- how many leading years are
            realised rather than extrapolated. A ticker with neither a realised
            year 1 nor an E0 stays NaN throughout.
    """
    # Length of the unbroken run of realised years starting at year 1.
    n_oracle: pd.Series = (realised.notna().astype(int).cumprod(axis=1)
                           .sum(axis=1).astype(int))
    out: pd.DataFrame = realised.copy()
    prev: pd.Series = e0.reindex(realised.index)
    rate: pd.Series = growth.reindex(realised.index)
    for year, col in enumerate(realised.columns):
        out[col] = realised[col].where(n_oracle > year, prev*(1.0+rate))
        prev = out[col]
    return out, n_oracle.rename('n_oracle_years')


def dividend_events (daily: pd.DataFrame, noise: float = DIVIDEND_NOISE
                     )-> pd.Series:
    """Cash distributions per share, recovered from the daily close and adjusted close.

    Yahoo's adjusted close is CRSP-style: at each ex-date `e`, every earlier
    price is multiplied by `1 - D_e / C_{e-1}`, with `C_{e-1}` the close the
    day before. The factor `f = adj_close / close` therefore steps up only at
    ex-dates, and each step gives the distribution back:

        D_e = C_{e-1} * (1 - f_{e-1} / f_e)

    `close` is split-adjusted but not dividend-adjusted, so `D` comes out per
    share on today's split basis -- the basis of `adj_shares` and of the price
    the ICC is solved against, with nothing further to adjust. Checked to the
    cent on KO (0.46 a quarter in 2023), AAPL (0.205 after its 2020 split), JNJ
    and MSFT, and on COST's $15 special dividend of December 2023. Specials and
    the spin-offs Yahoo books as cash (MO's Philip Morris distribution in 2008)
    count, as they do in Compustat's dividend field; a payout ratio they push
    past one is clipped by `payout_ratio`.

    Args:
        daily (pd.DataFrame): Daily price panel indexed (ticker, date) and
            sorted, with 'close' and 'adj_close'.
        noise (float): Smallest step read as a distribution. Defaults to
            `DIVIDEND_NOISE`.

    Returns:
        pd.Series: Named 'dps', indexed (ticker, ex-date), one row per
            distribution.
    """
    factor: pd.Series = daily['adj_close']/daily['close']
    step: pd.Series = 1.0-factor.groupby(level='ticker').shift(1)/factor
    prev_close: pd.Series = daily['close'].groupby(level='ticker').shift(1)
    return (prev_close*step)[step > noise].rename('dps')


def monthly_dividends (events: pd.Series)-> pd.DataFrame:
    """Distributions summed by calendar month: month ends down the rows, wide on tickers.

    Args:
        events (pd.Series): From `dividend_events`.

    Returns:
        pd.DataFrame: Zero in a month a ticker paid nothing.
    """
    month: pd.DatetimeIndex = (events.index.get_level_values('date')
                               +pd.offsets.MonthEnd(0))
    ticker: pd.Index = events.index.get_level_values('ticker')
    wide: pd.DataFrame = (events.groupby([month, ticker]).sum()
                          .unstack(fill_value=0.0).sort_index())
    wide.index.name = 'date'
    return wide


def trailing_dps (dividends: pd.DataFrame, end: pd.Timestamp,
                  tickers: list[str]|pd.Index)-> pd.Series:
    """Dividends per share over the twelve months ending at `end`.

    Args:
        dividends (pd.DataFrame): From `monthly_dividends`.
        end (pd.Timestamp): Last month end included.
        tickers (list[str] | pd.Index): Names to report.

    Returns:
        pd.Series: One value per ticker; zero for a name that paid nothing in
            the window, including one that has never paid at all.
    """
    months: pd.DatetimeIndex = pd.date_range(end=end+pd.offsets.MonthEnd(0),
                                             periods=MONTHS_PER_YEAR, freq='ME')
    return dividends.reindex(index=months, columns=tickers).sum()


def payout_ratio (dps: pd.Series, e0: pd.Series, assets_ps: pd.Series,
                  asset_return: float = icc_payout_ta)-> pd.Series:
    """GLS's dividend payout ratio k, clipped to [0, 1].

    Dividends over earnings where earnings are positive. Where they are not,
    dividends over `asset_return` times total assets: GLS's assumption that
    such a firm earns 6% of its assets in a normal year.

    Args:
        dps (pd.Series): Dividends per share over the year.
        e0 (pd.Series): Earnings per share over the same year.
        assets_ps (pd.Series): Total assets per share.
        asset_return (float): Earnings-to-assets assumed for a firm without
            positive earnings. Defaults to `constant.icc_payout_ta`.

    Returns:
        pd.Series: k per ticker. Zero for a firm that paid nothing, whatever
            its earnings; NaN only where something was paid and neither
            denominator is positive.
    """
    base: pd.Series = e0.where(e0 > 0, asset_return*assets_ps)
    ratio: pd.Series = dps/base.where(base > 0)
    return ratio.mask(dps == 0, 0.0).clip(0.0, 1.0)


def annual_roe (ni_q: pd.DataFrame, be_q: pd.DataFrame)-> pd.DataFrame:
    """Calendar-year ROE per firm: the year's net income over the prior December's book equity.

    GLS's ROE is a year's earnings over the book value it started from. The
    accounting panel is aligned to calendar quarters on the way in
    (`Data.utils.to_calendar_quarter`), so a calendar year is four of its rows
    whatever the firm's fiscal year.

    Args:
        ni_q (pd.DataFrame): Quarterly net income, quarter ends down the rows,
            one column per ticker.
        be_q (pd.DataFrame): Quarterly book equity, the same shape.

    Returns:
        pd.DataFrame: Calendar years down the rows (ints), one column per
            ticker. NaN for a year missing a quarter or starting from
            non-positive book equity.
    """
    income: pd.DataFrame = ni_q.groupby(ni_q.index.year).sum(
        min_count=QUARTERS_PER_YEAR)
    income.index.name = 'year'
    december: pd.DataFrame = be_q[be_q.index.month == 12]
    december.index = december.index.year
    opening: pd.DataFrame = december.reindex(income.index-1)
    opening.index = income.index
    return income/opening.where(opening > 0)


def industry_target_roe (pool: pd.DataFrame, last_year: int,
                         years: tuple[int, int] = industry_roe_years,
                         min_obs: int = industry_min_obs
                         )-> tuple[pd.Series, float]:
    """GLS's target ROE: the median over profitable firm-years, per industry and overall.

    Takes every firm-year of the pool in the `years[1]` calendar years ending
    at `last_year` and keeps the profitable ones -- GLS leave loss firm-years
    out, as saying little about where an industry's ROE settles.

    Args:
        pool (pd.DataFrame): One row per firm-year with 'year', 'roe' and
            'ff48' (float, NaN where unclassified), from `read_roe_pool` or
            `panel_roe_pool`. An unclassified row still counts towards the
            all-firm median.
        last_year (int): The last calendar year all of whose quarters are
            already public at the formation date.
        years (tuple[int, int]): Fewest and most years in the window. Defaults
            to `constant.industry_roe_years`.
        min_obs (int): Profitable firm-years an industry needs to have a median
            of its own. Defaults to `constant.industry_min_obs`.

    Returns:
        tuple[pd.Series, float]: The median per FF48 code (float index), for
            the industries with at least `min_obs` firm-years, and the all-firm
            median that every other firm falls back to.

    Raises:
        ValueError: If the window holds fewer than `years[0]` years with data.
    """
    shortest, longest = years
    window: pd.DataFrame = pool[(pool['year'] > last_year-longest)
                                & (pool['year'] <= last_year)
                                & pool['roe'].notna()]
    n_year: int = int(window['year'].nunique())
    if n_year < shortest:
        raise ValueError(f'{n_year} year(s) of ROE up to {last_year}; the '
                         f'industry median needs {shortest}')
    profitable: pd.DataFrame = window[window['roe'] > 0]
    stats: pd.DataFrame = (profitable.dropna(subset=['ff48'])
                           .groupby('ff48')['roe'].agg(['median', 'size']))
    stats.index = stats.index.astype(float)
    return (stats.loc[stats['size'] >= min_obs, 'median'].rename('roe_target'),
            float(profitable['roe'].median()))


def panel_roe_pool (roe: pd.DataFrame, industry: pd.Series)-> pd.DataFrame:
    """The S&P 500 panel's own firm-years as a pool: the fallback when there is no SEC pool.

    Args:
        roe (pd.DataFrame): From `annual_roe`: years down the rows, one column
            per ticker.
        industry (pd.Series): Ticker -> FF48 code, from `read_industry`.

    Returns:
        pd.DataFrame: One row per firm-year with an ROE: 'year', 'ticker',
            'roe', 'ff48' -- the shape `industry_target_roe` takes.
    """
    long: pd.Series = roe.stack().dropna()
    ticker: pd.Index = long.index.get_level_values(-1)
    return pd.DataFrame({
        'year': long.index.get_level_values(0).astype(int),
        'ticker': ticker,
        'roe': long.to_numpy(),
        'ff48': industry.reindex(ticker).astype(float).to_numpy()})


def read_roe_pool (path: Path)-> pd.DataFrame|None:
    """Every SEC filer's firm-years, from the table `Data.industry_pool` writes.

    Args:
        path (Path): `constant.industry_roe_pool_path`.

    Returns:
        pd.DataFrame | None: 'year', 'cik', 'roe', 'ff48' (float), one row per
            firm-year. None, with a warning, if the table has not been built --
            the caller then falls back to the S&P 500 panel's own firm-years,
            which is not GLS's pool and not a specification to report.
    """
    path = Path(path)
    if not path.is_file():
        warnings.warn(f'{path.name} not found; run `python -m '
                      'src.Data.industry_pool`. Until it exists the industry '
                      "median is over the S&P 500 panel's own firm-years, not "
                      'every filer.', RuntimeWarning)
        return None
    pool: pd.DataFrame = pd.read_csv(path, dtype={'cik': str},
                                     usecols=['cik', 'year', 'roe', 'ff48'])
    pool['ff48'] = pool['ff48'].astype(float)
    return pool[['year', 'cik', 'roe', 'ff48']]


def read_industry (path: Path)-> pd.Series:
    """Ticker -> FF48 code from the table `Data.industry` writes.

    Args:
        path (Path): `constant.industry_path`.

    Returns:
        pd.Series: Float codes indexed by ticker. Empty, with a warning, if the
            table has not been built -- every firm then fades to the all-firm
            median ROE, which is a different specification from GLS's and not
            one to report.
    """
    path = Path(path)
    if not path.is_file():
        warnings.warn(f'{path.name} not found; run `python -m src.Data.industry`. '
                      'Until it exists every firm fades to the all-firm median '
                      'ROE rather than its industry\'s.', RuntimeWarning)
        return pd.Series(dtype=float, name='ff48')
    table: pd.DataFrame = pd.read_csv(path)
    return table.set_index('ticker')['ff48'].astype(float)


def momentum (returns: pd.DataFrame, date: pd.Timestamp,
              tickers: list[str]|pd.Index)-> pd.Series:
    """12-1 momentum: the return from month end t-12 to month end t-1.

    The eleven monthly returns before the formation month, skipping the
    formation month itself -- the standard definition, which leaves out the
    most recent month's short-term reversal.

    Args:
        returns (pd.DataFrame): Monthly returns, month ends down the rows.
        date (pd.Timestamp): Formation date, a month end.
        tickers (list[str] | pd.Index): Names to report.

    Returns:
        pd.Series: NaN unless all eleven months are there.
    """
    months: pd.DatetimeIndex = pd.date_range(end=date-pd.offsets.MonthEnd(1),
                                             periods=MONTHS_PER_YEAR-1,
                                             freq='ME')
    window: pd.DataFrame = returns.reindex(index=months, columns=tickers)
    return (1.0+window).prod(min_count=MONTHS_PER_YEAR-1)-1.0


def return_window (returns: pd.DataFrame, date: pd.Timestamp,
                   tickers: list[str]|pd.Index, n_month: int)-> pd.DataFrame:
    """The `n_month` monthly returns ending at `date`, NaN where a name was not priced.

    Args:
        returns (pd.DataFrame): Monthly returns, month ends down the rows.
        date (pd.Timestamp): Formation date, a month end; its own month is
            already realised and is included.
        tickers (list[str] | pd.Index): Names to report.
        n_month (int): Window length.

    Returns:
        pd.DataFrame: Months down the rows, one column per ticker.
    """
    months: pd.DatetimeIndex = pd.date_range(end=date, periods=n_month,
                                             freq='ME')
    return returns.reindex(index=months, columns=tickers)

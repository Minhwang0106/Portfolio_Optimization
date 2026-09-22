"""Every SEC filer's calendar-year ROE with its FF48 industry: the pool for GLS's industry median.

Gebhardt, Lee & Swaminathan (2001) fade each firm's ROE towards the median over
profitable firm-years of *every* firm in its industry -- all of Compustat. This
repo's own accounting panel is the S&P 500, about 570 firms, which leaves some of
the 48 industries with one or two members, and is chosen with hindsight: a firm
is in it because it made the index at some point up to 2025. SEC's XBRL frames
API gives the same two numbers for every filer at once, one request per concept
and period:

    https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/{period}.json

* **Net income**, `CY{year}`: the fiscal year of each filer that best fits the
  calendar year (365 +/- 30 days). `NetIncomeLoss`, then `ProfitLoss` where a
  filer has none -- the order of `constant.ratio['net_income']`.
* **Common book equity at the prior calendar year-end**, `CY{year-1}Q4I`:
  `StockholdersEquity` (parent only), or failing that the total including
  non-controlling interests less `MinorityInterest`; then less
  `PreferredStockValue`. The same definition as `panel.book_equity`.

ROE is the year's net income over that opening book equity, as
`ICC_MVO.inputs.annual_roe` computes it on the panel. A firm-year starting from
less than `constant.industry_pool_min_equity` is left out: shells and blank-cheque
companies report an ROE on a few thousand dollars of equity that says nothing
about an industry. Each remaining filer's SIC code comes from its submissions
record (`Data.industry.fetch_sic_by_cik`, cached in `constant.sic_by_cik_path` so
an interrupted run resumes) and is mapped to FF48 with French's `Siccodes48`.
SIC is looked up only for filers with at least one profitable year, since loss
years never enter the median.

Limits:
* A frame holds the value *last filed*, so a restatement replaces the original.
* A non-December filer's year is aligned to the nearest calendar year, and its
  opening equity is its balance at the calendar year-end (from a 10-Q) rather
  than at its own fiscal year-end.
* SIC codes are SEC's current ones, not the ones in force at the time.
* XBRL was phased in over 2009-2011, largest filers first, so the pool's first
  years lean towards large firms.

    python -m src.Data.industry_pool --user-agent "Your Name you@example.com"

About a hundred frame requests (a minute), then one submissions request per
filer not yet cached (thousands; at SEC's pace, most of an hour on a first run).
Writes `constant.industry_roe_pool_path`: cik, entity, year, net_income,
book_equity, roe, sic, ff48.
"""
import argparse
import time
from pathlib import Path
import pandas as pd
import requests
from constant import (
    industry_roe_pool_path, sic_by_cik_path, industry_pool_min_equity,
    industry_min_obs,
)
from .accounting_info import sec_header
from .industry import (
    SEC_PAUSE, sec_get, fetch_sic_by_cik, load_siccodes, sic_to_ff48,
    unmatched_sic_codes,
)

FRAMES_URL: str = ('https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/'
                   '{period}.json')
INCOME_TAGS: tuple[str, ...] = ('NetIncomeLoss', 'ProfitLoss')
EQUITY_TAG: str = 'StockholdersEquity'
EQUITY_WITH_NCI_TAG: str = ('StockholdersEquityIncludingPortionAttributableTo'
                            'NoncontrollingInterest')
NCI_TAG: str = 'MinorityInterest'
PREFERRED_TAG: str = 'PreferredStockValue'
# The first calendar year with XBRL income to divide by 2008 year-end equity.
FIRST_YEAR: int = 2009


def frame_values (payload: dict)-> pd.DataFrame:
    """One frame's facts as a table: one row per filer.

    Args:
        payload (dict): The frames API's JSON, whose 'data' holds one fact per
            filer with 'cik', 'entityName' and 'val'.

    Returns:
        pd.DataFrame: Columns 'entity' and 'val' (float), indexed by the
            ten-digit CIK as a string, the form `Data.industry` uses. Empty for
            a payload with no facts.

    Example:
        >>> frame_values({'data': [{'cik': 320193, 'entityName': 'Apple Inc.',
        ...                         'val': 93736000000}]})
                        entity           val
        cik
        0000320193  Apple Inc.  9.373600e+10
    """
    rows: list[dict] = payload.get('data') or []
    frame: pd.DataFrame = pd.DataFrame({
        'cik': [str(row['cik']).zfill(10) for row in rows],
        'entity': [row.get('entityName') for row in rows],
        'val': pd.Series([row['val'] for row in rows], dtype=float)})
    # One fact per filer is what the API promises; keep the last should it not.
    return frame.drop_duplicates('cik', keep='last').set_index('cik')


def fetch_frame (session: requests.Session, tag: str, period: str,
                 header: dict[str, str], pause: float = SEC_PAUSE
                 )-> pd.DataFrame:
    """Download one frame, e.g. `NetIncomeLoss` for `CY2019`.

    Args:
        session (requests.Session): Reused across requests.
        tag (str): us-gaap concept.
        period (str): 'CY2019' for a year, 'CY2019Q4I' for a year-end balance.
        header (dict[str, str]): From `accounting_info.sec_header`.
        pause (float): Seconds to wait after the request. Defaults to
            `Data.industry.SEC_PAUSE`.

    Returns:
        pd.DataFrame: As `frame_values`; empty where SEC has no such frame
            (404), as for a year not yet reported.
    """
    response = sec_get(session, FRAMES_URL.format(tag=tag, period=period),
                       header)
    time.sleep(pause)
    if response.status_code == 404:
        return frame_values({})
    response.raise_for_status()
    return frame_values(response.json())


def common_equity (parent: pd.Series, with_nci: pd.Series, nci: pd.Series,
                   preferred: pd.Series)-> pd.Series:
    """Common book equity per filer, as `panel.book_equity` defines it.

    Parent-only stockholders' equity where the filer reports it; otherwise the
    total including non-controlling interests, less those interests (zero if
    not reported). Preferred stock, where reported, comes off either.

    Args:
        parent (pd.Series): `StockholdersEquity` per CIK.
        with_nci (pd.Series): The total including non-controlling interests.
        nci (pd.Series): `MinorityInterest`.
        preferred (pd.Series): `PreferredStockValue`.

    Returns:
        pd.Series: Per CIK with either equity figure.
    """
    fallback: pd.Series = with_nci-nci.reindex(with_nci.index).fillna(0.0)
    total: pd.Series = parent.combine_first(fallback)
    return total-preferred.reindex(total.index).fillna(0.0)


def firm_year_roe (net_income: pd.Series, book_equity: pd.Series,
                   min_equity: float = industry_pool_min_equity
                   )-> pd.DataFrame:
    """One calendar year's ROE per filer: net income over opening common equity.

    Args:
        net_income (pd.Series): The year's net income per CIK.
        book_equity (pd.Series): Common book equity at the prior year-end.
        min_equity (float): Smallest opening equity kept, in dollars. Defaults
            to `constant.industry_pool_min_equity`.

    Returns:
        pd.DataFrame: 'net_income', 'book_equity', 'roe', for the filers with
            both figures and opening equity of at least `min_equity`.
    """
    both: pd.DataFrame = pd.DataFrame({'net_income': net_income,
                                       'book_equity': book_equity}).dropna()
    both = both[both['book_equity'] >= min_equity].copy()
    both['roe'] = both['net_income']/both['book_equity']
    return both


def build_pool (header: dict[str, str], years: range|list[int],
                min_equity: float = industry_pool_min_equity,
                pause: float = SEC_PAUSE)-> pd.DataFrame:
    """Every filer's ROE for each calendar year in `years`, from SEC's frames.

    Args:
        header (dict[str, str]): From `accounting_info.sec_header`.
        years (range | list[int]): Calendar years of net income; each one's
            opening equity is read at the end of the year before.
        min_equity (float): See `firm_year_roe`.
        pause (float): Seconds between requests.

    Returns:
        pd.DataFrame: One row per filer-year: 'cik', 'entity', 'year',
            'net_income', 'book_equity', 'roe'. A year with no income frame yet
            is skipped with a message.

    Raises:
        ValueError: If no year produced a single firm-year.
    """
    session = requests.Session()

    def get (tag: str, period: str)-> pd.DataFrame:
        return fetch_frame(session, tag, period, header, pause)

    years_out: list[pd.DataFrame] = []
    for year in years:
        income_frames: list[pd.DataFrame] = [get(tag, f'CY{year}')
                                             for tag in INCOME_TAGS]
        if all(frame.empty for frame in income_frames):
            print(f'CY{year}: no income frame yet; skipped')
            continue
        opening: str = f'CY{year-1}Q4I'
        equity: pd.Series = common_equity(
            get(EQUITY_TAG, opening)['val'],
            get(EQUITY_WITH_NCI_TAG, opening)['val'],
            get(NCI_TAG, opening)['val'],
            get(PREFERRED_TAG, opening)['val'])
        income: pd.Series = income_frames[0]['val']
        entity: pd.Series = income_frames[0]['entity']
        for frame in income_frames[1:]:
            income = income.combine_first(frame['val'])
            entity = entity.combine_first(frame['entity'])
        one: pd.DataFrame = firm_year_roe(income, equity, min_equity)
        if one.empty:
            print(f'CY{year}: no filer with both figures; skipped')
            continue
        one.insert(0, 'entity', entity.reindex(one.index))
        one.insert(1, 'year', year)
        years_out.append(one)
        print(f'CY{year}: {len(one)} firm-years')
    if not years_out:
        raise ValueError(f'no firm-year in any of {list(years)}')
    pool: pd.DataFrame = pd.concat(years_out)
    pool.index.name = 'cik'
    return pool.reset_index()


def run_industry_pool (user_agent: str|None = None,
                       first_year: int = FIRST_YEAR,
                       last_year: int|None = None,
                       path: Path = industry_roe_pool_path,
                       sic_cache: Path = sic_by_cik_path,
                       min_equity: float = industry_pool_min_equity
                       )-> pd.DataFrame:
    """Build and write the industry ROE pool.

    Args:
        user_agent (str | None): SEC contact string. Falls back to
            `$SEC_USER_AGENT`; see `accounting_info.sec_header`.
        first_year (int): First calendar year of net income. Defaults to
            `FIRST_YEAR`.
        last_year (int | None): Last one. Defaults to last calendar year.
        path (Path): Output CSV. Defaults to `constant.industry_roe_pool_path`.
        sic_cache (Path): SIC per CIK already looked up. Defaults to
            `constant.sic_by_cik_path`.
        min_equity (float): See `firm_year_roe`.

    Returns:
        pd.DataFrame: The pool as written: 'cik', 'entity', 'year',
            'net_income', 'book_equity', 'roe', 'sic', 'ff48'. 'sic' and 'ff48'
            are empty for filers with no profitable year, which the median never
            uses, and for those SEC has no SIC for.
    """
    header: dict[str, str] = sec_header(user_agent)
    if last_year is None:
        last_year = pd.Timestamp.today().year-1
    pool: pd.DataFrame = build_pool(header, range(first_year, last_year+1),
                                    min_equity)
    profitable: pd.Series = pool['roe'] > 0
    needed: list[str] = sorted(pool.loc[profitable, 'cik'].unique())
    print(f'SIC codes: {len(needed)} filers with a profitable year')
    sic: dict[str, int|None] = fetch_sic_by_cik(needed, header,
                                                cache=sic_cache)
    pool['sic'] = pd.to_numeric(pool['cik'].map(sic),
                                errors='coerce').astype('Int64')
    siccodes: pd.DataFrame = load_siccodes()
    gaps: list[int] = unmatched_sic_codes(pool['sic'].astype(float), siccodes)
    pool['ff48'] = sic_to_ff48(pool['sic'].astype(float),
                               siccodes).astype('Int64')
    path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(path, index=False)

    usable: pd.DataFrame = pool[profitable & pool['ff48'].notna()]
    recent: pd.DataFrame = usable[usable['year'] > last_year-10]
    sizes: pd.Series = recent.groupby('ff48').size()
    print(f'{len(pool)} firm-years from {pool["cik"].nunique()} filers, '
          f'CY{first_year}-CY{last_year}; '
          f'{pool.loc[profitable, "ff48"].notna().mean():.1%} of the '
          f'profitable ones classified into {usable["ff48"].nunique()} of the '
          '48 industries.')
    print('firm-years per year: ' + ', '.join(
        f'{year} {n}' for year, n in pool.groupby('year').size().items()))
    print(f'profitable firm-years per industry over the last 10 years: min '
          f'{sizes.min()}, median {int(sizes.median())}; '
          f'{int((sizes < industry_min_obs).sum())} industries under '
          f'{industry_min_obs}.')
    if gaps:
        print(f'{len(gaps)} SIC code(s) fell in no FF48 range and went to 48 '
              f'Other by fallback, not by their own range: {gaps}. See '
              '`constant.sic_override` if one of these should map elsewhere.')
    print(f'written to {path.name}')
    return pool


def main (argv: list[str]|None = None)-> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(
        description="Build every SEC filer's calendar-year ROE with its "
                    'Fama-French 48 industry, the pool for the ICC industry '
                    'median.')
    parser.add_argument(
        '--user-agent', default=None, metavar='STRING',
        help='contact string sent to SEC EDGAR as the User-Agent, e.g. '
             '"Your Name you@example.com". Overrides $SEC_USER_AGENT.')
    parser.add_argument('--first-year', type=int, default=FIRST_YEAR,
                        help=f'first calendar year (default {FIRST_YEAR})')
    parser.add_argument('--last-year', type=int, default=None,
                        help='last calendar year (default: last year)')
    args = parser.parse_args(argv)
    run_industry_pool(user_agent=args.user_agent, first_year=args.first_year,
                      last_year=args.last_year)


if __name__ == '__main__':
    main()

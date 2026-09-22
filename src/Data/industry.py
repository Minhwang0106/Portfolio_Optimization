"""Ticker -> SIC -> Fama-French 48 industry, for `ICC_MVO`'s industry ROE.

Gebhardt, Lee & Swaminathan (2001) fade each firm's ROE towards the median of
its Fama-French (1997) 48-industry peers, and Bielstein & Hanauer (2019) keep
that. The industries are defined on four-digit SIC codes, which no panel here
carries, so this module collects them once:

1. SEC's `company_tickers.json` maps tickers to CIKs. Downloaded once for the
   whole list -- `accounting_info.Sec_Data_Restructure.cik_matching` downloads
   it once per ticker -- with `constant.cik_override` applied first, as there.
2. `https://data.sec.gov/submissions/CIK##########.json` carries each
   registrant's `sic`. One request per CIK, paced under SEC's limit of ten a
   second, and cached in `constant.sic_by_cik_path`, which
   `Data.industry_pool` shares for its thousands of filers.
3. Kenneth French's `Siccodes48.zip` gives each industry's SIC ranges. Cached in
   `Data/raw file/` as downloaded and parsed from there, so the mapping can be
   re-derived without the network.

A SIC code inside none of the 48 industries' ranges goes to 48, 'Other', as is
usual; the run names which codes did. On this SIC list that is rare and not a
sign of a scheme mismatch: SEC's generic and special-purpose codes (6770 blank
checks, 6798 REITs, 8880 ADRs, among others) each sit inside a real FF48 range,
because French's 598 ranges were drawn from the same SIC manual SEC uses. Only
codes the manual treats as "no industry" at all -- 0, 9995 (non-operating) and
9999 (non-classifiable) -- fall outside every range, and `Other` is the right
answer for them regardless. `constant.sic_override` is the escape hatch if a
different code is ever found to need a different one; `parse_siccodes` also
checks French's ranges for internal overlaps, so a future revision of the file
that contradicts itself is caught rather than silently resolved by whichever
range happens to come first.

Two limits. SEC's `sic`
is the registrant's *current* code, where GLS use the one in force at the time;
an S&P 500 firm rarely changes industry, but one that did is grouped by where it
ended up. And a ticker SEC's list no longer carries gets no CIK; `ICC_MVO` gives
such a firm the all-firm median ROE.

    python -m src.Data.industry --user-agent "Your Name you@example.com"

Writes `constant.industry_path`: ticker, cik, sic, ff48, ff48_short, ff48_name.
"""
import argparse
import re
import time
import warnings
import zipfile
from pathlib import Path
import pandas as pd
import requests
from constant import (
    accounting_path, cik_override, industry_path, siccodes48_path,
    sic_by_cik_path, sic_override,
)
from .accounting_info import sec_header

TICKER_URL: str = 'https://www.sec.gov/files/company_tickers.json'
SUBMISSIONS_URL: str = 'https://data.sec.gov/submissions/CIK{cik}.json'
SICCODES48_URL: str = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/'
                       'ftp/Siccodes48.zip')
# SEC's fair-access limit is ten requests a second; 0.12s between them stays
# under it even when a request returns at once.
SEC_PAUSE: float = 0.12
N_INDUSTRY: int = 48
OTHER_INDUSTRY: int = 48

# A range line, `dddd-dddd description`, is tried first: an industry header
# line starts with a one- or two-digit number, which a range would also match.
_RANGE = re.compile(r'^\s*(\d{4})-(\d{4})\b')
_HEADER = re.compile(r'^\s*(\d{1,2})\s+(\S+)\s+(.*\S)\s*$')


def parse_siccodes (text: str, expect: int = N_INDUSTRY)-> pd.DataFrame:
    """Parse French's `Siccodes48.txt` into one row per SIC range.

    The file gives each industry as a header line -- number, short name, long
    name -- followed by its ranges, one per line.

    Args:
        text (str): The file's contents.
        expect (int): Industries the file must define. Defaults to 48.

    Returns:
        pd.DataFrame: Columns 'ff48', 'ff48_short', 'ff48_name', 'sic_lo',
            'sic_hi', in the file's order.

    Also checks every pair of ranges belonging to *different* industries for an
    overlap -- `sic_to_ff48` would resolve one silently, by file order, which is
    fine for this file (there are none) but would not be a safe thing to do by
    accident on a future revision.

    Raises:
        ValueError: If a range comes before any industry, or the file does not
            define exactly `expect` industries -- a format change, which should
            stop the run rather than map every firm to 'Other'.

    Warns:
        RuntimeWarning: If two industries' ranges overlap, naming the SIC codes
            both claim and which industry (by file order) `sic_to_ff48` would
            give them.

    Example:
        >>> parse_siccodes(' 1 Agric  Agriculture\\n'
        ...                '          0100-0199 Crops\\n', expect=1)
           ff48 ff48_short    ff48_name  sic_lo  sic_hi
        0     1      Agric  Agriculture     100     199
    """
    rows: list[dict] = []
    current: dict|None = None
    for line in text.splitlines():
        found = _RANGE.match(line)
        if found:
            if current is None:
                raise ValueError(f'SIC range before any industry: '
                                 f'{line.strip()!r}')
            rows.append({**current, 'sic_lo': int(found[1]),
                         'sic_hi': int(found[2])})
            continue
        found = _HEADER.match(line)
        if found:
            current = {'ff48': int(found[1]), 'ff48_short': found[2],
                       'ff48_name': found[3]}
    table: pd.DataFrame = pd.DataFrame(rows, columns=[
        'ff48', 'ff48_short', 'ff48_name', 'sic_lo', 'sic_hi'])
    if table['ff48'].nunique() != expect:
        raise ValueError(f'expected {expect} industries, parsed '
                         f'{table["ff48"].nunique()}; has the file format '
                         'changed?')
    for overlap in _overlapping_ranges(table):
        warnings.warn(overlap, RuntimeWarning)
    return table


def _overlapping_ranges (table: pd.DataFrame)-> list[str]:
    """Pairs of different industries' ranges that share a SIC code.

    Args:
        table (pd.DataFrame): From `parse_siccodes`.

    Returns:
        list[str]: One message per overlapping pair, naming both industries,
            their ranges, and which one the first-range-wins rule in
            `sic_to_ff48` would pick.
    """
    messages: list[str] = []
    rows: list = list(table.itertuples(index=False))
    for i, first in enumerate(rows):
        for second in rows[i+1:]:
            if (first.ff48 != second.ff48
                    and first.sic_lo <= second.sic_hi
                    and second.sic_lo <= first.sic_hi):
                messages.append(
                    f'SIC {first.sic_lo}-{first.sic_hi} ({first.ff48} '
                    f'{first.ff48_short}) overlaps {second.sic_lo}-'
                    f'{second.sic_hi} ({second.ff48} {second.ff48_short}); '
                    f'{first.ff48} would win for the shared code(s)')
    return messages


def load_siccodes (path: Path = siccodes48_path,
                   url: str = SICCODES48_URL)-> pd.DataFrame:
    """French's FF48 definitions: downloaded on first use, then read from the cache.

    Args:
        path (Path): Where the zip is cached. Defaults to
            `constant.siccodes48_path`.
        url (str): Where to fetch it from if the cache is empty.

    Returns:
        pd.DataFrame: As `parse_siccodes`.
    """
    if not path.is_file():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    with zipfile.ZipFile(path) as archive:
        member: str = next(n for n in archive.namelist()
                           if n.lower().endswith('.txt'))
        text: str = archive.read(member).decode('latin-1')
    return parse_siccodes(text)


def sic_to_ff48 (sic: pd.Series, table: pd.DataFrame,
                 other: int|None = OTHER_INDUSTRY,
                 override: dict[int, int] = sic_override)-> pd.Series:
    """The FF48 industry of each SIC code.

    Args:
        sic (pd.Series): SIC codes, NaN where there is none.
        table (pd.DataFrame): From `parse_siccodes`.
        other (int | None): Industry for a code inside no range and not in
            `override`. Defaults to 48, 'Other'; None leaves it NaN, which is
            how `unmatched_sic_codes` finds them.
        override (dict[int, int]): SIC -> FF48, checked before the ranges.
            Defaults to `constant.sic_override`; empty by default.

    Returns:
        pd.Series: Float codes on `sic`'s index; NaN where `sic` is NaN. The
            first range in file order wins, should two ever overlap --
            `parse_siccodes` warns if they do.
    """
    code: pd.Series = pd.to_numeric(sic, errors='coerce')
    out: pd.Series = pd.Series(float('nan'), index=sic.index)
    for row in table.itertuples(index=False):
        inside: pd.Series = ((code >= row.sic_lo) & (code <= row.sic_hi)
                             & out.isna())
        out[inside] = float(row.ff48)
    if other is not None:
        out[code.notna() & out.isna()] = float(other)
    for raw, ff48 in override.items():
        out[code == raw] = float(ff48)
    return out


def unmatched_sic_codes (sic: pd.Series, table: pd.DataFrame,
                         override: dict[int, int] = sic_override
                         )-> list[int]:
    """SIC codes that land in no FF48 range and have no override.

    These are what `sic_to_ff48` assigns to `Other` purely by fallback, as
    opposed to a code that has its own explicit FF48=48 range (such as 4950,
    sanitary services) -- the distinction a plain count of `Other` firms would
    blur. A code appearing here is a candidate for `constant.sic_override`, not
    evidence by itself that one is needed: SEC's genuinely industry-less codes
    (0, 9995, 9999) belong in `Other` anyway.

    Args:
        sic (pd.Series): SIC codes, NaN where there is none.
        table (pd.DataFrame): From `parse_siccodes`.
        override (dict[int, int]): Codes already resolved by hand. Defaults to
            `constant.sic_override`.

    Returns:
        list[int]: Distinct unmatched codes actually present in `sic`, sorted.
    """
    code: pd.Series = pd.to_numeric(sic, errors='coerce')
    matched: pd.Series = sic_to_ff48(code, table, other=None,
                                     override=override)
    gap: pd.Series = code[code.notna() & matched.isna()]
    return sorted(int(v) for v in gap.unique())


def ticker_ciks (tickers: list[str], header: dict[str, str])-> dict[str, str]:
    """CIK per ticker from one download of SEC's list, `constant.cik_override` first.

    Args:
        tickers (list[str]): Symbols as the panels spell them.
        header (dict[str, str]): From `accounting_info.sec_header`.

    Returns:
        dict[str, str]: Ten-digit CIK per ticker found; absent tickers are
            left out.
    """
    response = requests.get(TICKER_URL, headers=header, timeout=30)
    response.raise_for_status()
    listed: dict[str, str] = {str(row['ticker']).upper():
                              str(row['cik_str']).zfill(10)
                              for row in response.json().values()}
    out: dict[str, str] = {}
    for ticker in tickers:
        key: str = ticker.upper().replace('.', '-')
        cik: str|None = cik_override.get(key) or listed.get(key)
        if cik is not None:
            out[ticker] = cik
    return out


def sec_get (session: requests.Session, url: str, header: dict[str, str]
             )-> requests.Response:
    """GET from SEC, retrying a 429 or a server error twice with a growing pause.

    Args:
        session (requests.Session): Reused across requests.
        url (str): Where to fetch.
        header (dict[str, str]): From `accounting_info.sec_header`.

    Returns:
        requests.Response: The last response; the caller decides what a 404 or
            a status that survived the retries means.
    """
    for attempt in range(3):
        response = session.get(url, headers=header, timeout=30)
        if response.status_code != 429 and response.status_code < 500:
            break
        time.sleep(2**attempt)
    return response


def fetch_sic_by_cik (ciks: list[str], header: dict[str, str],
                      pause: float = SEC_PAUSE, cache: Path|None = None,
                      report_every: int = 500)-> dict[str, int|None]:
    """Each CIK's SIC code from its SEC submissions record, one request per CIK.

    With a `cache`, CIKs already in it are not requested again, and every new
    answer is appended to it in batches -- so the pool's thousands of lookups
    (`Data.industry_pool`) survive an interruption, and this repo's own tickers,
    looked up by `run_industry`, are not fetched a second time.

    Args:
        ciks (list[str]): Ten-digit CIKs; repeats are looked up once.
        header (dict[str, str]): From `accounting_info.sec_header`.
        pause (float): Seconds between requests. Defaults to `SEC_PAUSE`.
        cache (Path | None): CSV of 'cik', 'sic' already looked up. None
            neither reads nor writes one.
        report_every (int): Print progress every this many requests.

    Returns:
        dict[str, int | None]: SIC per distinct CIK; None where SEC has none,
            including a CIK with no submissions record (404).

    Raises:
        requests.HTTPError: On any other failure that survives the retries.
            What was fetched before it is already in the cache.
    """
    known: dict[str, int|None] = {}
    if cache is not None and cache.is_file():
        stored: pd.DataFrame = pd.read_csv(cache, dtype={'cik': str})
        known = {cik: int(sic) if pd.notna(sic) else None
                 for cik, sic in zip(stored['cik'], stored['sic'])}
    wanted: list[str] = list(dict.fromkeys(ciks))
    todo: list[str] = [cik for cik in wanted if cik not in known]
    session = requests.Session()
    batch: list[dict] = []

    def flush ()-> None:
        if cache is not None and batch:
            cache.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(batch, columns=['cik', 'sic']).astype(
                {'sic': 'Int64'}).to_csv(cache, mode='a', index=False,
                                         header=not cache.is_file())
        batch.clear()

    try:
        for done, cik in enumerate(todo, 1):
            response = sec_get(session, SUBMISSIONS_URL.format(cik=cik),
                               header)
            if response.status_code == 404:
                known[cik] = None
            else:
                response.raise_for_status()
                sic: str = str(response.json().get('sic') or '').strip()
                known[cik] = int(sic) if sic.isdigit() else None
            batch.append({'cik': cik, 'sic': known[cik]})
            if len(batch) >= 100:
                flush()
            if done % report_every == 0:
                print(f'SIC: {done} of {len(todo)} looked up')
            time.sleep(pause)
    finally:
        flush()
    return {cik: known[cik] for cik in wanted}


def fetch_sic (ciks: dict[str, str], header: dict[str, str],
               pause: float = SEC_PAUSE, cache: Path|None = None
               )-> dict[str, int|None]:
    """Each ticker's SIC code, one request per distinct CIK.

    Share classes such as GOOG and GOOGL share a CIK and so one request.

    Args:
        ciks (dict[str, str]): Ticker -> CIK, from `ticker_ciks`.
        header (dict[str, str]): From `accounting_info.sec_header`.
        pause (float): Seconds between requests. Defaults to `SEC_PAUSE`.
        cache (Path | None): See `fetch_sic_by_cik`.

    Returns:
        dict[str, int | None]: SIC per ticker; None where SEC has none.
    """
    by_cik: dict[str, int|None] = fetch_sic_by_cik(list(ciks.values()), header,
                                                   pause, cache)
    return {ticker: by_cik[cik] for ticker, cik in ciks.items()}


def run_industry (user_agent: str|None = None,
                  tickers: list[str]|None = None,
                  path: Path = industry_path)-> pd.DataFrame:
    """Build and write the ticker -> FF48 table.

    Args:
        user_agent (str | None): SEC contact string. Falls back to
            `$SEC_USER_AGENT`; see `accounting_info.sec_header`.
        tickers (list[str] | None): Symbols to classify. Defaults to every
            ticker in the accounting panel, which is also every firm the
            industry median pools over.
        path (Path): Output CSV. Defaults to `constant.industry_path`.

    Returns:
        pd.DataFrame: One row per ticker: 'ticker', 'cik', 'sic', 'ff48',
            'ff48_short', 'ff48_name', the last four empty where SEC had no
            CIK or no SIC.
    """
    header: dict[str, str] = sec_header(user_agent)
    if tickers is None:
        tickers = sorted(pd.read_csv(accounting_path, usecols=['ticker'])
                         ['ticker'].astype(str).unique())
    table: pd.DataFrame = load_siccodes()
    names: pd.DataFrame = (table.drop_duplicates('ff48')
                           .set_index('ff48')[['ff48_short', 'ff48_name']])
    ciks: dict[str, str] = ticker_ciks(tickers, header)
    sics: dict[str, int|None] = fetch_sic(ciks, header, cache=sic_by_cik_path)

    frame: pd.DataFrame = pd.DataFrame({'ticker': list(tickers)})
    frame['cik'] = frame['ticker'].map(ciks)
    frame['sic'] = pd.array([sics.get(t) for t in frame['ticker']],
                            dtype='Int64')
    sic: pd.Series = frame['sic'].astype(float)
    gaps: list[int] = unmatched_sic_codes(sic, table)
    frame['ff48'] = sic_to_ff48(sic, table).astype('Int64')
    frame['ff48_short'] = frame['ff48'].map(names['ff48_short'])
    frame['ff48_name'] = frame['ff48'].map(names['ff48_name'])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

    covered: pd.Series = frame['ff48'].notna()
    print(f'{len(frame)} tickers: {frame["cik"].notna().sum()} with a CIK, '
          f'{frame["sic"].notna().sum()} with a SIC code, {covered.sum()} '
          f'classified ({covered.mean():.1%}) into '
          f'{frame["ff48"].nunique()} of the 48 industries.')
    if gaps:
        print(f'{len(gaps)} SIC code(s) fell in no FF48 range and went to 48 '
              f'Other by fallback, not by their own range: {gaps}. See '
              '`constant.sic_override` if one of these should map elsewhere.')
    if not covered.all():
        print('unclassified (the all-firm median ROE in ICC_MVO): '
              + ', '.join(frame.loc[~covered, 'ticker']))
    print(f'written to {path.name}')
    return frame


def main (argv: list[str]|None = None)-> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(
        description='Map every ticker in the accounting panel to its '
                    'Fama-French 48 industry.')
    parser.add_argument(
        '--user-agent', default=None, metavar='STRING',
        help='contact string sent to SEC EDGAR as the User-Agent, e.g. '
             '"Your Name you@example.com". Overrides $SEC_USER_AGENT.')
    args = parser.parse_args(argv)
    run_industry(user_agent=args.user_agent)


if __name__ == '__main__':
    main()

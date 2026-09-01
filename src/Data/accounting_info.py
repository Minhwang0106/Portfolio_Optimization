import os
import sys
import pandas as pd
import requests
from constant import (
    share_tags, share_scale_tolerance, share_multiplier, cik_override,
)
from .utils import (
    dedup_facts, facts_to_dates, dates_to_df, to_calendar_quarter,
    fix_misscaled_rows, drop_placeholder_rows,
)

SEC_USER_AGENT_ENV: str = 'SEC_USER_AGENT'


def _prompt_for_user_agent (attempts: int = 3)-> str:
    """Ask for a contact string at an interactive terminal.

    Written to stderr, and read with `input()`, so a run whose stdout is
    redirected to a log still shows the question on the terminal. Only called
    when stdin is a TTY; see `sec_header`.

    Args:
        attempts (int): How many blank entries to tolerate before giving up.

    Returns:
        str: The string entered, or '' if the user gave none, closed stdin
            (EOF) or interrupted -- all of which leave `sec_header` to raise.
    """
    print('SEC EDGAR identifies and throttles callers by User-Agent, so this '
          'collection\nneeds a contact string of your own -- your name and an '
          'email address,\ne.g. "Your Name you@example.com".',
          file=sys.stderr)
    for _ in range(attempts):
        # The prompt goes to stderr too, rather than through `input`'s own
        # argument, so the whole exchange stays on one stream: `input` writes
        # its prompt to stdout, which a run redirected to a log would swallow.
        print('SEC contact string: ', end='', file=sys.stderr, flush=True)
        try:
            entered: str = input().strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return ''
        if entered:
            if '@' not in entered:
                print('  (note: SEC asks for an address they can reach you at; '
                      'continuing anyway)', file=sys.stderr)
            print(f'  Set {SEC_USER_AGENT_ENV} in your environment to skip this '
                  'prompt next time.', file=sys.stderr)
            return entered
        print('  A contact string is required.', file=sys.stderr)
    return ''


def sec_header (user_agent: str|None = None,
                prompt: bool|None = None)-> dict[str, str]:
    """Build the EDGAR request header from an argument, the environment or a prompt.

    Resolution order: the `user_agent` argument, then `$SEC_USER_AGENT`, then --
    at an interactive terminal only -- a prompt. There is deliberately no
    default value. SEC's fair-access policy throttles and blocks by User-Agent,
    so a contact string baked into the source would attribute every user's
    collection -- some 570 tickers apiece -- to whichever address happened to be
    committed, and get that address rate-limited for traffic it never generated.
    The header also exists so SEC can reach whoever is making the requests,
    which a shared value defeats for everyone but its owner.

    The prompt is gated on `sys.stdin.isatty()` rather than being unconditional.
    This collection is a job measured in hours, and the ways it is actually
    started -- redirected to a log, backgrounded, resumed from a script, run in
    CI -- have no one sitting at the terminal to answer. Prompting there would
    either block forever or die on `EOFError` at whatever hour it was reached,
    where raising immediately says what is wrong while someone is still watching.

    Args:
        user_agent (str|None): Contact string to send. Takes precedence over
            `$SEC_USER_AGENT`; None or blank falls through to it.
        prompt (bool|None): Override the interactivity check -- True always
            asks, False never does. Defaults to whether stdin is a terminal.

    Returns:
        dict[str, str]: The `User-Agent` header for every EDGAR request.

    Raises:
        RuntimeError: If no source supplies one, naming every way in.

    Note:
        SEC documents the format as a descriptive string, not a bare address:
        'Your Name your.email@example.com'. See
        https://www.sec.gov/os/webmaster-faq#developers

    Example:
        >>> sec_header('Jane Doe jane@example.edu')
        {'User-Agent': 'Jane Doe jane@example.edu'}
    """
    resolved: str = ((user_agent or '').strip()
                     or os.environ.get(SEC_USER_AGENT_ENV, '').strip())
    if not resolved:
        interactive: bool = sys.stdin.isatty() if prompt is None else prompt
        if interactive:
            resolved = _prompt_for_user_agent()
    if not resolved:
        raise RuntimeError(
            'SEC EDGAR requires a contact string identifying whoever is making '
            'the requests. Set it in the environment:\n'
            '  $env:SEC_USER_AGENT = "Your Name you@example.com"   # PowerShell\n'
            '  export SEC_USER_AGENT="Your Name you@example.com"   # macOS/Linux\n'
            'or pass it to whichever entry point you are running:\n'
            '  python -m src.Data.run --user-agent "Your Name you@example.com"\n'
            '  python main.py --sec-user-agent "Your Name you@example.com"\n'
            'At an interactive terminal you are prompted for it instead.\n'
            'See https://www.sec.gov/os/webmaster-faq#developers')
    return {'User-Agent': resolved}


class Sec_Data_Restructure:
    class_description: dict = {}
    link: str = "https://www.sec.gov/files/company_tickers.json"
    root_companyfacts: str = "https://data.sec.gov/api/xbrl/companyfacts/CIK"

    def __init__(self,ticker:str, header: dict[str, str]|None = None) -> None:
        # Resolved per instance rather than bound once as a class-level default.
        # `run` builds one header and passes it in, so a whole collection reads
        # the environment once and fails before its first request, not at
        # ticker 570.
        self.ticker = ticker.upper().replace('.','-')
        self.header = header if header is not None else sec_header()
        self.share_info, self.fs_info = self.get_facts(ticker)

    def cik_matching(self, ticker: str) -> str:
        """Look up a ticker's 10-digit zero-padded SEC CIK number.

        `constant.cik_override` is consulted first, for the re-incorporations
        where SEC's own file points the symbol at a holding company that has
        filed no financials; see that constant for why the list is deliberately
        tiny. The override short-circuits the request as well as the match, so
        an overridden ticker costs no round trip.

        Args:
            ticker (str): Ticker symbol, case-insensitive.

        Returns:
            str: The CIK, zero-padded to 10 digits (e.g. '0000320193').

        Raises:
            ValueError: If the ticker is not found in SEC's company list.

        Example:
            >>> obj.cik_matching('AAPL')
            '0000320193'
        """
        ticker = ticker.upper().replace('.','-')
        if ticker in cik_override:
            return cik_override[ticker]
        response = requests.get(self.link, headers=self.header, timeout=30)
        response.raise_for_status()
        ticker_json: dict = response.json()
        for s in ticker_json.values():
            if s['ticker']==ticker:
                cik: str = str(s['cik_str']).zfill(10)
                return cik
        raise ValueError(f'{ticker} is unavailable in SEC database')

    def get_facts(self, ticker: str):
        """Fetch a ticker's raw SEC companyfacts and split into dei/us-gaap facts.

        Args:
            ticker (str): Ticker symbol, case-insensitive.

        Returns:
            tuple[dict, dict]: (shares_fact, fs_fact) — the raw 'dei' and
                'us-gaap' fact dictionaries from the SEC companyfacts API.

        Raises:
            ValueError: If the companyfacts response lacks dei/us-gaap taxonomy.
            requests.exceptions.RequestException: On HTTP failures.

        Example:
            >>> shares_fact, fs_fact = obj.get_facts('AAPL')
            >>> 'EntityCommonStockSharesOutstanding' in shares_fact
            True
        """
        cik: str = self.cik_matching(ticker)
        url: str = self.root_companyfacts + cik + '.json'
        response = requests.get(url, headers=self.header, timeout=30)
        response.raise_for_status()
        company_facts: dict = response.json()
        facts: dict = company_facts.get('facts', {})
        if 'dei' not in facts or 'us-gaap' not in facts:
            raise ValueError(
                f'{self.ticker}: SEC company facts missing dei/us-gaap taxonomy (cik={cik})'
            )
        shares_fact: dict = facts['dei']
        fs_fact: dict = facts['us-gaap']
        return shares_fact, fs_fact

    def share_compose (self, share_tags=share_tags)-> pd.DataFrame:
        """Build a shares-outstanding time series for this instance's ticker.

        Walks `share_tags` in order of preference and takes each tag's value
        only on the dates no higher-priority tag already covers, so the dei
        cover-page count wins wherever it is usable and the rest only fill its
        gaps. Two SEC realities force this:

        - Multi-class and reorganized filers tag the cover-page count *per
          share class*, and the non-dimensional fact that companyfacts exposes
          then comes back as a literal 0 (e.g. every TAP entry, and SPG from
          2010 on). Non-positive values are therefore discarded, not trusted.
        - The dei tag can be absent or near-empty (DDOG has none; SPG has four
          entries), which leaves the series too sparse to fill from.

        Values more than `constant.share_scale_tolerance` times below the
        ticker's pooled median are dropped as well: some pre-2010 filings tag
        the count in millions but still declare the unit as shares, which would
        otherwise put the market cap out by six orders of magnitude.

        Args:
            share_tags (tuple[tuple[str, str], ...]): Candidate (taxonomy, tag)
                pairs, highest priority first. 'dei' reads the dei facts, any
                other taxonomy reads the us-gaap facts.

        Returns:
            pd.DataFrame: MultiIndex ('ticker', 'date'), single column
                'shares' with the number of shares outstanding.

        Raises:
            ValueError: If no candidate tag yields a positive, plausibly-scaled
                value.

        Note:
            The last-resort tag is a period average, so its facts are durations.
            They are deduplicated on their end-date rather than put through
            `utils.quarterly_facts`, because differencing a weighted average
            does not give the next period's average.

        Example:
            >>> obj.share_compose().head(1)
                              shares
            ticker date
            AAPL   2020-01-31  4375479000
        """
        by_tag: list[dict] = []
        for taxonomy, tag in share_tags:
            facts: dict = self.share_info if taxonomy == 'dei' else self.fs_info
            if tag not in facts:
                continue
            unit_key: str = next(iter(facts[tag]['units']))
            usable: list = [entry for entry in facts[tag]['units'][unit_key]
                            if entry['val'] > 0]
            if not usable:
                continue
            by_tag.append(dedup_facts(usable))
        # Early XBRL filings sometimes tag the count in millions while still
        # declaring the unit as shares, so a handful of values come back 1e6 too
        # small (A reports 394 for 2007-10-31, and 348 as late as 2009). The
        # pooled median is unaffected by a minority of those, which makes it a
        # usable yardstick for throwing them out before the tags are merged --
        # dropping them at this point still lets a lower-priority tag cover the
        # dates they leave behind.
        if not by_tag:
            raise ValueError(f'{self.ticker}: no positive shares-outstanding '
                             f'value found in any of {share_tags}')
        # Largest per-tag median, not the median of everything pooled together.
        # A filer whose shell-period tags are all a placeholder 1000 (PSKY) drags
        # a pooled median down far enough to admit its own placeholders; taking
        # the tags one at a time lets the best-scaled tag set the yardstick.
        reference: float = max(pd.Series(list(tag_vals.values())).median()
                               for tag_vals in by_tag)
        # Banded both ways: the same mis-tagging that reports a count in millions
        # also shows up inverted, as BRK's five weighted-average facts of ~1.6e12
        # against a real 1.6e6.
        floor: float = reference/share_scale_tolerance
        ceiling: float = reference*share_scale_tolerance
        date_to_val: dict = {}
        for tag_vals in by_tag:
            date_to_val.update({date: val for date, val in tag_vals.items()
                                if floor <= val <= ceiling
                                and date not in date_to_val})
        if not date_to_val:
            raise ValueError(f'{self.ticker}: no plausibly-scaled '
                             f'shares-outstanding value found in any of '
                             f'{share_tags}')
        df: pd.DataFrame = pd.DataFrame(
            list(date_to_val.values()), columns=['shares'],
            index=pd.Index(pd.to_datetime(list(date_to_val.keys())),
                           name='date')).sort_index()
        df['shares'] = df['shares']*share_multiplier.get(self.ticker, 1.0)
        mul_index = [(self.ticker, i) for i in list(df.index)]
        mul_index = pd.MultiIndex.from_tuples(mul_index)
        df.index = mul_index
        df.index.names = ['ticker','date']
        return df

    def line_item_restructure (self, l_item: dict[str, list[str | tuple[str, ...]]],
                               optional_fields: frozenset[str] = frozenset()):
        """Assemble financial-statement line items into one DataFrame.

        For each output field name in `l_item`, tries its candidate SEC tags
        in order; a candidate may be a single tag or a group (list/tuple) of
        tags combined over their common dates. Within a group a tag is added,
        or subtracted if its name carries a leading '-' -- which is how
        `constant.ratio` nets non-controlling interests off the equity tag that
        includes them. A subtracted tag the filer never reports contributes
        zero rather than voiding the candidate. Each candidate only fills
        the dates no higher-priority one already covers, so an annual-only
        fallback cannot overwrite a preferred tag's quarterly figures. Flow
        items are reduced to single-quarter values (see
        `utils.quarterly_facts`) and dates are snapped to calendar
        quarter-ends via `utils.to_calendar_quarter`.

        Args:
            l_item (dict[str, list[str | tuple[str, ...]]]): Mapping of
                output field name to a list of candidate us-gaap tags. Each
                candidate is either a tag name or a tuple/list of tag names to
                combine, where a leading '-' subtracts rather than adds
                (e.g. {'revenue': ['Revenues', ('SalesA','SalesB')],
                'equity': [('EquityInclNCI', '-MinorityInterest')]}).
            optional_fields (frozenset[str]): Field names that may be missing
                entirely from the filings; such fields are filled with 0.0
                instead of raising.

        Returns:
            pd.DataFrame: MultiIndex ('ticker', 'date'), one column per key
                in `l_item`.

        Raises:
            ValueError: If a required (non-optional) field has no matching
                tag data available.

        Example:
            >>> obj.line_item_restructure({'revenue': ['Revenues']}).head(1)
                                    revenue
            ticker date
            AAPL   2020-03-31  58313000000
        """
        description: dict = dict()
        item_dfs: list[pd.DataFrame] = []
        for name, candidate_tags in l_item.items():
            date_to_val: dict = {}
            for tag in candidate_tags:
                unit_key: str
                if isinstance(tag, (list, tuple)):
                    added: list[str] = [t for t in tag if not t.startswith('-')]
                    # A subtracted tag that the filer never reports is nothing to
                    # net off, not a reason to abandon the candidate: a company
                    # with no NCI simply has no MinorityInterest fact.
                    taken: list[str] = [t[1:] for t in tag if t.startswith('-')
                                        and t[1:] in self.fs_info]
                    if not all(t in self.fs_info for t in added):
                        continue
                    sub_series, minus_series = [], []
                    for t, bucket in ([(t, sub_series) for t in added]
                                      + [(t, minus_series) for t in taken]):
                        unit_key = next(iter(self.fs_info[t]['units']))
                        bucket.append(facts_to_dates(
                            self.fs_info[t]['units'][unit_key]))
                        description.setdefault(name, self.fs_info[t]['description'])
                    common_dates = set.intersection(*(set(s) for s in sub_series))
                    candidate: dict = {
                        d: sum(s[d] for s in sub_series)
                           - sum(s.get(d, 0) for s in minus_series)
                        for d in common_dates}
                elif tag in self.fs_info:
                    unit_key = next(iter(self.fs_info[tag]['units']))
                    candidate = facts_to_dates(
                        self.fs_info[tag]['units'][unit_key])
                    description.setdefault(name, self.fs_info[tag]['description'])
                else:
                    continue
                # Candidates are ranked, so a later tag only fills dates no
                # earlier one reached. Pooling them instead would let a
                # annual-only fallback such as 'Revenues' overwrite the
                # quarterly figures the preferred tag already supplied.
                date_to_val.update({d: v for d, v in candidate.items()
                                    if d not in date_to_val})
            if not date_to_val:
                if name in optional_fields:
                    continue
                raise ValueError(
                    f"{self.ticker}: none of {candidate_tags} found for '{name}'"
                )
            temp_df: pd.DataFrame = dates_to_df(date_to_val,name)
            temp_df.index = temp_df.index.map(to_calendar_quarter)
            temp_df.index.name = 'date'
            temp_df = temp_df[~temp_df.index.duplicated(keep='first')]
            item_dfs.append(temp_df)
        df: pd.DataFrame = pd.concat(item_dfs, axis=1)
        df = fix_misscaled_rows(df, required=list(l_item))
        df = drop_placeholder_rows(df)
        for name in optional_fields:
            if name not in df.columns:
                df[name] = 0.0
            else:
                df[name] = df[name].fillna(0.0)
        list_index: list = df.index.tolist()
        multi_index = [(self.ticker,i) for i in list_index]
        multi_index = pd.MultiIndex.from_tuples(multi_index)
        df.index = multi_index
        df.index.names = ['ticker','date']
        return df

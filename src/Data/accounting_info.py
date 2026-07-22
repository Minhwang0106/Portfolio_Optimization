import pandas as pd
import requests


class Sec_Data_Restructure:
    class_description: dict = {}
    link: str = "https://www.sec.gov/files/company_tickers.json"
    header: dict = {
                'User-Agent': 'risotoblack@gmail.com'
            }
    root_companyfacts: str = "https://data.sec.gov/api/xbrl/companyfacts/CIK"

    def __init__(self,ticker:str, header=header) -> None:
        self.ticker = ticker.upper().replace('.','-')
        self.header = header
        self.share_info, self.fs_info = self.get_facts(ticker)

    def cik_matching(self, ticker: str) -> str:
        """Look up a ticker's 10-digit zero-padded SEC CIK number.

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

    @staticmethod
    def _dedup_facts(raw_data: list[dict]) -> dict:
        """Collapse a list of SEC fact entries into one value per end-date.

        When multiple entries share the same 'end' date, an amended filing
        ('form' ending in '/A') takes precedence over the original.

        Args:
            raw_data (list[dict]): Fact entries, each with keys 'end', 'val',
                'form' (as returned by the SEC companyfacts API).

        Returns:
            dict: Mapping of end-date string to value.

        Example:
            >>> Sec_Data_Restructure._dedup_facts([
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

    @classmethod
    def listdict_to_df(cls, raw_data: list[dict], ticker: str) -> pd.DataFrame:
        """Convert a list of SEC fact entries into a single-column DataFrame.

        Args:
            raw_data (list[dict]): Fact entries with keys 'end', 'val', 'form'.
            ticker (str): Column name to use for the values.

        Returns:
            pd.DataFrame: Indexed by datetime 'date', with one column named
                `ticker` holding the deduplicated values.

        Example:
            >>> Sec_Data_Restructure.listdict_to_df(
            ...     [{'end': '2020-12-31', 'val': 100, 'form': '10-K'}], 'AAPL')
                        AAPL
            date
            2020-12-31   100
        """
        date_to_val: dict = cls._dedup_facts(raw_data)
        date: list = list(date_to_val.keys())
        val: list = list(date_to_val.values())
        df: pd.DataFrame = pd.DataFrame(val,columns=[ticker],
                                        index=pd.Index(date,name='date'))
        df.index = pd.to_datetime(df.index)
        return df

    @staticmethod
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
            >>> Sec_Data_Restructure.to_calendar_quarter(pd.Timestamp('2020-02-15'))
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

    def share_compose (self)-> pd.DataFrame:
        """Build a shares-outstanding time series for this instance's ticker.

        Prefers the dei 'EntityCommonStockSharesOutstanding' tag; falls back
        to the us-gaap 'CommonStockSharesOutstanding' tag for multi-class
        filers where the dei aggregate is absent.

        Returns:
            pd.DataFrame: MultiIndex ('ticker', 'date'), single column
                'shares' with the number of shares outstanding.

        Raises:
            ValueError: If neither tag is present in the fetched facts.

        Example:
            >>> obj.share_compose().head(1)
                              shares
            ticker date
            AAPL   2020-01-31  4375479000
        """
        if 'EntityCommonStockSharesOutstanding' in self.share_info:
            unit_key = next(iter(self.share_info[
                'EntityCommonStockSharesOutstanding']['units']))
            info: list = self.share_info[
                'EntityCommonStockSharesOutstanding']['units'][unit_key]
        elif 'CommonStockSharesOutstanding' in self.fs_info:
            # multi-class filers (e.g. dual-class common stock) tag shares
            # outstanding per class, so the dei aggregate above is absent;
            # fall back to the us-gaap balance-sheet tag.
            unit_key = next(iter(self.fs_info['CommonStockSharesOutstanding']['units']))
            info = self.fs_info['CommonStockSharesOutstanding']['units'][unit_key]
        else:
            raise ValueError(f'{self.ticker}: no shares-outstanding tag found in dei or us-gaap')
        df: pd.DataFrame = self.listdict_to_df(info,self.ticker)
        df.columns = ['shares']
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
        tags summed together over their common dates. Dates are snapped to
        calendar quarter-ends via `to_calendar_quarter`.

        Args:
            l_item (dict[str, list[str | tuple[str, ...]]]): Mapping of
                output field name to a list of candidate us-gaap tags. Each
                candidate is either a tag name or a tuple/list of tag names
                to sum together (e.g. {'revenue': ['Revenues', ('SalesA','SalesB')]}).
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
            units: list[dict] = []
            for tag in candidate_tags:
                if isinstance(tag, (list, tuple)):
                    group_tags = tag
                    if not all(t in self.fs_info for t in group_tags):
                        continue
                    sub_series = []
                    for t in group_tags:
                        unit_key = next(iter(self.fs_info[t]['units']))
                        sub_series.append(self._dedup_facts(self.fs_info[t]['units'][unit_key]))
                        description.setdefault(name, self.fs_info[t]['description'])
                    common_dates = set.intersection(*(set(s) for s in sub_series))
                    for d in common_dates:
                        total = sum(s[d] for s in sub_series)
                        units.append({'end': d, 'val': total, 'form': '10-K'})
                    continue
                if tag not in self.fs_info:
                    continue
                unit_key: str = next(iter(self.fs_info[tag]['units']))
                units.extend(self.fs_info[tag]['units'][unit_key])
                description.setdefault(name, self.fs_info[tag]['description'])
            if not units:
                if name in optional_fields:
                    continue
                raise ValueError(
                    f"{self.ticker}: none of {candidate_tags} found for '{name}'"
                )
            temp_df: pd.DataFrame = self.listdict_to_df(units,name)
            temp_df.index = temp_df.index.map(self.to_calendar_quarter)
            temp_df.index.name = 'date'
            temp_df = temp_df[~temp_df.index.duplicated(keep='first')]
            item_dfs.append(temp_df)
        df: pd.DataFrame = pd.concat(item_dfs, axis=1)
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

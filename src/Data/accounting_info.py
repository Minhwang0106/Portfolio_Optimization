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
        cik: str = self.cik_matching(ticker)
        url: str = self.root_companyfacts + cik + '.json'
        response = requests.get(url, headers=self.header, timeout=30)
        response.raise_for_status()
        company_facts: dict = response.json()
        shares_fact: dict = company_facts['facts']['dei']
        fs_fact: dict = company_facts['facts']['us-gaap']
        return shares_fact, fs_fact

    @staticmethod
    def listdict_to_df(raw_data: list[dict], ticker: str) -> pd.DataFrame:
        date_to_val: dict = {}
        for inf in raw_data:
            end_date = inf['end']
            if end_date not in date_to_val or inf['form'].endswith('/A'):
                date_to_val[end_date] = inf['val']
        date: list = list(date_to_val.keys())
        val: list = list(date_to_val.values())
        df: pd.DataFrame = pd.DataFrame(val,columns=[ticker],
                                        index=pd.Index(date,name='date'))
        df.index = pd.to_datetime(df.index)
        return df

    @staticmethod
    def to_calendar_quarter(date: pd.Timestamp) -> pd.Timestamp:
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
        info: list = self.share_info['EntityCommonStockSharesOutstanding'
                                     ]['units']['shares']
        df: pd.DataFrame = self.listdict_to_df(info,self.ticker)
        df.columns = ['shares']
        mul_index = [(self.ticker, i) for i in list(df.index)]
        mul_index = pd.MultiIndex.from_tuples(mul_index)
        df.index = mul_index
        df.index.names = ['ticker','date']
        return df

    def line_item_restructure (self, l_item: dict[str, list[str]]):
        description: dict = dict()
        item_dfs: list[pd.DataFrame] = []
        for name, candidate_tags in l_item.items():
            units: list[dict] = []
            for tag in candidate_tags:
                if tag not in self.fs_info:
                    continue
                unit_key: str = next(iter(self.fs_info[tag]['units']))
                units.extend(self.fs_info[tag]['units'][unit_key])
                description.setdefault(name, self.fs_info[tag]['description'])
            if not units:
                raise ValueError(
                    f"{self.ticker}: none of {candidate_tags} found for '{name}'"
                )
            temp_df: pd.DataFrame = self.listdict_to_df(units,name)
            temp_df.index = temp_df.index.map(self.to_calendar_quarter)
            temp_df.index.name = 'date'
            item_dfs.append(temp_df)
        df: pd.DataFrame = pd.concat(item_dfs, axis=1)
        list_index: list = df.index.tolist()
        multi_index = [(self.ticker,i) for i in list_index]
        multi_index = pd.MultiIndex.from_tuples(multi_index)
        df.index = multi_index
        df.index.names = ['ticker','date']
        return df

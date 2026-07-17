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
        self.ticker = ticker
        self.header = header
        self.share_info, self.fs_info = self.get_facts(ticker)

    def cik_matching(self, ticker: str) -> str:
        ticker = ticker.upper().replace('.','-')
        ticker_json: dict = requests.get(self.link, headers=self.header).json()
        for s in ticker_json.values():
            if s['ticker']==ticker:
                cik: str = str(s['cik_str']).zfill(10)
                return cik
        raise ValueError(f'{ticker} is unavailable in SEC database')

    def get_facts(self, ticker: str):
        cik: str = self.cik_matching(ticker)
        url: str = self.root_companyfacts + cik + '.json'
        company_facts: dict = requests.get(url, headers=self.header).json()
        shares_fact: dict = company_facts['facts']['dei']
        fs_fact: dict = company_facts['facts']['us-gaap']
        return shares_fact, fs_fact

    @staticmethod
    def listdict_to_df(raw_data: list[dict], ticker: str) -> pd.DataFrame:
        date: list = []
        val: list = []
        previous_date: str = ''
        for inf in raw_data:
            if not len(previous_date):
                date.append(inf['end'])
                val.append(inf['val'])
            else:
                if inf['end']==previous_date:
                    if inf['form'][-1]=='A':
                        val[-1]=inf['val']
                else:
                    date.append(inf['end'])
                    val.append(inf['val'])
            previous_date = date[-1]
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
        info: list = self.share_info['EntityCommonStockSharesOutstanding']['units']['shares']
        df: pd.DataFrame = self.listdict_to_df(info,self.ticker)
        return df

    def line_item_restructure (self, l_item: list[str]):
        description: dict = dict()
        df: pd.DataFrame = pd.DataFrame()
        for li in l_item:
            unit_key: str = next(iter(self.fs_info[li]['units']))
            units: list[dict] = self.fs_info[li]['units'][unit_key]
            description[li] = self.fs_info[li]['description']
            temp_df: pd.DataFrame = self.listdict_to_df(units,li)
            temp_df.index = temp_df.index.map(self.to_calendar_quarter)
            temp_df.index.name = 'date'
            if not len(df):
                df = temp_df
            else:
                df = pd.concat([df,temp_df],axis=1)
        col_index = [(self.ticker, i) for i in l_item]
        col_index = pd.MultiIndex.from_tuples(col_index)
        df.columns = col_index
        return df

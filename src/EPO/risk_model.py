import pandas as pd
import numpy as np
from constant import (daily_FF5_path,
                       monthly_FF5_path,
                       daily_price_path,
                       monthly_price_path,
                       applicable_ticker_path,
                       testing_period)
from .excess_return import cal_excess_return


class Risk_Model:
    def __init__(self, ticker: list[str], date:str) -> None:
        self.ticker = ticker
        self.date = date
    
    @classmethod
    def class_config(cls, com=60, rolling = 3 ):
        #er is abbreviation for excess return
        daily_er: pd.DataFrame = cal_excess_return(daily_price_path,
                                                daily_FF5_path)
        monthly_er: pd.DataFrame = cal_excess_return(monthly_price_path,
                                                    monthly_FF5_path)
        overlap_r: pd.DataFrame = daily_er.rolling(rolling).sum()
        applicable_ticker: pd.DataFrame = pd.read_csv(applicable_ticker_path)
        app
        cls.var = daily_er.ewm(com=com, adjust=False).var()
        
        
        pass
    


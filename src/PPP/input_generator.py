import pandas as pd
import numpy as np
from constant import (
    accounting_path, monthly_price_path
)
from pathlib import Path

def read_csv_file (path: Path, date: str|None = 'date',
                   index: list[str]=['ticker','date'])-> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(path)
    if date is not None:
        try:
            df[date] = pd.to_datetime(df[date])
        except KeyError:
            raise KeyError(f"{date} is not in data columns")
    try:
        df.set_index(index, inplace=True)
        df.sort_index(inplace=True)
    except KeyError as e:
        raise e
    return df

def generator (accout_path: Path = accounting_path,
               price_path: Path = monthly_price_path)->pd.DataFrame:
    account_df: pd.DataFrame = read_csv_file(accout_path)
    price_df: pd.DataFrame = read_csv_file(price_path)
    columns_value: list[pd.Series] = []
    col_keys: list[str] = ['book_value', 'market_cap' , 'adj_close']
    
    #book value
    book_value: pd.Series = account_df[
        'book_value']-account_df['minority_interest'
                                 ]-account_df['preferred_stock']
    columns_value.append(book_value.shift(1))
    #market_cap
    columns_value.append(price_df['close']*price_df['shares'])
    #adj_close
    columns_value.append(price_df['adj_close'])
    
    df: pd.DataFrame = pd.concat(columns_value,axis=1,keys=col_keys)
    df = df.sort_index().groupby(level=0).ffill()
    return df
    
    
    


    
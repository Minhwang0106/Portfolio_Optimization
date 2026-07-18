import pandas as pd
import requests
from .accounting_info import Sec_Data_Restructure
from .price_data import take_price
from .constant import ticker, ratio, price_csv_path, accounting_csv_path

def run (ticker=ticker, ratio=ratio,
         price_path=price_csv_path,
         accounting_path=accounting_csv_path)->tuple[pd.DataFrame, pd.DataFrame]:
    price_listframe: list[pd.DataFrame] = []
    accounting_listframe: list[pd.DataFrame] = []
    errors: list[str] = []
    for tick in ticker:
        try:
           tick_object: Sec_Data_Restructure = Sec_Data_Restructure(tick)
           price: pd.DataFrame = take_price(tick)
           shares: pd.DataFrame = tick_object.share_compose()
           accounting: pd.DataFrame = tick_object.line_item_restructure(ratio)

           price = pd.concat([price,shares],axis=1).sort_index(level='date')
           price['shares'] = price['shares'].ffill()
           price.dropna(inplace=True)

           accounting = pd.concat([accounting,shares],axis=1).sort_index(level='date')
           accounting['shares'] = accounting['shares'].ffill()
           accounting.dropna(inplace=True)

           price_listframe.append(price)
           accounting_listframe.append(accounting)
        except (ValueError, requests.exceptions.RequestException) as e:
            errors.append(f'{tick}: {e}')

    if errors:
        print(f'{len(errors)} ticker(s) skipped due to errors:')
        for err in errors:
            print(f'  - {err}')

    if not price_listframe or not accounting_listframe:
        raise ValueError('No ticker succeeded; nothing to save.')

    multi_tick_price: pd.DataFrame = pd.concat(price_listframe,axis=0)
    multi_tick_accounting: pd.DataFrame = pd.concat(accounting_listframe,axis=0)

    price_path.parent.mkdir(parents=True, exist_ok=True)
    accounting_path.parent.mkdir(parents=True, exist_ok=True)
    multi_tick_price.to_csv(price_path)
    multi_tick_accounting.to_csv(accounting_path)

    return multi_tick_price, multi_tick_accounting

if __name__ == "__main__":
    run()

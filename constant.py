import pandas as pd
import numpy as np
from pathlib import Path

#General variable
PROJECT_DIR: Path = Path(__file__).resolve().parent
RAW_DATA_DIR: Path = PROJECT_DIR / "Data" / "raw file"

daily_price: str = "daily_price.csv"
monthly_price: str = "monthly_price.csv"
accounting: str = "accounting.csv"
daily_FF5: str = "daily_FF5.csv"
monthly_FF5: str = "monthly_FF5.csv"
price: str = "price.csv"
sp_500_historical_components: str = "sp_500_historical_components.csv"
ticker: str = "ticker.csv"
applicable_ticker: str = "applicable_ticker.csv"

daily_price_path: Path = RAW_DATA_DIR / daily_price
monthly_price_path: Path = RAW_DATA_DIR / monthly_price
accounting_path: Path = RAW_DATA_DIR / accounting
daily_FF5_path: Path = RAW_DATA_DIR / daily_FF5
monthly_FF5_path: Path = RAW_DATA_DIR / monthly_FF5
price_path: Path = RAW_DATA_DIR / price
sp_500_path: Path = RAW_DATA_DIR / sp_500_historical_components
ticker_path: Path = RAW_DATA_DIR / ticker
applicable_ticker_path: Path = RAW_DATA_DIR / applicable_ticker

testing_period:pd.DatetimeIndex = pd.date_range("2015-01-01", 
                                                "2026-01-01", freq='ME')

#Data Preprocessing
ticker_data: list[str] = pd.read_csv(ticker_path, index_col=0)[
    "ticker"].tolist()

ratio: dict[str, list[str | tuple[str, ...]]] = {
    'net_income': ['NetIncomeLoss', 'ProfitLoss'],
    'book_value': [
        'StockholdersEquity',
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
    ],
    'total_assets': ['Assets'],
    'revenue': [
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
        'Revenues',
        'SalesRevenueNet',
        # banks/financials report revenue split as interest + noninterest income
        ('InterestAndDividendIncomeOperating', 'NoninterestIncome'),
        ('InterestAndFeeIncomeLoansAndLeases', 'NoninterestIncome'),
    ],
}
n_quarter: int = 20
n_month: int = 120

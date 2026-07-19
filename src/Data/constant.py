from pathlib import Path

import pandas as pd

# src/Data/constant.py -> src/Data -> src -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_FILE_ROOT: Path = PROJECT_ROOT / "Data" / "raw file"
TICKER_CSV: str = "ticker.csv"
daily_price: str = "daily_price.csv"
monthly_price: str = "monthly_price.csv"
accounting_data: str = "accounting.csv"

TICKER_CSV_PATH: Path = RAW_FILE_ROOT/ TICKER_CSV
monthly_price_csv_path: Path = RAW_FILE_ROOT/ monthly_price
daily_price_csv_path: Path = RAW_FILE_ROOT/ daily_price
accounting_csv_path: Path = RAW_FILE_ROOT/accounting_data




ticker: list[str] = pd.read_csv(TICKER_CSV_PATH, index_col=0)[
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

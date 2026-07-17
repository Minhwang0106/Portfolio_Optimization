from pathlib import Path

import pandas as pd

# src/Data/constant.py -> src/Data -> src -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TICKER_CSV_PATH: Path = PROJECT_ROOT / "Data" / "raw file" / "ticker.csv"

ticker: list[str] = pd.read_csv(TICKER_CSV_PATH, index_col=0)[
    "ticker"].tolist()

ratio: dict[str, list[str]] = {
    'net_income': ['NetIncomeLoss'],
    'book_value': ['StockholdersEquity'],
    'total_assets': ['Assets'],
    'revenue': [
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
        'Revenues',
        'SalesRevenueNet',
    ],
}
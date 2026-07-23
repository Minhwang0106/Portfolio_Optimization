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
risk_aversion: float = 5.0

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
    # RIM clean-surplus needs BV net of NCI and preferred (see Instruction/data-collection-rim.md §4.1).
    # Most single-class filers never report these tags at all (no NCI, no preferred stock),
    # so they are optional: missing -> defaults to 0 instead of excluding the ticker.
    'minority_interest': ['MinorityInterest'],
    'preferred_stock': ['PreferredStockValue'],
}

# Fields in `ratio` that legitimately default to 0 when the tag is absent from a filer's
# XBRL facts (e.g. no NCI, no preferred stock) rather than causing the ticker to be dropped.
optional_ratio_fields: frozenset[str] = frozenset({'minority_interest', 'preferred_stock'})

# Shares-outstanding tags in descending order of preference, as (taxonomy, tag).
# The dei cover-page count is the point-in-time figure we actually want, but it
# is thin or zero-valued for multi-class and reorganized filers, so the rest are
# there to fill the dates it leaves uncovered (see Sec_Data_Restructure.share_compose).
# The last is a period average rather than a point-in-time count -- accepted as a
# last resort because a slightly stale share count beats no market cap at all.
share_tags: tuple[tuple[str, str], ...] = (
    ('dei', 'EntityCommonStockSharesOutstanding'),
    ('us-gaap', 'CommonStockSharesOutstanding'),
    ('us-gaap', 'CommonStockSharesIssued'),
    ('us-gaap', 'WeightedAverageNumberOfSharesOutstandingBasic'),
)

# A shares figure is carried forward at most this many rows before going NaN.
# Filings are quarterly, so ~4 quarters of staleness is the most that can be
# defended; beyond that the number is invented rather than merely stale.
share_ffill_month: int = 12
share_ffill_quarter: int = 4
n_quarter: int = 20
n_month: int = 75

#Enhanced Portfolio Optimization(EPO)
risk_com: int = 60
risk_correl_com: int = 150
risk_rolling: int = 3
risk_n_day: int = 261
risk_n_month: int = 12
epo_shrinkage: float = 0.75
epo_theta: float = 1.0

#Parametric Portfolio Policy
n_cum_month = 12



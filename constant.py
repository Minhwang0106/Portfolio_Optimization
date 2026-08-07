import pandas as pd
import numpy as np
from pathlib import Path

#General variable
PROJECT_DIR: Path = Path(__file__).resolve().parent
RAW_DATA_DIR: Path = PROJECT_DIR / "Data" / "raw file"
# Everything the project produces lives under Data/Result, split by how far it
# has been worked up: `raw_backtest` is the full output of a run, most of which
# no table ever shows; `processed_backtest` and `latex` hold *only* the numbers
# that appear in the paper, the same content in the two forms it is needed in.
RESULT_DIR: Path = PROJECT_DIR / "Data" / "Result"
RAW_BACKTEST_DIR: Path = RESULT_DIR / "raw_backtest"
PROCESSED_BACKTEST_DIR: Path = RESULT_DIR / "processed_backtest"
LATEX_DIR: Path = RESULT_DIR / "latex"

daily_price: str = "daily_price.csv"
monthly_price: str = "monthly_price.csv"
accounting: str = "accounting.csv"
daily_FF5: str = "daily_FF5.csv"
monthly_FF5: str = "monthly_FF5.csv"
price: str = "price.csv"
sp_500_historical_components: str = "sp_500_historical_components.csv"
ticker: str = "ticker.csv"
applicable_ticker: str = "applicable_ticker.csv"
splits: str = "splits.csv"

daily_price_path: Path = RAW_DATA_DIR / daily_price
monthly_price_path: Path = RAW_DATA_DIR / monthly_price
accounting_path: Path = RAW_DATA_DIR / accounting
daily_FF5_path: Path = RAW_DATA_DIR / daily_FF5
monthly_FF5_path: Path = RAW_DATA_DIR / monthly_FF5
price_path: Path = RAW_DATA_DIR / price
sp_500_path: Path = RAW_DATA_DIR / sp_500_historical_components
ticker_path: Path = RAW_DATA_DIR / ticker
applicable_ticker_path: Path = RAW_DATA_DIR / applicable_ticker
splits_path: Path = RAW_DATA_DIR / splits

# Split-basis alignment. The share count changes basis when a filing reports
# the new count, and that ran from 18 days before the split (GOOGL) to 7 weeks
# after it (NVDA), so the search window has to cover a two-quarter filing lag.
# Observed jumps miss the exact ratio by ~0.5% -- LRCX reads 9.948 for a 10:1
# -- because buybacks move the count over the same period.
split_search_days: int = 160
split_ratio_tolerance: float = 0.25

testing_period:pd.DatetimeIndex = pd.date_range("2015-01-01",
                                                "2026-01-01", freq='ME')

# Universe basis, read by `Data.exclusion.applicable_ticker`. None reconstitutes
# the candidate pool at every formation date from that date's S&P 500
# membership. A date freezes the pool to the names already applicable then --
# membership *and* the full data screen -- and every later date is that same pool
# re-screened for data availability. Re-screening is the part that matters: it is
# what makes a delisted name leave the universe at the next rebalance instead of
# being carried at a flat return forever by `Empirical_Analysis.engine`.
#
# None is the default because freezing is the more restrictive claim, not the
# safer one. A frozen pool excludes every index entrant after the freeze date,
# so the strategy is measured on a cross-section that the S&P 500 stopped being
# eleven years ago; and the pool it does keep is thin for a mechanical reason.
# `n_quarter` complete quarters ending at the freeze date reach back to 2009,
# before XBRL coverage was general, so only 172 names qualified at 2015-01-31
# against 373 at 2025-12-31 -- a filing-coverage artifact standing in for the
# investable set. Reconstituting fixes both: every date is screened against the
# membership actually in force then.
#
# Freezing is still not lookahead, and a date here is worth setting for a
# robustness check on a fixed opening cross-section -- read it as that and never
# as an S&P 500 strategy. `pd.Timestamp('2015-01-31')` is the date to use: the
# first formation date rather than 2015-01-01, because `testing_period` is
# month-end so the sample opens at 2015-01-31, and dating the freeze nine days
# earlier would back `account_asof` off a whole quarter, to 2014-09-30, and drop
# names that pass the opening date's own screen.
#
# Switching between the two changes what `Data.ticker_list.build_ticker_list`
# writes -- 464 names frozen, 707 as a union over `testing_period` -- so it is
# a re-fetch (`python -m src.Data.run --resume`), not just a re-screen. Once the
# panels cover the wider list, moving back and forth needs only `--skip-fetch`.
#
# As collected: 572 of the 707 resolve, giving 172 names at 2015-01-31 rising to
# 381 at 2025-12-31 and 466 distinct names across the sample.
universe_asof: pd.Timestamp|None = None

risk_aversion: float = 5.0

#Data Preprocessing
# Written by `Data.ticker_list.build_ticker_list` off the constituent history,
# not maintained by hand. Empty when the file has not been built yet, so that
# importing this module is not what stands between a fresh checkout and the
# build step that creates it -- every other name here is still usable.
ticker_data: list[str] = (
    pd.read_csv(ticker_path, index_col=0)["ticker"].tolist()
    if ticker_path.is_file() else [])

ratio: dict[str, list[str | tuple[str, ...]]] = {
    'net_income': ['NetIncomeLoss', 'ProfitLoss'],
    # Common equity, always net of non-controlling interests. 'StockholdersEquity'
    # is already parent-only, so nothing is deducted from it; the fallback tag
    # includes NCI by definition, so MinorityInterest is netted off there (a
    # leading '-' marks a subtracted tag). Getting this wrong in the other
    # direction -- deducting NCI from the parent-only figure -- halves or flips
    # the sign of book equity for the ~48% of rows that have any NCI at all.
    'book_value': [
        'StockholdersEquity',
        ('StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
         '-MinorityInterest'),
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
    # NCI is already netted out of 'book_value' above; this column is kept for
    # reference and must NOT be deducted again by consumers. Preferred stock is
    # still included in the equity tags, so that one does get deducted.
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

# A duration fact (revenue, net income) counts as one fiscal quarter when its
# span in days falls inside this window. Real quarters run 88-97 days (13-14
# weeks under a 4-4-5 calendar); the next span up, a half-year, starts at 180,
# so there is a wide margin either side. Facts outside the window are cumulative
# year-to-date figures and get differenced into quarters, not used as they are.
quarter_span_days: tuple[int, int] = (80, 100)

# A shares-outstanding value is rejected when it is more than this many times
# smaller than the ticker's median. Early XBRL filings sometimes tag the count
# in millions (394 where 394,000,000 was meant, e.g. A's 2007-2009 weighted
# average), and no legitimate corporate action moves a share count 100x.
share_scale_tolerance: float = 100.0

# Manual share-count conversions, ticker -> factor, for filers whose SEC count is
# denominated in a different share class than the price series. The companyfacts
# API only exposes non-dimensional facts, so a per-class count is unreachable in
# code -- these have to be looked up and stated.
#
# BRK-B: Berkshire reports on an equivalent Class A basis (1,643,393 shares at
# 2015-12-31 per its 10-K, matching the value in its XBRL), while the price
# series is Class B. One Class A converts into 1,500 Class B, and Yahoo's close
# is split-adjusted back through the January 2010 50:1 split, so the one factor
# holds across the whole history: 1,643,393 * 1500 * $132 gives the ~$325bn
# Berkshire was actually worth at the end of 2015.
share_multiplier: dict[str, float] = {'BRK-B': 1500.0}

# Ticker -> CIK, overriding what SEC's `company_tickers.json` resolves. Only for
# re-incorporations, where the symbol has been reassigned to a freshly
# registered holding company filing under its own CIK and carrying none of the
# operating company's history.
#
# XOM is the case that forced this. SEC maps it to 0002115436, 'ExxonMobil
# Holdings Corp', whose companyfacts carry the 'ffd' taxonomy alone -- no
# 'us-gaap', no 'dei', no financials of any kind -- so `get_facts` raises and
# Exxon leaves the collection entirely. The operating company, 0000034088, has
# the full 438-tag history. This is not a judgement about which entity is
# 'really' Exxon; it is that the accounting panel wants the one with accounts.
#
# Keep this list short. A ticker resolving to the wrong company is a data bug to
# be confirmed against the filings one at a time, not a list to grow by reflex:
# every entry silently overrides the authoritative source for that symbol, and
# will keep overriding it after SEC fixes the mapping.
cik_override: dict[str, str] = {'XOM': '0000034088'}

# Balance-sheet field used to detect filings that report a placeholder balance
# sheet, and how far below the ticker's own median it has to fall to count. A
# real one cannot collapse 100x from one quarter to the next; when it does, the
# filer is a newly-registered holding company reporting before its merger closed
# (ICE files total assets of 10, SW 111, TRIP 1000, QRVO 0) while still carrying
# the predecessor's income statement. Those rows are dropped.
scale_screen_field: str = 'total_assets'
scale_screen_tolerance: float = 100.0

# A filing whose every field sits the same power-of-1000 factor away from the
# ticker's median was tagged in thousands or millions while still declaring the
# unit as USD (LUV's 2011-06-30 10-Q reports revenue of 4136 between quarters of
# 3.10e9 and 4.31e9). Rescaled rather than dropped, since dropping would also
# break the quarter-on-quarter chain for the filing that follows. The row only
# qualifies when the per-field factors agree within `scale_fix_agreement` and
# land within `scale_fix_rel_error` of an exact power of 1000.
scale_fix_powers: tuple[int, ...] = (1_000, 1_000_000)
scale_fix_agreement: float = 10.0
scale_fix_rel_error: float = 0.5

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
# Shrinkage intensity of the correlation matrix toward the identity:
# `Sigma_w = vol @ ((1-w)*C + w*I) @ vol`. w = 0 is plain mean-variance, w = 1
# ignores correlations entirely.
#
# This is the weight on the *identity*. Chosen per formation date now, by the
# paper's own out-of-sample procedure (Pedersen, Babu & Levine 2021, the w
# selection algorithm): `EPO.Config`/`EPO._select_w` walks `testing_period`
# forward and, at each date, picks the point in `epo_shrinkage_grid` whose
# *unconstrained* closed-form weight earned the best realised Sharpe ratio over
# every earlier formation date -- a w is never scored against the date it is
# then applied to. `epo_shrinkage` below is therefore no longer "the"
# shrinkage; it is only the fallback used before `epo_w_min_periods` months of
# realised history exist to select from.
#
# Selection scores the unconstrained closed form even though the book actually
# held is long-only-constrained (`EPO._long_only_weight`): the constrained
# solve is a ~0.1s SLSQP call, and the grid search scores every candidate w
# against every earlier date at every date, so paying that cost per candidate
# would turn a few seconds of linear algebra into hours. This also matches the
# paper's own algorithm, which has no long-only constraint to begin with -- so
# selection is not solving a materially different problem, just the one the
# paper specifies rather than the constrained variant this repo additionally
# imposes on the weights actually held.
#
# `C` is estimated on ~150 names from 261 daily observations of 3-day
# overlapping returns and its condition number runs 7.5e3 to 4.0e4, so the
# unshrunk inverse is mostly reading estimation noise: at w = 0 the raw
# solution carries 350x gross exposure, at w = 0.25 still 16x. That is the
# error-maximisation EPO exists to correct, and why a grid weighted toward
# heavier shrinkage (this one runs 0 to 1) matters more than the exact step.
epo_shrinkage: float = 0.90

# Candidate values `EPO._select_w` scores, step 0.05 as in the paper.
epo_shrinkage_grid: tuple[float, ...] = tuple(
    round(float(x), 2) for x in np.arange(0.0, 1.01, 0.05))

# Formation dates of realised history required before the out-of-sample
# selection above runs; before this many, `epo_shrinkage` is used instead. The
# paper warms up over 15 years before its first out-of-sample pick; this
# sample spans 11 years end to end (`testing_period`), so matching that would
# leave nothing left to test on. 24 months is a compromise: enough that the
# grid's realised-Sharpe estimates are not pure noise, short enough to leave
# most of the sample under actual selection rather than the fallback.
epo_w_min_periods: int = 24

# Weight on `C` in a first blend toward the identity, applied before the
# selected w. At 1.0 it is the identity operation, which is what it has
# always been set to; the two-stage form is kept because it is the paper's.
epo_theta: float = 1.0

#Parametric Portfolio Policy
n_cum_month = 12
#Proposed Model
n_quarter_ahead: int = 80

# Quarters of complete characteristics a ticker needs before it is fitted, out
# of the `n_quarter` in the training window. `Data.exclusion` guarantees the raw
# accounting columns are present, but not that they survive the transforms in
# `Proposed_Model.data_generator` -- a ratio outside its transform's domain
# becomes NaN, so a filer with complete filings can still reach the copula with
# nothing usable. At 2024-12-31 twelve tickers fall under 5 rows and AVB, EXR and
# ESS have none at all, which is what `copula_structure` reports as 'no
# association available'. Set at 12 of 20: comfortably above `n_lags` (4), which
# `sampling` needs for its signal, while still leaving a copula fit something to
# work with. Raise it for a stricter universe, lower it to admit more names.
min_char_obs: int = 12

# Largest per-quarter ROE `rim_mapping` will compound residual income forward
# at, applied symmetrically. Reinvesting at the simulated ROE puts that ROE in
# an exponent, `exp(sum of roe over the horizon)`, which turns the right tail of
# the leverage draw into a numerical blow-up rather than an unlikely path.
# `Ate_Trans.inverse` is `exp(x)+1` and so unbounded: at 2021-06-30 QCOM draws
# paths at 1315x assets-to-equity and 634% quarterly ROE, AMT reaches a summed
# ROE of 587, and `exp(587)` is 1e254. Unclipped, 7.9% of AMT's paths exceed
# exp(sum) > 1e6 and the cross-sectional mean implied return reaches 7.2e46 --
# the mean is then the tail, not the distribution.
#
# 0.15 per quarter is ~60% a year, comfortably above any sustained real ROE, so
# it binds on simulation artefacts rather than on firms. It is a tourniquet, not
# a fix: the same unbounded `ate` also drives `be_arr = rps/(ate*ato)` toward
# zero on those paths and distorts residual income directly, which clipping here
# does not touch. Winsorising the characteristics in `sampling` is the real
# repair. Lowering this tightens the cross-section -- 0.15 leaves 44.4% of names
# with a positive implied return at 2021-06-30, 0.05 leaves 33.1%.
reinvest_rate_cap: float = 0.15

# Smallest value `terminal_val` will divide its Gordon perpetuity by. The
# denominator is `1 - g_factor*exp(-re*n)`, and `g_factor` is built from
# simulated growth, so nothing bounds it below the discount factor: once
# `g_factor*exp(-re*n)` passes 1 the denominator turns negative and the
# perpetuity stops converging. That is outside the formula's domain, not merely
# near a singularity. Measured over 22 formation dates, 2.6% of paths land there
# universe-wide but 68.7% of the paths behind predictions above 60% do -- and
# those are the ones the optimiser buys. Weighting by realised portfolio weight,
# terminal value averages 29x the share price against 0.1x equal-weighted. TFC
# at 2020-07-31 is the extreme: every path negative, a terminal value 562x price,
# a predicted return of 546%, and 87.7% of the book.
#
# Clamped from below rather than in magnitude, so the sign is forced positive.
# Flooring |denominator| and preserving the sign leaves the invalid branch intact
# and measures as a no-op -- rank correlation 0.9978 against the unfloored model,
# TFC unmoved at 546% -- because those paths are not close to zero. TFC's median
# |denominator| is 0.083, above any floor worth setting.
gordon_den_floor: float = 0.01

#Empirical Analysis
# One-way transaction cost in basis points of traded notional, charged against
# the month following each rebalance. The universe is S&P 500 large caps traded
# at month end, where 10bp is the conventional assumption. Not a parameter to
# tune -- it is here so that a comparison between a quarterly and a monthly
# strategy is not decided by trading nobody paid for.
backtest_cost_bps: float = 10.0

# Estimation window, in months, for the parametric portfolio policy's theta.
# `PPP.find_theta` takes the same number as its own default; named here so the
# backtest and the model cannot drift apart silently.
ppp_estimation_month: int = 60



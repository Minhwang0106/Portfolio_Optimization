# ICC_MVO

Bielstein & Hanauer (2019), *Mean-variance optimization using forward-looking
return estimates*, Review of Quantitative Finance and Accounting 52, 815–840:
a maximum-Sharpe portfolio whose expected returns are Gebhardt, Lee &
Swaminathan's (2001) implied cost of capital plus rescaled momentum. In this
paper it is the design's **Arm A**, the point-estimate benchmark, and it runs as
`icc_mvo_ex_post` in `src.Empirical_Analysis.run`.

```
python -m src.Data.industry --user-agent "Your Name you@example.com"        # once
python -m src.Data.industry_pool --user-agent "Your Name you@example.com"   # once
python -m src.Empirical_Analysis.run --only equal_weight icc_mvo_ex_post
```

## What is computed at each formation date

| step | B&H / GLS | here | code |
|---|---|---|---|
| ICC | GLS residual income, T = 12, three explicit years, ROE fading linearly to the industry median, clean surplus at payout k | same | `icc_adjusted.gls_path`, `icc_gls` |
| Expected return | ICC + 12-1 momentum rescaled to the ICC's cross-sectional sd − r_f, both winsorised 1%/99% | same, divided by 12 to monthly | `icc_adjusted.expected_excess_return` |
| Covariance | Ledoit–Wolf (2004, JPM) shrinkage to constant correlation, 60 months | same | `volatility.ledoit_wolf_cc` |
| Weights | max Sharpe, long-only, ≤ 5% per name, < 0.01% set to 0 | same | `utils.max_sharpe_weight` |

The GLS implementation reproduces both of GLS's worked examples (GM 13.94%, JNJ
7.12%; `tests/test_icc_mvo.py`).

## Where the inputs differ from B&H's

Each is a substitution for data this repo does not have. Each needs a sentence in
the Methods section.

1. **Explicit EPS are realised, not forecast. This is lookahead.** B&H use IBES
   consensus. Here FY1–FY3 are the EPS realised over the three four-quarter
   blocks after the accounting cutoff (net income over split-adjusted shares).
   The book formed at `t` has therefore seen three years of future earnings,
   which is why the row is `icc_mvo_ex_post` and is read against
   `proposed_ex_post`, never against a tradeable row. The two oracles are not
   the same window: this one sees 12 quarters, and the proposed model's ex post
   moments see up to 80.
2. **The panel runs out.** Most firms' accounting ends 2026-03-31. From the
   2023-09-30 formation date onwards, the years past the panel are extrapolated
   from the last realised one at the sustainable growth rate `(1 − k)·ROE_ind`,
   and from E0 if no year is realised at all. By 2025-12-31 no year is realised.
   `Icc_Mvo.inputs['n_oracle_years']` records the count per name. Firms acquired
   before their third year are extrapolated the same way.
3. **The payout ratio comes from adjusted closes.** There is no Compustat
   dividend field. Dividends are recovered from Yahoo's close/adjusted-close
   factor, `D_e = C_{e−1}(1 − f_{e−1}/f_e)`, summed over the same four quarters
   as E0. The recovered amounts match the cash paid to the cent (KO, AAPL, JNJ,
   MSFT, COST's special). GLS's rule is then applied as they state it:
   `k = DPS/EPS`, or `DPS/(0.06·TA)` when EPS ≤ 0, clipped to [0, 1].
4. **The industry ROE pool is every SEC XBRL filer, not Compustat.**
   - The median over profitable firm-years in the FF48 industry, pooled over the
     last 5–10 calendar years (ROE = the year's net income over the prior
     December's common book equity, defined as `panel.book_equity`).
   - The pool comes from SEC's XBRL frames (`src.Data.industry_pool`): every
     filer, not only this repo's S&P 500 panel, so who is in it involves no
     hindsight. Firm-years starting from under $10m of equity
     (`constant.industry_pool_min_equity`) are left out as shells.
   - A frame holds the value last filed, so restatements replace originals. A
     non-December filer's year is aligned to the nearest calendar year, and its
     opening equity is its balance at the calendar year-end.
   - XBRL phased in over 2009–2011, largest filers first, and the 2015
     formation dates sit on the five-year floor, so their pool leans towards
     large firms.
   - The SIC code is SEC's *current* one, not the code in force at the time.
   - An industry with fewer than `constant.industry_min_obs` (10) profitable
     firm-years takes the all-firm median instead.
   - Without `industry_roe_pool.csv` the median falls back to the S&P 500
     panel's own firm-years, and without `industry.csv` every name takes the
     all-firm median; each with a warning. `Icc_Mvo.roe_pool_source` says
     which pool was used. Neither fallback is a specification to report; the
     panel pool can be asked for on purpose, as a robustness check, with
     `Icc_Mvo.Config(force=True, pool_file=None)`.
5. **Book value is at the latest quarter, not the fiscal year-end.** B0 is
   common book equity (`panel.book_equity`) per split-adjusted share at the
   accounting cutoff. The explicit years start right after the cutoff, so B0
   and year 1 line up.
6. **The risk-free rate** is the Fama–French one-month bill rate of the
   formation month, compounded to a year.
7. **Rebalancing** is quarterly by default, on the proposed model's calendar, so
   that the two ex post rows trade equally often. `--icc-annual` rebalances each
   June, as B&H do.
8. **The universe** is the repo's point-in-time S&P 500 screen
   (`Data.exclusion`), not B&H's sample. On top of that screen, a name needs:
   - a positive price and positive book equity;
   - an ICC root in (0.01%, 100%);
   - 11 months of momentum;
   - all 60 months of returns, because the estimator needs a complete panel.

   `Icc_Mvo.dropped` records which screen removed each name.

## Files

- `icc_adjusted.py`: the GLS forecast path, value and root; winsorising,
  rescaled momentum, and the expected excess return.
- `volatility.py`: Ledoit–Wolf constant-correlation shrinkage.
- `inputs.py`: every input built from the panels, one function each.
- `model.py`: `Icc_Mvo` (panels in `Config`, one date per instance), and
  nothing else.
- `utils.py`: `max_sharpe_weight`, the convex solve `Icc_Mvo.weight` calls,
  and `_at`, a one-row lookup on a wide panel.

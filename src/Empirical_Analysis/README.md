# Empirical_Analysis

Backtests the three portfolio models on one universe and one calendar, so that a
difference between them is a difference in the weighting rule and nothing else.

```
python -m src.Empirical_Analysis.run --n-workers 5
```

Writes to `constant.RESULT_DIR` (`Result/`). Add `--only equal_weight epo ppp`
to skip the slow ones while iterating.

**Changed a metric? Don't rerun the backtest.**

```
python -m src.Empirical_Analysis.run --rebuild
```

`--rebuild` (or `run.rebuild()`) replays the weights the last run already wrote
back through the same engine — same drift, same turnover, same costs — and
recomputes everything downstream. Seconds instead of hours, and it imports no
model at all. It replays rather than re-reading `summary.csv` because any
statistic of the *held* book, `avg_weight_entropy` among them, is a per-month
quantity the saved aggregates cannot reconstruct.

It verifies the replayed returns against the saved `monthly_returns.csv` and
raises if they differ, so a `--cost-bps` or date range that does not match the
original run fails loudly instead of quietly producing a table of wrong
numbers. Pass `verify=False` when a change to the returns is the point.

## Strategies

| label | model | rebalance | weights |
|---|---|---|---|
| `equal_weight` | — | monthly | `1/N`, sums to 1 |
| `epo` | `src.EPO` | monthly | long-only, sums to 1 |
| `ppp` | `src.PPP` | monthly | long-only, sums to 1 |
| `proposed_historical` | `src.Proposed_Model` | quarterly | long-only, sums to 1 |
| `proposed_forward` | `src.Proposed_Model`, `forward=True` | quarterly | long-only, sums to 1 |

All five are long-only and fully invested, which is what makes the return column
comparable across them at all. `--long-short` restores the forms the three
papers state — and those are *not* comparable to each other on a return basis:
unconstrained EPO comes out close to market neutral, and unconstrained PPP runs
to ~3.6x gross exposure. Each model imposes the constraint in its own idiom;
see "How long-only is imposed" below.

`proposed_forward` **is not a strategy.** It elicits each characteristic's
unconditional moments from the realised future window, so a portfolio formed at
`t` has already seen accounting figures through `t + 3 years`. It is in the run
because the gap between it and `proposed_historical` decomposes the model's
error into the simulation machinery and the moment forecast — a ceiling on what
perfect moment forecasting would buy, never a result to quote on its own.

`equal_weight` is the benchmark `PPP` tilts away from, on the same universe, so
PPP's contribution is readable as the gap between the two.

## What the harness fixes

Everything outside the weighting rule has to be identical or the comparison
measures the harness:

- **Universe.** All five get `Data.exclusion.applicable_ticker`'s list at each
  date, spelled out below. Each model then screens it further its own way, which
  is a property of the model and shows up in `avg_weight_entropy`, not equalised
  away.
- **No lookahead.** A weight formed at month end `t` earns the month ending at
  `t+1`. Each model applies its own publication lag upstream; the engine's job
  is not to undo it.
- **Holdings drift.** Between rebalances the book is held, not re-set to target.
  Skipping this gives a quarterly strategy a monthly strategy's turnover and a
  free contrarian tilt.
- **Turnover** is measured against the drifted book, and costs
  (`constant.backtest_cost_bps`, 10bp one-way) are charged against the month the
  trade is held into.

## The exclusion rule

`Data.exclusion.applicable_ticker`, stated exactly. Formation dates are the
month ends in `constant.testing_period` (2015-01-31 through 2025-12-31). At a
date `t`, write

- `q = t - QuarterEnd(1)`, the last quarter end strictly before `t`;
- **price window** `P(t)` = the month ends in `[t - 75 months, t]`, both ends
  included — 76 monthly observations (`constant.n_month = 75`);
- **accounting window** `A(t)` = the quarter ends in `[q - 20 quarters, q]`,
  both ends included — 21 quarterly observations (`constant.n_quarter = 20`).

A ticker is a candidate at `t` iff **all five** hold:

1. **Membership.** It is in the S&P 500 constituent list in force at `t` — the
   most recent snapshot dated at or before `t` (`asof`, not the snapshot nearest
   `t`), read from `constant.sp_500_path`. Symbols are uppercased and `.` is
   mapped to `-` before matching. **Replaced by a fixed pool when
   `constant.universe_asof` is set** — see below.
2. **Price completeness.** `adj_close` is non-missing at *every* date in `P(t)`.
3. **Accounting completeness.** Every accounting column is non-missing at every
   date in `A(t)`: `net_income`, `book_value`, `total_assets`, `revenue`,
   `minority_interest`, `preferred_stock`. `shares` and `adj_shares` are exempt —
   `adj_shares` is derived from `shares` and carries the same gaps, so screening
   on both would count one gap twice. The last two accounting columns are
   zero-filled upstream (`constant.optional_ratio_fields`), so in practice they
   never bind.
4. **Strictly positive book equity.** `BE = book_value - preferred_stock`
   (`panel.book_equity`; NCI is *not* deducted, it is already out upstream) is
   `> 0` at every date in `A(t)`. Coded as `~(BE <= 0)`, so a missing `BE` is
   left to rule 3 rather than being dropped twice, and an exact zero is excluded
   alongside negatives — every consumer divides by `BE`.
5. **Strictly positive revenue.** `revenue > 0` at every date in `A(t)`, same
   treatment of missing and zero, because the residual income model both divides
   by revenue (`ros`) and scales from it (`rps`).

Two consequences worth stating. The accounting window ends one quarter *before*
`t` while the price window ends *at* `t` — that gap is the publication lag built
into the universe itself, before any model applies its own. And a date whose S&P
snapshot is missing gets no entry at all; `Empirical_Analysis.data.universe`
likewise omits dates the table leaves blank, so `universe.get(t)` returning
`None` means "no universe here", not "an empty one".

### Frozen universe (`constant.universe_asof`)

Setting `universe_asof` to a date `a` replaces rule 1 with a fixed pool: rules
1-5 are evaluated once at `a`, and every formation date then applies rules 2-5
to that pool alone. Set it to `None` to get the reconstituting universe
described above. The default is `2015-01-31`, the first formation date — not
2015-01-01, because `q = a - QuarterEnd(1)` would back the accounting window off
to 2014-09-30 and drop names that pass the opening date's own screen.

Rules 2-5 still run per date, and that is the load-bearing part: a name that
delists leaves the universe at the next formation date instead of sitting in it
while `engine` carries it at a flat return in `unpriced_weight`. Freezing
*without* re-screening is what would bias the run upward.

Nothing observed after `a` decides who is in the pool, so this is not lookahead
— but it is not an S&P 500 strategy either, and the breadth cost is steep. 172
names qualify at 2015-01-31 against 373 at 2025-12-31, and only 131 of the 2015
set are still applicable at the end. The 172 is thin for a mechanical reason:
21 complete quarters ending 2014-09-30 reach back to 2009, before XBRL coverage
was general (340 of 567 tickers have any accounting by then). A frozen pool
therefore makes a filing-coverage artifact the investable set for eleven years,
and excludes every index entrant after the freeze date. Report it as a
robustness check on a fixed opening cross-section, alongside the reconstituting
run rather than in place of it.

Changing `universe_asof` needs no data re-fetch — the screen is a pure function
of `accounting.csv` and `monthly_price.csv`, so `Data.run.run_applicable_ticker`
rebuilds `applicable_ticker.csv` from the panels already on disk.

## How long-only is imposed

There is no one way to do this, and the three models need three different
treatments. What they have in common is that the constraint is imposed *where
the weights are decided*, never by clipping a finished answer.

**EPO — constrained solve.** `EPO.portfolio_weight` maximises its own
mean-variance objective `w'mu - gamma/2 w'Sigma w` subject to `w >= 0` and
`sum(w) = 1`, by SLSQP on the shrunk covariance with an analytic gradient
(~0.1s for a 150-name date). Clipping the closed-form `Sigma^-1 @ signal / gamma`
instead would have been one line, but it discards the hedge the short leg was
funding and leaves a book that is close to just "long the positive-signal
names": the covariance stops being able to hold a name down for what it is
correlated with. The constrained solve keeps ~70-85 names of ~150, at an
effective N near 55.

**PPP — truncate inside the fit.** `w+ = max(w, 0) / sum(max(w, 0))`, which is
Brandt-Santa-Clara-Valkanov's own treatment (sec. 4), applied in `ppp_weight` —
which `ppp_op` also calls. That placement is the whole point: `theta` is fitted
on the policy that is actually traded. Fitting `theta` unconstrained and
truncating afterwards would optimise one policy and hold a different one. The
cost is a kink in the objective wherever a weight crosses zero, so `find_theta`
now checks convergence and warns.

**Proposed model — bounded optimiser.** `port_weight` already solved on `[0, 1]`
with an equality constraint; unchanged.

## Two decisions worth knowing about

**EPO's correlation shrinkage is the shrinkage, not its complement.**
`constant.epo_shrinkage` is the weight on the *identity* in
`Sigma_hat = vol @ ((1-s)C + sI) @ vol`. It was previously read as the weight on
`C`, so the stated 0.75 was applying a shrinkage of 0.25 — far too light for a
correlation matrix estimated on ~150 names from 261 daily observations, whose
condition number runs 7.5e3 to 4.0e4. A sweep of the whole backtest put
long-short net Sharpe at 0.06 (s=0.25), 0.28 (0.75), 0.33 (0.90) and 0.19
(0.99); it is now 0.90. **Results produced before this change are not
comparable to results produced after it.**

**The risk-free charge scales with net exposure.** All five strategies are now
fully invested, so `r - rf` is right for every one of them. The mechanism still
matters under `--long-short`, where EPO comes out close to market neutral: its
longs are funded by its shorts, not by capital, and charging it a full risk-free
rate would subtract a financing cost it never paid. `metrics.performance`
subtracts `net_exposure * rf`, which is correct in both cases and needs no
per-strategy flag.

## Is the Sharpe ratio gap real?

The summary table tests it, rather than leaving a ranking to be read as a
result. Each strategy's Sharpe ratio is compared with `equal_weight`'s by
Ledoit and Wolf (2008), *Robust performance hypothesis testing with the Sharpe
ratio* (JEF 15, 850–859) — `Instruction/RobustTestingSharpeRatio-2008-LedoitWolf.pdf`.
Four rows:

| row | meaning |
|---|---|
| `sr_diff` | annualised `SR_strategy − SR_equal_weight` |
| `sr_diff_se` | its HAC standard error |
| `sr_diff_pval_hac` | p-value from the prewhitened-QS HAC estimate |
| `sr_diff_pval_boot` | p-value from the studentized circular block bootstrap |

**Read `sr_diff_pval_boot`.** Both are two-sided tests of `H0: SR_x = SR_ew`.
The HAC one is asymptotically valid but liberal at this sample size — the
paper's simulations put it at 12.4% for a nominal 10% test on heavy-tailed
time-series data, and a rerun of that experiment in
`tests/test_sharpe_inference.py` reproduces the effect. The bootstrap held its
nominal level across every process the paper tried, and is the default.

The usual test in this literature (Jobson–Korkie, corrected by Memmel) is *not*
implemented, deliberately: it assumes i.i.d. bivariate normal returns, and the
paper's whole point is that this fails exactly where it matters. Its rejection
rate reaches 22.5% for a nominal 10% test on t₆-VAR data.

One implementation choice is worth knowing before quoting a number: the
bootstrap covariance is scaled by `1/l` rather than the `1/T` printed in the
paper's Section 3.2.2. The two agree only at block size 1, which is the case its
footnote 9 checks; `1/T` is short by a factor of `b` everywhere else. See the
top of `sharpe_inference.py`.

### Why the block size is fixed at 5 rather than calibrated

Algorithm 3.1 is implemented (`sharpe_inference.calibrate_block_size`, or
`--sr-block calibrate`) and it is cheap — about 4s at `K = 100`, 40s at the
`K = 1000` the paper asks for, against a backtest that takes ten hours. Cost is
not the reason it is off by default.

The reason is that **it does not identify a block size on these series.** At
`K = 1000` the estimated coverage over the paper's own grid `{1, 2, 4, 6, 8, 10}`
spans 0.932 to 0.939 — a spread of 0.007 — while each point carries a Monte
Carlo standard error of `√(0.94·0.06/1000)` ≈ 0.0075. The differences between
block sizes are smaller than the error in measuring them, so a plain argmin
returns a different answer on every seed (4, 1, 2, 1 over four of them).
`calibrate_block_size` therefore treats every candidate within one standard
error of the best as tied and takes the middle of that set, which keeps the pick
off the endpoints; the fixed default of 5 lands in the same region, and is close
to the `b = 4` and `b = 6` the paper's two applications calibrated to at a
comparable `T`.

It makes little difference either way: across the whole grid the p-values here
move by less than 0.02. The one case worth naming is `proposed_forward`, which
goes from 0.111 at `b = 5` to 0.098 at `b = 1` — i.e. it straddles 10%, which is
a reason to distrust that threshold on this data rather than to prefer a block
size.

`--sr-benchmark none` skips the test; `--n-boot 0` leaves only the HAC p-value.

## Modules

- `data.py` — the universe, the realised monthly return panel, the risk-free rate.
- `sharpe_inference.py` — the Ledoit-Wolf test above, and the block size
  calibration.
- `strategy.py` — adapters putting all three models on one `(date, tickers) ->
  pd.Series` signature, plus `normalise`.
- `engine.py` — the backtest loop and `BacktestResult`.
- `metrics.py` — performance statistics, including the CRRA certainty equivalent
  at `constant.risk_aversion`, which is the objective PPP and the proposed model
  are actually fitted on.
- `run.py` — configures each model inside its own worker process and fans the
  strategies out over a pool.

## What the table reports

Eight rows, `metrics.SUMMARY_ROWS`:

| row | note |
|---|---|
| `ann_return` | geometric, after cost |
| `ann_vol` | |
| `max_drawdown` | |
| `sharpe` | annualised, on excess returns |
| `sharpe_pval` | bootstrap p-value against `equal_weight` — see above |
| `ann_crra_ce` | at `constant.risk_aversion`, the objective two of the models fit |
| `ann_turnover` | |
| `avg_weight_entropy` | breadth, in nats |

That is the whole table, and the whole computation — Sortino, Calmar, hit rate,
skew, kurtosis, best and worst month, the exposure averages and the holding
counts are gone, not hidden. Nothing is lost that a run cannot show another
way: the per-month series they summarised are still in each result's
`diagnostics` (and in `diagnostics_<strategy>.csv`), and formation dates whose
rule raised are still in `failures` and in `failures.csv`.

**Entropy is the breadth measure.** `-Σ p ln p` on `p = |w| / gross`, averaged
over months. An equal-weighted book of N names scores `ln N`, so `exp(entropy)`
reads as a name count: 5.63 for `equal_weight` is ~279 names, 4.34 for `epo` is
~77. It replaces `avg_n_holdings`, `avg_effective_n` and `avg_max_weight`, which
were three answers to one question — and unlike the inverse Herfindahl, which is
driven by the largest positions, entropy also notices whether the rest of the
capital is spread or itself clustered. Taken on absolute weights over gross so
it stays defined under `--long-short`, where a signed weight is not a
probability; it then measures where risk is spread, not net position.

`n_failed_date` is no longer reported. It is a correctness caveat rather than a
metric — currently 0 for every strategy — so after any run whose universe or
data changed, check for `failures.csv`, which `run.save` writes only when a
formation date's weighting rule raised.

## Output

`summary.csv`, `monthly_returns.csv`, `cumulative_wealth.csv`,
`diagnostics_<strategy>.csv`, `weights_<strategy>.csv`, and `failures.csv` when
a formation date's weighting rule raised.
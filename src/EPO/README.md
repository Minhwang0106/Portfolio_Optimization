# EPO — Enhanced Portfolio Optimization

Pedersen, Babu & Levine (2021). Mean-variance portfolio choice,
`max w'mu - gamma/2 w'Sigma w`, with the correlation matrix shrunk toward the
identity before `Sigma` is built — the fix for the fact that a ~150-name
correlation matrix estimated from 261 daily returns is mostly noise, and an
unshrunk inverse turns that noise into 350x gross exposure. One of the two
baselines `Proposed_Model` is benchmarked against; see the top-level
[README](../../README.md) for how the three models relate.

## Reproduce standalone

```python
from src.EPO.model import EPO

EPO.Config(n_workers=5)                    # sweeps every date in constant.testing_period
w = EPO('2020-03-31').portfolio_weight()   # long-only, sums to 1
```

`Config` is a one-time, class-level precomputation — unlike `PPP` and
`Proposed_Model.RIM_PortOp`, `EPO.__init__` does **not** call it for you, since
triggering a full parallel sweep over every formation date as a side effect of
constructing one object would be a surprise. Call `Config()` once, then
instantiate `EPO(date)` per date as needed. It's idempotent; pass `force=True`
to rebuild after changing `com`/`correl_com`/`n_day`/`n_month`.

In practice this is driven by the backtest harness rather than run standalone:

```bash
python -m src.Empirical_Analysis.run --only epo --epo-workers 5
```

## Performance

Fast. `Config`'s own inner process pool computes vol/correlation/TSMOM per date
in parallel (`n_workers`, defaults to `cpu_count - 1`); the per-date solve
itself is SLSQP with an **analytic gradient** supplied (`_mean_variance_jac`),
~0.1s for a ~150-name date. This is the one model where a hand-derived gradient
was worth writing — see `Proposed_Model/README.md`'s Performance section for
what happens when a comparably-sized optimization is left to finite differences.

## Key parameters (`constant.py`)

| Constant | Role |
|---|---|
| `risk_com`, `risk_correl_com` | EWM center-of-mass for volatility / correlation |
| `risk_n_day` | Trailing daily window for vol and correlation |
| `risk_n_month` | Trailing window (months) for the TSMOM signal |
| `epo_theta` | First-stage shrinkage of the raw correlation matrix (kept at 1.0 = no-op; the two-stage form exists because it's the paper's) |
| `epo_shrinkage`, `epo_shrinkage_grid`, `epo_w_min_periods` | The identity-shrinkage weight — **read `epo_shrinkage`'s comment in `constant.py` before changing it**, it's the weight on the identity, not on the correlation matrix, and getting the direction backwards silently reintroduces the noise-amplification problem it exists to fix |

`_select_w` picks the shrinkage per date out-of-sample, by realised Sharpe over
every *earlier* formation date's grid candidates; a date's own return never
scores its own pick. See [`Empirical_Analysis/README.md`](../Empirical_Analysis/README.md#two-decisions-worth-knowing-about)
for the incident that motivated pinning this down.

## Differences from Pedersen, Babu & Levine (2021)

`Instruction/epo-implementation-guide.md` is the from-scratch implementation
guide this was built against; the deviations below are what separates
`src/EPO` from a literal reading of it.

| | Paper | Here |
|---|---|---|
| **Signal** | Recommends **XSMOM** (eq. 18) for equity portfolios — cross-sectional outperformance vs. the universe mean | Implements the **Global-asset TSMOM** signal instead (eq. 1a): `sign(12-month compounded excess return) × annualized vol × 0.1`, i.e. time-series sign momentum, not cross-sectional. See `compute_date`'s `tsmom` line. |
| **Risk model** | Two recipes: "Global" (EWMA vol/correlation, used for macro-asset universes) vs. a dedicated equity recipe (60-month equal-weighted covariance, 5% off-diagonal shrink) | Uses the **Global** recipe (`risk_com=60`, `risk_correl_com=150` on 3-day overlapping returns, `risk_n_day=261`) on an equity universe — not the paper's own equity-sample recipe. `epo_theta` exists to carry a first-stage shrink like the equity recipe's 5%, but is currently `1.0`, a no-op. |
| **Sign constraint** | Simple EPO (eq. 16) is unconstrained; no `w >= 0` anywhere in the paper | Defaults to `long_only=True`, a bounded SLSQP solve — needed so EPO's return series is comparable to `PPP` and `Proposed_Model`, both of which are also long-only by default here. `long_only=False` recovers the paper's exact closed form. See [`Empirical_Analysis/README.md`](../Empirical_Analysis/README.md#how-long-only-is-imposed). |
| **w selection** | Expanding-window OOS grid search on the *unconstrained* closed form (Figure 2's algorithm) | Same algorithm, same scoring on the unconstrained closed form (`EPO._select_w`) — but the book actually held under `long_only=True` is the *constrained* solve at whatever `w` that search picked, not the form the search scored. Rationale in `constant.epo_shrinkage`'s comment. |
| **gamma** | Notes Simple EPO's Sharpe ratio is invariant to `gamma` (linear in `1/gamma`); the paper calibrates `gamma_t = 4*n_t` only to match TSMOM's own scale | Fixed at `constant.risk_aversion = 5.0`, shared across all three models so their CRRA certainty-equivalents are comparable, not chosen for a TSMOM match. |

None of these make the EPO baseline "wrong" — they're what it takes to run EPO
as a fair baseline against `PPP` and `Proposed_Model` on the same restricted,
long-only, single-gamma universe rather than reproduce the paper's own global
multi-asset backtest. State them explicitly if quoting EPO's numbers against
the paper's.

## Long-only vs. long-short

`portfolio_weight(long_only=True)` (default) solves the constrained problem
above under `w >= 0, sum(w) = 1`. `long_only=False` returns the closed-form
unconstrained maximiser `Sigma^-1 @ signal / gamma`, whose leverage is whatever
the covariance implies — generally close to market neutral, and not comparable
to the long-only return series. See `Empirical_Analysis/README.md`'s "How
long-only is imposed" for why clipping the closed form instead would discard
the hedge the short leg was funding.

## Modules

- `model.py` — the `EPO` class: `Config` (the parallel per-date sweep and
  shrinkage selection) and `portfolio_weight` (the per-date solve).
- `utils.py` — `init_worker`/`compute_date`, the per-process-pool-worker unit:
  builds vol, correlation, and the TSMOM signal for one date, plus the
  unconstrained candidate weight at every point on `epo_shrinkage_grid`.
- `excess_return.py` — return/excess-return helpers feeding the signal.

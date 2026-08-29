# EPO — Enhanced Portfolio Optimization

Pedersen, Babu & Levine (2021), the paper's **Equity 4** specification. Built
against `Instruction/epo_equity4_implementation.md`; the paper is the authority
for everything here and is cited by page.

One thing to have straight first, because the obvious reading is wrong. EPO
looks like mean-variance with a shrunk covariance and is *not* that. §II.C poses
a **robust** max-min over an ellipsoidal uncertainty region for $\mu$ (eq. 11),
and Proposition 2 (p.13, proved p.36) solves it in closed form:

$$
\mathrm{EPO}_s(w) \;=\; \frac{1}{\gamma}\,\Sigma_w^{-1}s,
\qquad
\Sigma_w \;=\; \sigma\bigl[(1-w)\,\tilde\Omega + w I\bigr]\sigma
$$

— equations (16) and (15). So $\Sigma_w$ is neither a risk model nor a better
covariance estimate; it is what the algebra of the robust objective leaves
behind, and $w$ is a statement about noise in $\mu$, not in $\Omega$. That
problem carries no sign or budget constraint, which is why the unconstrained
run — `long_only=False` — is the one to quote against the paper.

## Configuration

At each month end $t$, over the tickers applicable then:

1. **Universe.** Drop any name without a complete daily window, a complete
   12-month window, and a return in $t+1$. One NaN column makes `DataFrame.cov`
   pairwise-complete, which need not be positive semi-definite.
2. **Covariance.** Equal-weighted over the trailing `cov_window(n_t)` daily
   returns, `ddof=1` — the $1/(K-1)$ of fn.17 (p.20) — annualised by
   `trading_day_per_year`, then split into $\sigma$ and $\Omega$.
3. **Pre-shrink.** $\tilde\Omega = 0.95\,\Omega + 0.05\,I$ (`epo_theta`). This is
   the *risk model's* 5%, applied before EPO sees the matrix. Not the EPO
   shrinkage.
4. **Signal.** $s = \sigma \cdot \mathrm{XSMOM}$, XSMOM being the 12-month
   cumulative excess return demeaned cross-sectionally and scaled so the
   positive leg sums to $+1$ (eq. 18). No skip-month, per the paper.
5. **Solve.** At every $w$ on `epo_shrinkage_grid`, in whichever of the two
   forms `Config(long_only=...)` asked for: the maximiser over
   $x \ge 0,\ \mathbf{1}^{\top}x = 1$ by SLSQP (the default), or the closed
   form $x = \Sigma_w^{-1}s/\gamma$.
6. **Select $w$ out of sample.** Expanding-window argmax of realised Sharpe over
   strictly *earlier* formation dates (`_select_w`); a date's own return never
   scores its own pick, and the first `epo_w_min_periods` months fall back to
   `epo_shrinkage`.

**One book at a time.** Step 5 builds the constrained book or the unconstrained
one, never both, and step 6 ranks $w$ against whichever was built. That is not
an optimisation — the two are close to uncorrelated (below), so a $w$ scored on
one is nearly uninformative about the other, and a class holding both could not
say which EPO its `chosen_w` belongs to. The long-only default puts EPO on the
same footing as `ppp` and `Proposed_Model` and follows the same run-level
`long_only` they do; `--long-short` runs all four as their papers state them,
which for EPO is the form to quote against Pedersen, Babu & Levine.

## Differences from Pedersen, Babu & Levine (2021)

| | Paper | Here |
|---|---|---|
| **Test assets** | 49 value-weighted industry portfolios, 1927–2018 | Point-in-time S&P 500 constituents, 2015–2025, 172–381 names. This is the deviation the others follow from, and why the paper's Table 5/6 targets (SR 0.96→0.99, alpha 6.25%) are **not** reproducible here |
| **Covariance window** | 120 days flat | $\max(120,\,2n_t)$ — 344 days in 2015, 762 in 2025. A flat 120 is rank-deficient on a universe this wide |
| **Signal** | $\mathrm{XSMOM}\times\sigma$ (eq. 18) | Same, literally. $c_t$ normalises the cross-section, so on 381 names a typical $\lvert s_i \rvert$ is ~0.0015 against ~0.02 on the paper's 49 — identical in shape, much smaller in level |
| **Risk model** | 120-day equal-weighted covariance, correlations pre-shrunk 5% | Same, modulo the window |
| **$\gamma$** | Notes Simple EPO's Sharpe is invariant to $\gamma$; calibrates $\gamma_t = 4n_t$ only to match TSMOM's scale | Fixed at `risk_aversion = 5.0`, shared with `PPP` and `Proposed_Model` so the certainty-equivalents compare. Harmless unconstrained, **not** under the long-only default |
| **Sign constraint** | None anywhere; eq. (16) is the unconstrained solution to (11) | $x \ge 0,\ \mathbf{1}^{\top}x = 1$ by default, so the EPO row compares like-for-like against `ppp` and `Proposed_Model`. `long_only=False` restores the paper's form, and is the one to quote against it |
| **$w$ selection** | Expanding-window OOS grid search on the closed form (Fig. 2) | Same algorithm, same scoring, grid step 0.05, run against whichever book the run holds |

**What the long-only constraint costs.** The constrained solve is still the
robust objective: the two differ by a positive scalar and an additive constant,
so they share an argmax over any feasible set — checked against eq. (11) *as
written*, square root and ellipsoidal inner minimisation intact, matching
`_long_only_weight` to $10^{-6}$ on the simplex. What breaks is everything
around it. $w = 1$ no longer returns the anchor exactly (eq. 14) but its
projection; $\mathbf{1}^{\top}x = 1$ pins leverage, so $\gamma$ stops being free
and a market beta the derivation never contemplated is injected into a signal
that is demeaned and so near market-neutral. At the optimum the mean term
$x^{\top}s$ is only 4.5–17% of the variance penalty
$\tfrac{\gamma}{2}x^{\top}\Sigma_w x$. On three sampled dates the constrained
book correlates 0.95–0.99 with plain long-only *minimum variance* at the same
$\Sigma_w$ when $w$ is low (decaying to 0.10–0.51 at $w = 1$), and −0.04 to
+0.19 with unconstrained EPO at every $w$ below 0.75. Read the long-only book as
minimum variance with a faint momentum tilt.

That near-orthogonality is why $w$ is selected against the book actually held.
Ranking the closed form and handing the winner to the constrained solve — what
this did while the two ran as separate rows sharing one selection — was scoring
a portfolio the book is uncorrelated with. The two selections disagree sharply
over the full sample: unconstrained, $w$ concentrates on $[0.85, 0.95]$, 63 of
132 dates at 0.90; long-only it is bimodal and unstable, 53 dates at 0.00
against 52 at 1.00, flipping regime around 2021. Selecting on the right object
is correct, but on this universe it does not buy a stable parameter.

## Usage

```python
from src.EPO.model import EPO

EPO.Config(n_workers=5)                    # sweeps every date in constant.testing_period
w = EPO('2020-03-31').portfolio_weight()   # long-only, sums to 1

EPO.Config(long_only=False, n_workers=5)   # rebuilds against the closed form
x = EPO('2020-03-31').portfolio_weight()   # eq. (16), unconstrained
```

`portfolio_weight` takes no constraint argument, because the choice was already
made: `Config` built one candidate set and selected one `chosen_w` against it,
and the solve has to match both.

`Config` is a one-time, class-level precomputation — unlike `PPP` and
`RIM_PortOp`, `EPO.__init__` does **not** call it for you, since triggering a
full parallel sweep as a side effect of constructing one object would be a
surprise. It is idempotent *within a mode*: a repeat call in the same mode
returns at once, while one asking for the other book rebuilds rather than
handing back a config that answers a different question. Pass `force=True`
after changing `n_day`/`day_per_name`/`n_month`.

The long-only default costs one SLSQP solve per date per grid point — ~2,800
over the default sample, minutes rather than seconds. `long_only=False` is one
`np.linalg.solve` each instead, and runs in seconds.

Usually driven by the backtest harness:

```bash
python -m src.Empirical_Analysis.run --only epo --epo-workers 5
python -m src.Empirical_Analysis.run --only epo --long-short   # the paper's form
```

Each worker loads the full daily panel independently (~630 MiB peak, settling to
~165 MiB), so size `--epo-workers` against free memory rather than core count.

## Modules

- `model.py` — the `EPO` class: `Config` (the parallel per-date sweep and the
  shrinkage selection, and where `long_only` is fixed for the class),
  `_select_w` (agnostic about which book it ranks), and `portfolio_weight` (the
  per-date solve, in the configured book, at the $w$ chosen for it).
- `utils.py` — `init_worker`/`compute_date`, the per-worker unit: covariance,
  volatilities, correlations and signal for one date, plus the candidate weight
  at every point on `epo_shrinkage_grid`, in whichever book `long_only` names.
- `functions.py` — helpers: `cal_return`/`cal_excess_return` (raw returns for the
  risk model, excess for the signal and P&L), `cov_window`,
  `cross_sectional_momentum` (eq. 18), and the mean-variance objective, its
  analytic gradient, and `_long_only_weight` (SLSQP on the simplex).

## Key parameters (`constant.py`)

| Constant | Meaning |
|---|---|
| `risk_aversion` | $\gamma = 5.0$, shared with `PPP` and `Proposed_Model` so the CRRA certainty-equivalents compare. Irrelevant to the unconstrained book's Sharpe (scale-invariant), a real trade-off parameter for the long-only one |
| `risk_n_day`, `risk_day_per_name` | Daily covariance window, $\max(120,\,2n_t)$. A flat 120 — the paper's number, sized for its 49 industry portfolios — would leave rank $\le 119$ on 172–381 names at every date, with the 5% pre-shrink supplying the only information the inverse has along the null directions. 120 stays as the floor |
| `trading_day_per_year` | 252, annualising the daily covariance. **Not** a free units choice under $\mathbf{1}^{\top}x = 1$; see the constant's comment |
| `risk_n_month` | 12, the trailing window for the XSMOM signal |
| `epo_theta` | 0.95, the *risk model's* correlation pre-shrink. Applied before EPO — **not** the EPO shrinkage, and conflating the two is what makes the paper's $w = 0$ column score 0.96 rather than an unshrunk MVO's 0.84 |
| `epo_shrinkage_grid` | The $w$ values `_select_w` scores: 0 to 1, step 0.05, as in the paper |
| `epo_shrinkage`, `epo_w_min_periods` | Fallback $w$ (0.90) and the months of realised history required before selection runs (24). The fallback is inherited from a sweep under an older risk model — read the constant's comment before quoting it as evidence |

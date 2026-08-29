# PPP — Parametric Portfolio Policy

Brandt, Santa-Clara & Valkanov (2009), *RFS* 22(9):3411–3447. Built against
`Instruction/PPP_2009.md`, with the working notes and the already-fixed bugs in
`Instruction/PPP_note.md`; the paper is the authority for everything here.

One thing to have straight first, because the machinery looks like the rest of
this repo and is not. PPP forecasts nothing. There is no expected-return vector
and no covariance matrix anywhere in it — the weight itself is parameterised as
a linear function of firm characteristics,

$$
w_{i,t} \;=\; \bar w_{i,t} \;+\; \frac{1}{N_t}\,\theta^{\top}\hat x_{i,t}
$$

and the three numbers in $\theta$, shared by every stock at every date in the
window, are chosen to maximise the CRRA utility the policy would actually have
earned:

$$
\hat\theta_t \;=\; \arg\max_{\theta}\ \frac{1}{T}\sum_{s}
\frac{\bigl(1+\sum_i w_{i,s}(\theta)\,r_{i,s+1}\bigr)^{1-\gamma}}{1-\gamma}
$$

So the estimation problem stays three-dimensional however wide the universe
gets — the paper's whole point — and $\theta$ is a statement about how a
characteristic should be *traded*, not about what it predicts. `ppp` is a
**baseline for `Proposed_Model` to be measured against, not a reproduction
attempt**; the deviations below are consequences of the available data, not
defects.

## Configuration

At each month end $t$, over the tickers applicable then:

1. **Characteristics.** `mom`, the compounded return from $t-13$ to $t-2$,
   $P_{t-2}/P_{t-13}-1$, skipping the most recent month against short-term
   reversal; `btm` $= \log(1+BE/ME)$; `me` $= \log(ME)$, with $ME$ = close ×
   shares. Stored **raw** by `Config` — the z-score cannot be taken before the
   date's investable universe is known.
2. **Universe.** Drop any name without a complete record of *all three*
   characteristics at *every* date of the estimation window. Screening `btm`
   alone leaves behind the 903 rows whose `mom` is missing for want of a
   13-month lookback, which a valid `btm` does not imply.
3. **Standardize.** Z-score each characteristic across the $N_t$ survivors, per
   date, `ddof=1`. This is what makes $\sum_i \hat x_{i,t}=0$, hence
   $\sum_i w_{i,t}=1$ for any $\theta$. Never over the time series and never
   over the full panel: z-scoring all ~567 tickers while holding ~289 put the
   weight sum at −22.68.
4. **Benchmark.** $\bar w_{i,t} = 1/N_t$, equal weight over those survivors.
5. **Fit $\theta$.** 61 monthly cross-sections — characteristics over
   $[t-61\text{M},\,t-1\text{M}]$, each paired with the return it earns the
   following month, $[t-60\text{M},\,t]$. Strictly backward-looking; a date's
   own return never enters its own fit. `scipy.optimize.minimize` with no method
   given (BFGS on a numeric gradient), started at `np.random.random(3)`.
6. **Apply.** Standardize the cross-section observed *on* $t$, over those same
   survivors, and tilt the benchmark by it.

**The long-only constraint belongs in step 5, not after it.** Under
`long_only=True` (the default) the short leg is truncated and the remainder
renormalised, $w^{+} = \max(w,0)\big/\sum_i \max(w_i,0)$ — BSV's own
no-short-sale treatment, sec. 4 — applied *inside* the objective, so $\theta$ is
fitted on the policy that is actually traded. Fitting unconstrained and
truncating afterwards would optimise one policy and hold a different one. The
price is differentiability: truncation kinks the objective wherever a weight
crosses zero, so `find_theta` warns rather than raises when `minimize` reports
failure and returns the last iterate — the caller's alternative is no portfolio
at all for that date.

## Differences from Brandt, Santa-Clara & Valkanov (2009)

| | Paper | Here |
|---|---|---|
| **Universe** | ~full CRSP cross-section, thousands of names | Point-in-time S&P 500 constituents, 2015–2025, 172–381 names. Little size dispersion survives, so `me` cannot do what it does in the paper. This is the deviation the others follow from |
| **Benchmark $\bar w$** | Value-weighted market | $1/N_t$, equal weight. The policy is a *tilt away from* $\bar w$, so changing the anchor changes the answer — and it makes the `equal_weight` row elsewhere in this repo a measure of the tilt in isolation rather than a comparison against the paper's own benchmark |
| **Estimation window** | Full/rolling over a long sample | 61 cross-sections, `n_month = 60`. Three parameters, a short window, no penalty and no bound on $\theta$ — the main driver of the leverage below |
| **Book equity** | $BE = A - L + \mathrm{DT\&ITC} - PS$ (Compustat 6, 181, 35, 10/56/130) | `StockholdersEquity` net of minority interest and preferred stock. Liabilities and the deferred-tax add-back cannot be separated out of XBRL; a standard modern proxy, but not the paper's number |
| **Reporting lag** | ≥ 6 months from fiscal year end to formation | One filing row (`book_value.shift(1)`), i.e. ~1 quarter — shorter than the paper's, and the shift is positional rather than per ticker (`PPP_note.md` §4.2) |
| **Screens** | Drop $btm<0$; drop the bottom 20% by `me` | $BE>0$ is enforced upstream by `Data.exclusion` across the trailing `n_quarter`, so the first is satisfied by the universe rather than inside PPP. The size filter is not applied, and would be close to vacuous on S&P 500 names |
| **Constraints** | Reports no-short-sale variants alongside the unconstrained policy | `long_only=True` by default, which *is* the paper's no-short-sale variant; `long_only=False` restores the unconstrained one |
| **$\gamma$** | CRRA, reported at several levels of risk aversion | Fixed at `risk_aversion = 5.0`, shared with `EPO` and `Proposed_Model` so the certainty-equivalents compare |

**What the unconstrained policy does here.** `long_only=False` is the equation
at the top of this file, and on this universe it is not a book anyone could
hold. Over the 132 formation dates the gross exposure $\sum_i|w_i|$ has a median
of **9.9×**, a 10th–90th percentile range of 5.9–26.0 and a maximum of 32.9,
with a single name reaching 87% of capital at one date; $\lVert\theta\rVert$
runs from 6.5 to 40.0 and every coefficient changes sign somewhere in the
sample. Nothing bounds it — three parameters fitted on 61 cross-sections with no
penalty will take whatever position in-sample utility rewards, and a short
window makes that position large. The `long_only=True` default removes it by
construction, the truncated book being non-negative and summing to one, which is
also why the two return series are not comparable to each other and why
`--long-short` changes what `ppp` *is* rather than merely relaxing it.

Net of 10 bps, that default earns 10.3% a year at 18.3% volatility for a Sharpe
of 0.53, against equal weighting's 11.5%, 16.4% and 0.63, at 5.2× annualised
turnover against 0.85×. Read that as what the tilt costs on this universe, not
as a verdict on the paper.

## Usage

```python
from src.PPP.model import PPP

model = PPP()                                              # Config runs itself
w = model.portfolio_weight('2020-03-31', ticker_list)      # long-only, sums to 1
w = model.portfolio_weight('2020-03-31', ticker_list, long_only=False)
```

`Config` builds the class-level characteristic and return panels and is
idempotent. Unlike `EPO`, `PPP.__init__` calls it for you, so a bare `PPP()`
works standalone; call it explicitly only to `force=True` a rebuild after
`Data.run` rewrites a CSV — together with `panel.clear_panel_cache`, the parse
being cached too. `portfolio_weight` calls `find_theta` internally: the fitted
coefficients land in `model.theta[str(date)]` and the *narrowed* universe in
`model.ticker`, which is what the returned weights align to — not the `ticker`
argument that went in.

Usually driven by the backtest harness:

```bash
python -m src.Empirical_Analysis.run --only ppp
```

**Reproducibility, and what $\theta$ is worth reading.** `find_theta` seeds its
optimiser's start with `np.random.random(3)` off the *global* numpy state rather
than a `seed` argument, so an exact repeat needs `np.random.seed(...)` before
the call; the harness does this once per worker process. How much that matters
depends on the constraint. Unconstrained, three starts land within 0.02 of each
other on $\theta$ and $4\times10^{-4}$ on the weights — the window moves the
fit, not the start. Under the `long_only=True` default $\theta$ is **not
identified**: three starts finish up to 196 apart and $\lVert\theta\rVert$ has a
median of 240 against 13 unconstrained. The weights are stable anyway, to 0.02,
because once $\lVert\theta\rVert$ is large the $1/N_t$ benchmark term is
negligible beside the tilt and
$w^{+}\approx\max(\hat x^{\top}\theta,0)\big/\sum_i\max(\hat x_i^{\top}\theta,0)$
is homogeneous of degree zero in $\theta$ — scaling a fitted $\theta$ by 200×
moves the largest weight by $4\times10^{-4}$ and drops two names from the book.
Read the *portfolio* out of a long-only fit; do not read the coefficients.

## Modules

- `model.py` — the `PPP` class: `Config` (the characteristic and return panels),
  `_standardize` (the per-date cross-sectional z-score, stock axis always $-2$),
  `find_theta` (the universe screen, the date-major panel build and the CRRA
  fit) and `portfolio_weight` (fit, then apply to $t$'s own cross-section).
- `input_generator.py` — `generator()`: joins the quarterly accounting panel to
  the monthly price panel and forward-fills each book value across the months
  that follow it, yielding `book_value`, `market_cap` and `adj_close`. The same
  clean-surplus book equity `Proposed_Model.data_generator` uses.
- `utils.py` — `crra_utility` (shared with `Proposed_Model.utils.port_optimize`),
  `ppp_weight` (the policy, including the long-only truncation) and `ppp_op`
  (the negated objective `find_theta` minimises).

## Key parameters (`constant.py`)

| Constant | Meaning |
|---|---|
| `risk_aversion` | $\gamma = 5.0$ in the CRRA objective, shared with `EPO` and `Proposed_Model` so the certainty-equivalents compare. Not a free scale here — unlike a Sharpe ratio, it changes which policy wins |
| `ppp_estimation_month` | 60, the trailing window $\theta$ is fitted over, giving 61 monthly cross-sections. `PPP.find_theta` carries the same number as its own default; named here so the backtest and the model cannot drift apart silently |
| `n_month` | 75, and **not** the estimation window. It is `Data.exclusion`'s screen for how much price history a ticker needs: the 60 months of the fit plus ~12 for `mom`'s lookback. Do not unify the two |
| `n_quarter` | 20 quarters of complete accounting data for that same screen, and the window the $BE>0$ test runs over |
| `n_cum_month` | 12, filed under PPP in `constant.py` and **imported nowhere**. The lookback is hardcoded as `shift(2)/shift(13)`; the constant survives only as a cross-reference to `PPP_2009.md` |

See `Instruction/PPP_2009.md` for the characteristic definitions and
`Instruction/PPP_note.md` §3 for the bugs already fixed — several of them
produced weights that still summed to exactly one.

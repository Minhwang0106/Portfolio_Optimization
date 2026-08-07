# PPP — Parametric Portfolio Policy

Brandt, Santa-Clara & Valkanov (2009). Rather than forecasting each stock's
moments, the portfolio weight is parameterised directly as a linear function of
standardized firm characteristics:

```
w_it = w_bar_it + (1/N_t) * theta' * x_hat_it
```

with `w_bar` an equal-weighted benchmark and `x_hat` the cross-sectionally
standardized `mom` (11-month momentum), `btm` (log book-to-market), `me` (log
size). `theta` — three numbers, shared across every stock at a date — is fit
once per formation date by maximising realised CRRA utility over a trailing
window. One of the two baselines `Proposed_Model` is benchmarked against; see
the top-level [README](../../README.md) for how the three models relate.

## Reproduce standalone

```python
from src.PPP.model import PPP

PPP.Config()                     # builds the class-level characteristic + return panels
model = PPP()
w = model.portfolio_weight('2020-03-31', ticker_list, n_month=60, long_only=True)
```

`Config` is idempotent and `PPP()` calls it for you, so a bare `PPP()` works
standalone; call `Config()` explicitly only to `force=True` a rebuild after the
underlying panels change. `portfolio_weight` calls `find_theta` internally —
the fitted coefficients and the surviving universe end up on
`model.theta[str(date)]` and `model.ticker` respectively.

In practice this is driven by the backtest harness:

```bash
python -m src.Empirical_Analysis.run --only ppp
```

**Reproducibility note:** `find_theta` seeds its optimiser's starting guess with
`np.random.random(3)` off the *global* numpy state, not a `seed` argument — three
independent starts agree to within 0.01 (the window dominates, not the start),
but an exact repeat needs `np.random.seed(...)` set before the call. The
backtest harness (`Empirical_Analysis.run`) does this once per worker process.

## Key parameters (`constant.py`)

| Constant | Role |
|---|---|
| `ppp_estimation_month` | Default `n_month` — the trailing window `theta` is fit over. Named separately from `PPP.find_theta`'s own default so the backtest and the model can't drift apart silently. |
| `n_cum_month` | Unused by `mom` directly here — the 11-month lookback is hardcoded (`shift(2)/shift(13)`); kept for reference against `PPP_2009.md`. |
| `n_quarter`, `n_month` | Shared with the universe screen — see `Data/README.md`. |
| `risk_aversion` | The CRRA gamma `ppp_op` maximises utility under. |

## Differences from Brandt, Santa-Clara & Valkanov (2009)

`Instruction/PPP_note.md` has the full working notes — reproduced here because
it's the single most important thing to know before quoting a PPP number. **This
is a baseline for comparison against `Proposed_Model`, not a reproduction
attempt**, and without stating these explicitly the results read as a failed
reproduction rather than as unconstrained PPP on a restricted universe:

| | Paper | Here | Consequence |
|---|---|---|---|
| Universe | ~full CRSP cross-section, thousands of names | 187-382 S&P constituents | Little size dispersion survives, so `me` can't do what it does in the paper |
| Benchmark `w̄` | Value-weighted market | `np.ones(n_t)/n_t`, equal-weight | The policy is a *tilt away from* `w̄` — changing the anchor changes the result, and the equal-weight comparison elsewhere in this repo measures the tilt in isolation rather than against the paper's own benchmark |
| Estimation window | Full/rolling, long sample | 61 months rolling (`n_month=60`) | Main driver of the paper-unconstrained variant's ~11x gross leverage: three parameters, a short window, no bound on theta |
| Constraints | Reports no-short-sale variants alongside the unconstrained policy | `long_only=True` by default here, matching the harness's other two models | See Long-only below |

The cheapest way to move closer to a literal reproduction is already the
default here (`long_only=True`, the paper's own no-short-sale variant) — it's
the *unconstrained* (`long_only=False`) numbers that need the caveats above
before they're quoted against the paper.

## Long-only

`long_only=True` (default) truncates the short leg and renormalises,
`w+ = max(w, 0) / sum(max(w, 0))` — Brandt-Santa-Clara-Valkanov's own treatment
(sec. 4) — applied **inside** `ppp_op`, so `theta` is fit on the policy that is
actually traded rather than fit unconstrained and truncated after the fact.
`long_only=False` restores the paper's unconstrained policy, which runs to
~3.6x gross exposure on this universe and is not comparable to the long-only
return series. `find_theta` warns if `minimize` fails to converge (the
truncation puts a kink in the objective wherever a weight crosses zero) — the
theta is still returned rather than discarded, since the alternative is no
portfolio at all for that date.

## Modules

- `model.py` — the `PPP` class: `Config` (characteristic + return panels),
  `find_theta` (per-date CRRA fit), `portfolio_weight`.
- `input_generator.py` — `generator()`: builds the raw `mom`/`btm`/`me` panel
  from the price and accounting data (book value net of minority interest and
  preferred stock, the same clean-surplus definition
  `Proposed_Model.data_generator` uses).
- `utils.py` — `crra_utility` (shared with `Proposed_Model.utils.port_optimize`),
  `ppp_op` (the objective `find_theta` minimises), `ppp_weight` (applies a
  fitted `theta` to one cross-section).

See `Instruction/PPP_2009.md` and `Instruction/PPP_note.md` for the paper-level
derivation this implements.

# Proposed_Model — Residual-Income Portfolio Optimization

The paper's own model, `RIM_PortOp`: simulate each firm's forward accounting
characteristics from a fitted copula, map every simulated path to an implied
return through a residual-income valuation, then choose portfolio weights that
maximise expected CRRA utility over the resulting cross-section of return
distributions. See the top-level [README](../../README.md) for how this
compares against the `EPO` and `PPP` baselines.

## Reproduce standalone

```python
from src.Proposed_Model.model import RIM_PortOp

model = RIM_PortOp('2020-03-31', ticker_list)   # Config() runs automatically
w = model.weight(n_samples=10000, n_lags=4, ex_post=False,
                  common_theta=True, seed=0, long_only=True)
```

`RIM_PortOp.Config()` builds the class-level panels (`generator()`'s output)
once; `__init__` calls it for you, so a bare `RIM_PortOp(date, ticker)` works
standalone. Instantiate fresh per formation date — the universe narrowing in
`__init__` (`take_training_data`, then the `min_char_obs` completeness screen)
is per-instance, not shared.

`ex_post=True` elicits each characteristic's unconditional moments from the
*realised future* window instead of the training window — this is
`proposed_ex_post` in the backtest, a lookahead diagnostic that decomposes the
model's error into simulation machinery vs. moment forecasting, **never a
strategy to trade or quote on its own.**

In practice this is driven by the backtest harness:

```bash
python -m src.Empirical_Analysis.run --only proposed_historical proposed_ex_post
```

which rebalances quarterly by default (`quarter_ends`, since the accounting
inputs only move once a quarter) and varies the seed per formation date
(`_dated_seed`) so the simulation noise isn't correlated across the backtest.

## Performance — read this before running a full backtest

**This is the slow one**, and it's why `Empirical_Analysis/README.md` notes the
full backtest takes about ten hours. Profiling one real formation date
(2020-09-30, 272 usable tickers) at the production defaults
(`n_samples=10000`, `constant.n_quarter_ahead=80`) gives **591s (~9.9 min) for
one `.weight()` call**, and there are 44 quarterly formation dates times two
variants (`proposed_historical`, `proposed_ex_post`) in a full run.

Where the time goes, by internal (`tottime`) cost:

| Step | Share | Scales with |
|---|---|---|
| Vine copula conditional draws (`Bicop.hinv1`/`hinv2` inside `simulation._step`) | **56%** | `n_edges × n_quarter_ahead × n_samples` |
| Inverse-normal transform in `sample_conditional` (mostly `scipy.stats.norm.ppf` call overhead) | 17% | `n_tickers × n_quarter_ahead × n_samples` |
| RIM valuation arithmetic (`implied_return.rim_mapping`) | 8% | `n_tickers × n_quarter_ahead × n_samples` |
| Per-ticker/per-quarter signal loop (`sampling_distribution.signal_cal` etc.) | 7% | `n_tickers × n_quarter_ahead` (Python-loop overhead) |
| Pooled decay fit (`theta.find_theta`) | 1% | `n_tickers` only — independent of `n_samples` |
| Copula fitting (`dependence_structure.copula_structure`) | 1% | `n_tickers` only |
| Portfolio optimizer (SLSQP, `port_optimize.port_weight`) | <1% | negligible — despite ~270 decision variables |

The dominant cost is **not** the portfolio optimization (a reasonable first
guess, given ~270 SLSQP decision variables — but it converges in well under a
second here). It's that `constant.n_quarter_ahead = 80` — an 80-quarter,
20-year simulation horizon — multiplies every per-ticker, per-sample operation
80x, for every one of the ~1,087 copula edges (the cross-sectional tree over
tickers plus each ticker's own 4-characteristic inner tree), at every
formation date, for both `proposed_historical` and `proposed_ex_post`.

**To iterate quickly**, drop `n_samples` (Monte Carlo standard error scales as
`1/sqrt(n_samples)`, so this is a pure precision/speed trade) and/or run a
narrow date range or `--only proposed_historical` while developing:

```python
w = model.weight(n_samples=1000, ...)   # ~10x faster than the n_samples=10000 default
```

`n_quarter_ahead` is a modelling choice about the valuation horizon, not just a
speed knob — changing it changes how far out cash flows are simulated before
the Gordon-growth terminal value (`implied_return.terminal_val`) takes over, so
treat it as a methodology decision, not a default to tune for wall-clock time.

## Pipeline

1. **`data_generator.generator()`** — builds the characteristic panel
   (`ros`, `ato`, `ate`, `g`, three of them transformed onto the real line by
   `utils.variable_tranformation` so the Gaussian conditional law in step 3
   doesn't put mass where the ratio can't go) and the four RIM inputs
   (`pb`, `bvps`, `rps`, `close`).
2. **`RIM_PortOp.depedence_structure()`** — fits one vine-copula tree across
   tickers (on `g` alone, `utils.dependence_structure.copula_structure`) and
   one per-ticker tree across the four characteristics.
3. **`RIM_PortOp.sampling()`** — draws `n_samples` paths per ticker,
   `n_quarter_ahead` quarters deep (`utils.simulation.sample_joint`), then
   carries each draw through the persistence model of eqs (10)/(17)/(18)
   (`utils.sampling_distribution`, decay fit in `utils.theta.find_theta`) to
   put it back on the characteristic's natural scale.
4. **`RIM_PortOp.joint_return()`** — maps each simulated path to an annualised
   implied return via the residual-income valuation
   (`utils.implied_return.rim_mapping`, `.terminal_val`, `.implied_return`),
   using cost of funds from `utils.cost_of_fund` (Fama-French 5-factor).
5. **`RIM_PortOp.weight()`** — `utils.port_optimize.port_weight`: SLSQP over
   the simulated return cross-section, maximising expected CRRA utility
   (`constant.risk_aversion`), bounded to `[0, 1]` and summing to 1 under
   `long_only=True`.

## Key parameters (`constant.py`)

| Constant | Role |
|---|---|
| `n_quarter_ahead` | Simulation horizon, in quarters — **the dominant cost driver**, see Performance above |
| `min_char_obs` | Quarters of complete characteristics a ticker needs before it's fitted (12 of the `n_quarter`=20 window) |
| `reinvest_rate_cap` | Caps the per-quarter ROE residual income compounds forward at — a tourniquet against simulation artefacts in the unbounded `ate` tail (see the constant's own comment for the 1e254 case this prevents) |
| `gordon_den_floor` | Floors the Gordon perpetuity's denominator, so a simulated growth path that outruns the discount rate doesn't flip the terminal value's sign |
| `risk_aversion` | CRRA gamma for the portfolio optimization |

## Modules

- `model.py` — `RIM_PortOp`: `Config`, `depedence_structure`, `sampling`,
  `joint_return`, `weight`.
- `data_generator.py` — `generator()`, `take_training_data`, `accounting_cutoff`
  (the publication-lag logic — see its docstring for why the accounting window
  is cut one quarter before the market-data window).
- `utils/dependence_structure.py` — Kendall-tau ordering and per-edge copula
  fitting (`copula_structure`).
- `utils/simulation.py` — sampling from a fitted tree (`sample_tree`,
  `sample_joint`); `_step` is the hot path, see Performance.
- `utils/sampling_distribution.py` — the persistence model: `signal_cal` (eq
  10), `conditional_moments`/`conditional_mean` (eqs 17/18), `sample_conditional`.
- `utils/theta.py` — `find_theta`: fits the decay in eq (10) by grid + golden
  section search, pooled across the universe by default (`common_theta`).
- `utils/variable_tranformation.py` — the real-line transforms for `ros`,
  `ato`, `ate` and their inverses.
- `utils/elicitation.py` — `parameter_derived`: quantile-based mean/variance
  elicitation, robust to the outliers a raw sample moment would be sensitive to.
- `utils/implied_return.py` — `rim_mapping`, `terminal_val`, `implied_return`:
  the residual-income valuation itself.
- `utils/cost_of_fund.py` — Fama-French 5-factor cost of capital per ticker.
- `utils/port_optimize.py` — `port_weight`: the CRRA-utility portfolio solve.

See `Instruction/data-collection-rim.md` for the accounting-tag mapping behind
the characteristic panel, and `Instruction/CRRA_CE_Inference_Obsidian.md` for
the certainty-equivalent metric `Empirical_Analysis.backtest.metrics` reports alongside
the Sharpe ratio.

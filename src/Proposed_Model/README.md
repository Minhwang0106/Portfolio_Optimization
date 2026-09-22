# Proposed_Model

The paper's own model, `RIM_PortOp`. For each firm it simulates forward
accounting characteristics from a fitted vine copula, maps every simulated path
to an implied return through a residual-income valuation, and chooses the
portfolio weights that maximise expected CRRA utility over the resulting
cross-section of return distributions.

## How to run

Through the backtest harness:

```bash
python -m src.Empirical_Analysis.run --only proposed_historical proposed_ex_post
```

Or on its own, for one formation date:

```python
from src.Proposed_Model.model import RIM_PortOp

model = RIM_PortOp('2020-03-31', ticker_list)
w = model.weight(n_samples=10000, ex_post=False, seed=0, long_only=True)
```

- **It is slow.** One formation date takes about ten minutes at the default
  `n_samples=10000`, so a full backtest runs for hours. Lower `n_samples` (or
  `--n-samples` on the command line) to iterate faster.
- **`ex_post=True` looks ahead.** It takes each characteristic's moments from
  the realised future window, so `proposed_ex_post` is a diagnostic, not a
  strategy to quote.

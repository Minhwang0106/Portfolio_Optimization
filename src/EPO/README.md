# EPO — Enhanced Portfolio Optimization

Pedersen, Babu & Levine (2021), the paper's Equity 4 specification: a robust
mean-variance portfolio whose closed form, $\Sigma_w^{-1}s/\gamma$, shrinks the
correlation matrix toward the identity by $w$. The signal is 12-month
cross-sectional momentum times volatility, the covariance is equal-weighted over
a trailing daily window, and $w$ is chosen out of sample at each date. Here it
is a baseline for the proposed model, long-only by default so that it compares
with the other strategies.

## How to run

Through the backtest harness:

```bash
python -m src.Empirical_Analysis.run --only epo --epo-workers 5
python -m src.Empirical_Analysis.run --only epo --long-short   # unconstrained, as in the paper
```

Each worker holds the full daily panel (~630 MiB at peak), so size
`--epo-workers` to free memory, not core count.

Or on its own. Unlike the other models, `EPO.Config` must be called first:

```python
from src.EPO.model import EPO

EPO.Config(n_workers=5)                    # long-only; long_only=False for the paper's form
w = EPO('2020-03-31').portfolio_weight()
```

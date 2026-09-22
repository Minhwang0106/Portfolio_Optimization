# PPP — Parametric Portfolio Policy

Brandt, Santa-Clara & Valkanov (2009), *RFS* 22(9): 3411–3447. PPP forecasts
nothing: each stock's weight is equal weight tilted by a linear function of its
characteristics (momentum, book-to-market, size), and the three tilt
coefficients are fitted to maximise the CRRA utility the policy would have
earned over the trailing 60 months. Here it is a baseline for the proposed
model, long-only by default.

## How to run

Through the backtest harness:

```bash
python -m src.Empirical_Analysis.run --only ppp
```

Or on its own, for one formation date:

```python
from src.PPP.model import PPP

model = PPP()
w = model.portfolio_weight('2020-03-31', ticker_list)                   # long-only
w = model.portfolio_weight('2020-03-31', ticker_list, long_only=False)  # as in the paper
```

The weights come back on the names that survive PPP's own completeness screen
(`model.ticker`), which can be fewer than the tickers passed in.

# ICC_MVO

Bielstein & Hanauer (2019), *Mean-variance optimization using forward-looking
return estimates* (Review of Quantitative Finance and Accounting 52, 815–840): a
maximum-Sharpe portfolio whose expected returns are the Gebhardt, Lee &
Swaminathan (2001) implied cost of capital plus rescaled momentum, with a
Ledoit–Wolf covariance and at most 5% in any name. It is the point-estimate
benchmark of the paper's ex post table (Table 1), which also runs it under
quadratic (mean-variance) utility.

Without analyst forecasts, it reads *realised* earnings in their place, so like
the proposed model's ex post rows it is a lookahead benchmark, not a tradeable
strategy.

## How to run

```bash
python -m src.Empirical_Analysis.run --only equal_weight icc_mvo_ex_post icc_mvo_ex_post_qu
```

It needs `industry.csv` and `industry_roe_pool.csv` in `Data/raw file/`, for the
industry ROE. A full `python -m src.Data.run` writes both; to rebuild just
these two:

```bash
python -m src.Data.industry --user-agent "Your Name you@example.com"
python -m src.Data.industry_pool --user-agent "Your Name you@example.com"
```

Or on its own, for one formation date:

```python
from src.ICC_MVO.model import Icc_Mvo

w = Icc_Mvo('2020-03-31', ticker_list).weight()                # max Sharpe
w = Icc_Mvo('2020-03-31', ticker_list).weight(objective='quadratic_utility')
```

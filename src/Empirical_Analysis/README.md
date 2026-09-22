# Empirical_Analysis

Backtests every model on the same universe, calendar and transaction costs, so
that a difference between strategies is a difference in the weighting rule and
nothing else. It tests whether each difference is statistically real (mean
return, Sharpe ratio and CRRA certainty equivalent, by HAC and block-bootstrap
tests), and builds the paper's three tables.

## How to run

**Backtest** — writes returns, weights and diagnostics to `Data/Result/raw_backtest/`:

```bash
python -m src.Empirical_Analysis.run --n-workers 5
python -m src.Empirical_Analysis.run --only equal_weight epo ppp   # skip the slow ones
```

The full run takes hours, almost all of it the proposed model. `--help` lists the
strategy labels `--only` and `--skip` take.

- `--rebuild` — recompute every metric from the saved weights, without running
  any model. Takes seconds; use it after changing a metric, or to merge
  separate `--only` runs, each of which rewrites `summary.csv`.
- `--long-short` — run EPO, PPP and the proposed model unconstrained, as their
  papers state them.
- `--dump-implied-return` — save the proposed model's simulations; the five
  re-solved `*_n` strategies read them back (`--replay-dir`).

**Tables** — reads `raw_backtest/`, writes CSV to `processed_backtest/` and
LaTeX to `latex/`:

```bash
python -m src.Empirical_Analysis.tables
python -m src.Empirical_Analysis.tables --n-sim 1000   # faster Table 3 Monte Carlo
```

`proposed_ex_post`, `icc_mvo_ex_post` and every strategy built on them look
ahead at realised data: they are diagnostics, not strategies to quote.

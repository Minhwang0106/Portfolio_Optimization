# Portfolio Optimization

Code and data pipeline for a residual-income-model (RIM) portfolio construction method,
benchmarked against Enhanced Portfolio Optimization (EPO) and the Parametric Portfolio
Policy (PPP) on a point-in-time S&P 500 universe. Written to accompany the paper
*[paper title]* — see [Citation](#citation).

## What's here

Three portfolio-weighting rules are backtested on the same universe and calendar, so a
performance difference between them is a difference in the weighting rule and nothing else:

| Model | Source | Idea |
|---|---|---|
| `src/Proposed_Model` | this paper | Simulates each firm's forward accounting characteristics from a fitted copula, maps each simulated path to an implied return via a residual-income valuation, and optimises a CRRA-utility portfolio over the resulting return distribution |
| `src/EPO` | Pedersen, Babu & Levine (2021) | Enhanced Portfolio Optimization — the closed-form solution to a robust max-min problem over an uncertainty region for expected returns, which reduces to mean-variance against a correlation matrix shrunk toward the identity. The paper's Equity 4 specification: XSMOM × σ signal, equal-weighted daily covariance, 5% pre-shrink |
| `src/PPP` | Brandt, Santa-Clara & Valkanov (2009) | Parametric Portfolio Policy — portfolio weight parameterised directly as a linear function of firm characteristics (momentum, book-to-market, size) |

`equal_weight` is included as the zero-skill benchmark all three are compared against.

`src/Data` collects and cleans the raw panels every model reads: point-in-time S&P 500
membership, monthly adjusted prices, and quarterly XBRL accounting data. `src/Empirical_Analysis`
is the backtest harness — it fixes the universe, rebalance calendar, transaction costs, and
performance metrics so all four strategies are compared on equal footing, and tests whether
each strategy's Sharpe ratio differs from equal-weight's using the Ledoit-Wolf (2008)
studentized bootstrap. See [`src/Empirical_Analysis/README.md`](src/Empirical_Analysis/README.md)
for the full methodology: the exclusion rule, how each model's long-only constraint is imposed,
and what the output tables mean.

## Repository structure

```
constant.py                  All tunable parameters and file paths, in one place.
main.py                      Runs the whole pipeline end to end: data, backtest, tables.
src/
  Data/                      Fetches and cleans the raw panels (SEC EDGAR, Yahoo Finance,
                              S&P 500 constituent history) and builds the per-date universe.
  Proposed_Model/             The paper's residual-income portfolio model.
  EPO/                        Enhanced Portfolio Optimization baseline.
  PPP/                        Parametric Portfolio Policy baseline.
  Empirical_Analysis/         Backtest engine, performance metrics, Sharpe-ratio inference.
  panel.py                    Shared panel I/O (cached CSV reads, book-equity definition).
tests/                        pytest suite.
Data/raw file/                Collected data panels (prices, accounting, factors, universe).
                              Gitignored -- rebuild with `python -m src.Data.run`.
Data/Result/                  All generated output. Tracked, so the tables can be rebuilt
                              without re-running the data collection or the backtest.
  raw_backtest/               Everything a run produces: returns, weights, diagnostics.
  processed_backtest/         Only the numbers the paper reports, as CSV.
  latex/                      The same three tables as \input-able .tex.
```

## Setup

Requires Python >= 3.12. Dependencies are pinned in `uv.lock`.

```bash
uv sync
```

or, without `uv`:

```bash
pip install -e .
```

Either route puts `constant` and `src.*` on the path, so the commands below work from any
directory. This is a repository to clone and run rather than a library to depend on —
`constant` and `src` are generic top-level names.

`pyvinecopulib` (used by the copula fit in `Proposed_Model`) ships prebuilt wheels for the
common platforms; if it fails to install, check that your Python/OS combination has one
available.

**Windows, non-ASCII checkout path.** If the path you clone into contains non-ASCII
characters, set `PYTHONUTF8=1` — Python otherwise defaults stdout to cp1252 and any script
printing a path dies with `UnicodeEncodeError: 'charmap' codec can't encode character ...`.
`src/Data/` has a second, related caveat about the CA bundle; see
[`src/Data/README.md`](src/Data/README.md#prerequisites).

## Reproducing the results

**All at once:** `python main.py` runs the three steps below in order — data
collection, backtest, tables — and can resume from any point with
`--skip-data`/`--skip-backtest`/`--skip-tables`. See `python main.py --help`.
Because `Data/Result/` is tracked, `python main.py --skip-data --skip-backtest`
rebuilds every reported table from the saved run — neither the hours of collection
nor the ten-hour backtest is needed to check the numbers. Tables 1 and 2 take about
a second; Table 3's Monte Carlo takes roughly two minutes on top of that, or ~14
seconds with `--n-sim 1000`, and `--skip-experiment` leaves it out entirely.
The steps are broken out individually below for anyone who wants to run just
one of them, or needs a flag `main.py` doesn't expose.

**1. Collect the data.** Fetches prices from Yahoo Finance, accounting data from SEC EDGAR,
and rebuilds the point-in-time investable universe from the S&P 500 constituent history:

```bash
python -m src.Data.run --user-agent "Your Name you@example.com"
```

SEC identifies and throttles EDGAR callers by `User-Agent`, so this step needs a contact
string of your own — there is no shared default. At a terminal you are simply **prompted**
for one, so the flag is optional; supply it up front to skip the prompt, or set
`SEC_USER_AGENT` once per shell to skip it for good:

```bash
$env:SEC_USER_AGENT = "Your Name you@example.com"   # PowerShell
export SEC_USER_AGENT="Your Name you@example.com"   # macOS/Linux
```

`python main.py` takes the same string as `--sec-user-agent`. There is no prompt when
stdin is not a terminal — a run redirected to a log, backgrounded, or driven from CI
raises immediately instead, rather than blocking for hours on a question nobody is there
to answer. SEC asks for a descriptive string rather than a bare address; see their
[webmaster FAQ](https://www.sec.gov/os/webmaster-faq#developers).

This is network-bound and takes hours on a full run; `--resume` picks up an interrupted
collection, and `--skip-fetch` rebuilds only the universe table from panels already on disk
and needs no contact string. See `python -m src.Data.run --help`

**2. Run the backtest.** Needs the panels from step 1 — `Data/raw file/` is
gitignored, so a fresh clone has to collect them first. Without it this step (and
`--rebuild`, which recomputes returns from saved weights and prices) stops at a
missing `monthly_price.csv`.

```bash
python -m src.Empirical_Analysis.run --n-workers 5
```

Writes `summary.csv`, `monthly_returns.csv`, `cumulative_wealth.csv`, and per-strategy
diagnostics to `Data/Result/raw_backtest/`. Add `--only equal_weight epo ppp` to skip the slower
proposed-model runs while iterating, or `--rebuild` to recompute metrics from previously saved
weights without re-running the backtest. Full flag reference and methodology notes are in
[`src/Empirical_Analysis/README.md`](src/Empirical_Analysis/README.md).

**3. Build the paper's tables.**

```bash
python -m src.Empirical_Analysis.tables
```

Reads `raw_backtest/` and writes the three reported tables twice — as CSV to
`processed_backtest/` and as `booktabs` LaTeX to `latex/`, ready to `\input{}` into Overleaf.
Table 3 is the thought experiment of `src/Empirical_Analysis/experiment_thought.py`; pass
`--skip-experiment` to build only Tables 1 and 2, or `--n-sim 1000` for a faster Monte Carlo.

**4. Tests.**

```bash
pytest
```

`pytest -m "not slow"` skips the simulation-backed Sharpe-ratio inference checks, which take
seconds rather than milliseconds.

## Data sources

- Accounting data: [SEC EDGAR](https://www.sec.gov/edgar) (XBRL company facts)
- Price data: [Yahoo Finance](https://finance.yahoo.com/)
- Fama-French 5-factor returns: [Kenneth French's data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- Historical S&P 500 constituents: [hanshof/sp500_constituents](https://github.com/hanshof/sp500_constituents)

## Citation

If you use this code, please cite:

```bibtex
[bibtex entry]
```

## AI use

AI was used in building this repository and in drafting parts of the accompanying paper —
implementing code from written specifications, documentation, debugging, and the results
discussion. [`AI_USE.md`](AI_USE.md) states where, how the output was verified, and what
it was not used for.

## License

[MIT](LICENSE).

# Portfolio Optimization

Code and data pipeline for a residual-income-model (RIM) portfolio construction method,
benchmarked against Bielstein & Hanauer's ICC mean-variance portfolio, Enhanced Portfolio
Optimization (EPO) and the Parametric Portfolio Policy (PPP) on a point-in-time S&P 500
universe. Written to accompany the paper *Value-Investing Portfolio Optimization: Accounting
for the Uncertainty of Fundamentals* by Minh Quang Nguyen, Thu Thuy Cu and Cuong Quoc Nguyen
— see [Citation](#citation).

## What's here

Four portfolio-weighting rules are backtested on the same universe and calendar, so a
performance difference between them is a difference in the weighting rule and nothing else:

| Model | Source | Idea |
|---|---|---|
| `src/Proposed_Model` | this paper | Simulates each firm's forward accounting characteristics from a fitted copula, maps each simulated path to an implied return via a residual-income valuation, and optimises a CRRA-utility portfolio over the resulting return distribution |
| `src/EPO` | Pedersen, Babu & Levine (2021) | Enhanced Portfolio Optimization — the closed-form solution to a robust max-min problem over an uncertainty region for expected returns, which reduces to mean-variance against a correlation matrix shrunk toward the identity. The paper's Equity 4 specification: XSMOM × σ signal, equal-weighted daily covariance, 5% pre-shrink |
| `src/ICC_MVO` | Bielstein & Hanauer (2019) | Mean-variance optimization with forward-looking return estimates — a maximum-Sharpe portfolio whose expected returns are the Gebhardt, Lee & Swaminathan (2001) implied cost of capital plus rescaled momentum, with a Ledoit-Wolf covariance and a 5% cap per name. The point-estimate benchmark Table 1 is built around |
| `src/PPP` | Brandt, Santa-Clara & Valkanov (2009) | Parametric Portfolio Policy — portfolio weight parameterised directly as a linear function of firm characteristics (momentum, book-to-market, size) |

`equal_weight` is included as the zero-skill benchmark all four are compared against.

`src/Data` collects and cleans the raw panels every model reads: point-in-time S&P 500
membership, monthly adjusted prices, and quarterly XBRL accounting data. `src/Empirical_Analysis`
is the backtest harness — it fixes the universe, rebalance calendar, transaction costs, and
performance metrics so every strategy is compared on equal footing, and tests whether
each strategy's Sharpe ratio differs from equal-weight's using the Ledoit-Wolf (2008)
studentized bootstrap. Each package's own `README.md` says what it does and how to run it;
the methodology is in the paper and in the modules' docstrings.

## Repository structure

```
constant.py                  All tunable parameters and file paths, in one place.
main.py                      Runs the whole pipeline end to end: data, backtest, tables.
src/
  Data/                      Fetches and cleans the raw panels (SEC EDGAR, Yahoo Finance,
                              S&P 500 constituent history) and builds the per-date universe.
  Proposed_Model/             The paper's residual-income portfolio model.
  ICC_MVO/                    Bielstein & Hanauer's ICC mean-variance baseline.
  EPO/                        Enhanced Portfolio Optimization baseline.
  PPP/                        Parametric Portfolio Policy baseline.
  Empirical_Analysis/         The harness: run.py (backtest), tables.py (paper tables),
    backtest/                 Engine, data loading, performance metrics.
    inference/                Sharpe-ratio, certainty-equivalent and mean-return tests.
    experiment/               The thought experiment behind Table 3.
  panel.py                    Shared panel I/O (cached CSV reads, book-equity definition).
tests/                        pytest suite.
Data/raw file/                Data panels. Only the three input files under "Data" below
                              are committed; step 1 fetches the rest.
Data/Result/                  Generated output, committed (except implied_return/).
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

`pyvinecopulib` (used by the copula fit in `Proposed_Model`) ships prebuilt wheels for the
common platforms; if it fails to install, check that your Python/OS combination has one
available.

On Windows, keep the clone's path short (under ~100 characters, e.g.
`C:\src\Portfolio_Optimization`) unless [long paths are enabled](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation#enable-long-paths-in-windows-10-version-1607-and-later):
some files the dependencies install sit ~145 characters below the repository root, and past
Windows' 260-character limit `pip install` fails with `[WinError 206]` and Python cannot import
them (`ModuleNotFoundError` inside `scipy`).

Every command below is run from the repository root with that environment active
(`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere). With `uv` you
can skip activating and prefix each command with `uv run` instead, e.g.
`uv run python -m src.Empirical_Analysis.tables`.

## Data

Three input files ship with the repository in `Data/raw file/`, because nothing in the code
fetches them:

| File | Source | Format |
|---|---|---|
| `sp_500_historical_components.csv` | [hanshof/sp500_constituents](https://github.com/hanshof/sp500_constituents) | `date,tickers` — one row per date, the members as one comma-separated string |
| `monthly_FF5.csv` | [Kenneth French's data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), 5 factors (2x3), monthly | `date,Mkt-RF,SMB,HML,RMW,CMA,RF`, date as `YYYYMM`, values in percent |
| `daily_FF5.csv` | the same, daily | the same columns, date as `YYYYMMDD` |

To extend them past their last date, download fresh copies and put them in the same format:
French's CSVs open with a few lines of description and end with an annual section, and
both need removing, with the first column named `date`.

Everything else in `Data/raw file/` — prices, accounting data, splits, industries, the
universe — is written by step 1 below, and not committed (`daily_price.csv` alone is over
GitHub's 100 MB limit). The backtest output that step 2 writes to `Data/Result/` *is*
committed, so step 3 runs on a fresh clone without steps 1 and 2.

## Reproducing the results

**What reproduces exactly, and what cannot.** `Data/Result/` is the run of record: step 3
alone rebuilds the paper's three tables from it, number for number. Every source of
randomness in steps 2 and 3 is seeded, so the same data gives the same results. The data
itself is not frozen, though: step 1 fetches prices and filings as Yahoo Finance and SEC
EDGAR serve them *today*, and both drop names over time — acquired and delisted index
members stop resolving, so they fall out of the universe. A full rerun from step 1
therefore gives close, but not identical, numbers.

**All at once:** `python main.py` runs the three steps below in order — data
collection, backtest, tables — and can resume from any point with
`--skip-data`/`--skip-backtest`/`--skip-tables`. See `python main.py --help`.
The steps are broken out individually below for anyone who wants to run just
one of them, or needs a flag `main.py` doesn't expose.

**1. Collect the data.** Fetches prices from Yahoo Finance, accounting data from SEC EDGAR,
and rebuilds the point-in-time investable universe from the S&P 500 constituent history:

```bash
python -m src.Data.run --user-agent "Your Name you@example.com"
```

SEC EDGAR identifies and throttles callers by User-Agent, so it needs a contact string of
your own: pass `--user-agent`, set `SEC_USER_AGENT` in your environment, or answer the
prompt at a terminal. Without one, a non-interactive run stops before sending anything.

This is network-bound and takes hours on a full run; `--resume` picks up an interrupted
collection, and `--skip-fetch` rebuilds only the universe table from panels already on disk.
See `python -m src.Data.run --help`

**2. Run the backtest.**

```bash
python -m src.Empirical_Analysis.run --n-workers 5
```

Needs step 1's panels in `Data/raw file/` — `--rebuild` included, since it replays the saved
weights against the price panel. Without them it stops at a `FileNotFoundError` naming the
missing file.

Writes `summary.csv`, `monthly_returns.csv`, `cumulative_wealth.csv`, and per-strategy
diagnostics to `Data/Result/raw_backtest/`. Add `--only equal_weight epo ppp` to skip the slower
proposed-model runs while iterating, or `--rebuild` to recompute metrics from previously saved
weights without re-running the backtest. `python -m src.Empirical_Analysis.run --help` lists
every flag; see also [`src/Empirical_Analysis/README.md`](src/Empirical_Analysis/README.md).

**3. Build the paper's tables.**

```bash
python -m src.Empirical_Analysis.tables
```

Reads `raw_backtest/` and writes the three reported tables twice — as CSV to
`processed_backtest/` and as `booktabs` LaTeX to `latex/`, ready to `\input{}` into Overleaf.
Table 3 is the thought experiment of `src/Empirical_Analysis/experiment/experiment_thought.py`; pass
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
@unpublished{nguyen2026value,
  author = {Nguyen, Minh Quang and Cu, Thu Thuy and Nguyen, Cuong Quoc},
  title  = {Value-Investing Portfolio Optimization: Accounting for the Uncertainty of Fundamentals},
  year   = {2026},
  note   = {Working paper}
}
```

This entry will be updated with the journal reference once the paper is published.

## License

MIT — see [`LICENSE`](LICENSE).

# Data

Collects and cleans the three raw panels every model reads — monthly and daily
adjusted prices, quarterly XBRL accounting facts, and the point-in-time S&P 500
membership history — then derives the per-date investable universe from them.
Nothing downstream (`EPO`, `PPP`, `Proposed_Model`, `Empirical_Analysis`) fetches
its own data; they all read the CSVs this package writes to `Data/raw file/`.

## Reproduce from scratch

```bash
python -m src.Data.run
```

This is **network-bound and takes hours** on a full ~570-ticker run: one SEC
EDGAR company-facts request and one Yahoo Finance history request per ticker,
plus splits. Runs, in order (`run.main`'s docstring has the full rationale for
why the order is load-bearing):

1. `ticker_list.build_ticker_list()` — writes `ticker.csv`, the fetch list,
   derived from `sp_500_historical_components.csv` and `constant.universe_asof`.
2. `run()` — fetches prices and accounting facts per ticker, writes
   `monthly_price.csv`, `daily_price.csv`, `accounting.csv`.
3. `run_splits()` — fetches split factors, writes `splits.csv`.
4. `run_share_adjustment()` — adds `adj_shares` to the price and accounting
   panels so the back-adjusted price and the point-in-time SEC share count sit
   on one split basis (see `split_adjust.py`'s module docstring — this is not
   optional, `pb`, `bvps`, `rps` and the RIM valuation price are all wrong
   without it).
5. `run_applicable_ticker()` — derives `applicable_ticker.csv`, the per-date
   candidate universe, from the panels as they now stand.

Useful flags:

- `--resume` — fetch only tickers missing from the panels on disk and append,
  instead of refetching everything. Use after widening the universe or to pick
  up an interrupted collection.
- `--checkpoint N` — flush to disk every `N` fetched tickers (default 25).
  Lower survives an interruption with less lost.
- `--skip-fetch` — skip steps 1-4 and only rebuild `applicable_ticker.csv` from
  the panels already on disk. This is all that changing `constant.universe_asof`,
  `n_quarter`, or `n_month` needs — the universe screen is a pure function of
  `accounting.csv` and `monthly_price.csv`, so it costs seconds and touches
  neither SEC nor Yahoo.

```bash
python -m src.Data.run --help
```

## Prerequisites

- Outbound HTTPS to `www.sec.gov` and Yahoo Finance.
- **A SEC contact string of your own.** SEC identifies and throttles callers by
  `User-Agent`, so there is no shared default: a descriptive string
  (`Your Name you@example.com`, not a bare address). At a terminal you are
  prompted for one; pass it up front, or set it once per shell, to skip that.

  ```bash
  python -m src.Data.run --user-agent "Your Name you@example.com"
  # or, once per shell:
  $env:SEC_USER_AGENT = "Your Name you@example.com"   # PowerShell
  export SEC_USER_AGENT="Your Name you@example.com"   # macOS/Linux
  ```

  The prompt is gated on stdin being a terminal, so a redirected, backgrounded
  or CI run raises straight away rather than blocking on a question nobody is
  there to answer. `--skip-fetch` touches no network and needs none of this.
  See `accounting_info.sec_header` and
  [SEC's webmaster FAQ](https://www.sec.gov/os/webmaster-faq#developers).
- **Windows, project path with non-ASCII characters**: `network.py` exists
  because `yfinance`'s `curl_cffi` backend cannot open a CA bundle whose path
  isn't representable in the process ANSI codepage — this project's own path
  (`Học tập và làm việc`) triggers it. It's handled automatically
  (`ensure_encodable_cacert` relocates the bundle to an ASCII temp path), but
  if every ticker fails with "possibly delisted" on a fresh clone, that's the
  first thing to check.

## Output (`Data/raw file/`, paths in `constant.py`)

| File | Contents |
|---|---|
| `ticker.csv` | Fetch list |
| `monthly_price.csv`, `daily_price.csv` | `adj_close`, `close`, `shares`, `adj_shares` per ticker/date |
| `accounting.csv` | `net_income`, `revenue`, `total_assets`, `book_value`, `minority_interest`, `preferred_stock`, `shares`, `adj_shares` per ticker/quarter |
| `splits.csv` | Split events per ticker |
| `applicable_ticker.csv` | Per-date candidate universe — one comma-joined ticker list per formation date |

## The universe screen

`exclusion.applicable_ticker` is what `run_applicable_ticker` writes. It is
stated precisely, rule by rule, in
[`src/Empirical_Analysis/README.md`](../Empirical_Analysis/README.md#the-exclusion-rule) —
not repeated here to avoid the two drifting apart. The short version: a ticker
qualifies at date `t` iff it's in the S&P 500 (or a frozen pool, see
`constant.universe_asof`), has a complete trailing price window, a complete
trailing accounting window, and strictly positive book equity and revenue
throughout that window.

## Modules

- `ticker_list.py` — builds the fetch list from S&P 500 constituent history.
- `accounting_info.py` — `Sec_Data_Restructure`: SEC EDGAR XBRL fetch and
  line-item restructuring (`constant.ratio` maps each accounting field to its
  candidate XBRL tags, since filers don't all tag the same concept the same
  way). See `Instruction/data-collection-rim.md` for the tag-mapping rationale.
- `price_data.py` — Yahoo Finance price and split fetch.
- `split_adjust.py` — puts the back-adjusted price and the point-in-time SEC
  share count on one split basis.
- `exclusion.py` — the universe screen, `applicable_ticker`.
- `network.py` — the Windows CA-bundle workaround above.
- `run.py` — the CLI driver and the four functions it calls, each independently
  callable and documented with its own reproduction case (see their docstrings
  for `resume`/checkpoint semantics).
- `utils.py` — shared fetch helpers.

## Changing the universe after the fact

Moving `constant.universe_asof` between a frozen date and `None` (reconstituting)
changes the fetch list itself (464 names frozen vs. 707 as a union over
`constant.testing_period`), so it needs `python -m src.Data.run --resume` to
widen the panels, not just `--skip-fetch`. Once the panels already cover the
wider list, switching back and forth only needs `--skip-fetch`. See
`constant.universe_asof`'s comment for the full trade-off.

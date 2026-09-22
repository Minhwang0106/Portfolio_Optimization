# Data

Collects and cleans the panels every model reads — monthly and daily adjusted
prices (Yahoo Finance), quarterly XBRL accounting facts (SEC EDGAR), split
factors and each firm's Fama-French 48 industry — and builds from them the
point-in-time investable universe: at each formation date, the S&P 500 members
with complete price and accounting histories and positive book equity and
revenue. No other package fetches data; they all read what this one writes to
`Data/raw file/`.

## How to run

```bash
python -m src.Data.run --user-agent "Your Name you@example.com"
```

Network-bound: hours on a full run. SEC requires a contact string of your own,
passed as `--user-agent`, set once as `SEC_USER_AGENT`, or typed at the prompt.
It needs `sp_500_historical_components.csv` in `Data/raw file/`, which ships
with the repository.

- `--resume` — fetch only tickers missing from the panels on disk; picks up an
  interrupted run.
- `--skip-fetch` — no network; only rebuild the universe (`applicable_ticker.csv`)
  from the panels already on disk.

`python -m src.Data.run --help` lists every flag.

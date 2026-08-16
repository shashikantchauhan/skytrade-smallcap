# SkyTrade Smallcap

Forked from the main SkyTrade/p-trade project on 2026-08-14, scoped to the
**Nifty Smallcap 250** universe on **weekly** signals instead of the parent
project's hourly signals over its ~220-symbol large/midcap universe.

**Why weekly, why smallcap:** hourly-cadence trading doesn't suit smallcaps
well (wider spreads, lower liquidity, more noise) -- 2026-08-14 research
resampled the parent's engine onto weekly bars and found win rate/profit
factor held up (56-60%, PF 3.5-3.8) while cutting trade frequency ~90x,
which matters more here than it did for the parent's universe. See
`infrastructure/weekly_resample.py`'s docstring for the mechanism (Kite has
no native weekly interval, so this resamples real day-interval Kite data
client-side) and git history for the full research.

**What's different from the parent project** (everything else -- the
Lorentzian classifier in `alpha_engine.py`, ranking, cost model, paper
trading mechanics -- is unchanged, reused as-is):
- `config/nifty_smallcap250_symbols.txt` (250 symbols) instead of
  `config/symbols.txt` -- same Kite account, no separate app registration
  needed.
- `TRADING_SCANNER_CANDLE_INTERVAL=week` (bridged client-side, see above)
  instead of `1h`.
- **No stop-loss.** Weekly losers average -6.25%, well past any tick-level
  threshold that would make sense intraday -- positions exit on the
  engine's own signal only. `live_pipeline.py` (the WebSocket
  live-ticker/stop-loss service) is **not part of this fork's deployment**
  -- weekly signals only need a once-daily check, not a continuously
  connected session. `deploy/p-trade-live.service` is left in the repo
  unused/for reference only; do not enable it here.
- Capital: Rs 5L / 10 slots, Rs 50k each (`TRADING_SCANNER_PAPER_CAPITAL`
  / `_PAPER_SLOTS`), separate pool from the parent project's live account.
- Deployment is a **daily crontab entry** running
  `trading_scanner.signals` once after market close (matching the parent's
  actual production method -- see `.github/workflows/hourly-signals.yml`'s
  own comment: scheduling is real crontab on the VPS, not GitHub Actions or
  systemd), not the hourly cadence or the WebSocket service.
- ~55 of the 250 symbols are recent listings with <200 weekly bars
  (~4 years) of history -- `_MINIMUM_CANDLES` in both `backtest.py` and
  `application/signal_pipeline.py` requires that much warm-up before a
  symbol signals at all, same rule as the parent project reused unchanged.
  They'll simply stay dormant until they individually cross that bar.

---

An NSE market signal scanner built around a Pine Script-derived technical
strategy, with an hourly ingestion pipeline and a simulated (paper) trading
account for tracking performance over time.

## Project Structure

```text
src/trading_scanner/
├── alpha_engine.py      # Strategy signal engine
├── application/         # Pipeline, backtest, and paper-trading logic
├── config/               # Centralized application configuration
├── infrastructure/       # Market-data providers and storage adapters
├── validation/            # Per-bar signal validation tooling
├── main.py                # Application orchestration
├── signals.py              # Hourly pipeline entry point
├── webapp.py                # Web dashboard
└── validate.py               # Validation command-line entry point
config/
└── symbols.txt              # One market symbol per line
```

## Running

Python 3.12 is required.

```bash
poetry install
poetry run trading-scanner
```

Or install dependencies directly:

```bash
PYTHONPATH=src python -m trading_scanner.main
```

## Configuration

Configuration is centralized in `trading_scanner.config.settings` and can be
overridden with environment variables or a `.env` file. See
`config/settings.py` for the full list of settings.

## Symbols File

Place one market symbol per line in `config/symbols.txt`. Blank lines,
whitespace, and duplicate symbols are ignored.

## Signal Validation

Export every candle's signal-engine result to compare it bar-by-bar against
an external reference chart:

```bash
PYTHONPATH=src python -m trading_scanner.validate \
  --symbol SYMBOL.NS \
  --interval 1h \
  --days 10
```

## Hourly Signal Pipeline

`trading_scanner.signals` runs the scanner across every symbol in
`config/symbols.txt` on a schedule, accumulating candles in a database so the
strategy has real warm-up history instead of re-downloading a short window
every run. See `application/signal_pipeline.py` for details.

### Running locally with no account

The storage layer speaks directly to a local SQLite file, so the pipeline can
be developed and tested without a hosted database account:

```bash
TRADING_SCANNER_TURSO_URL="file:local.db" \
PYTHONPATH=src python -m trading_scanner.signals
```

## Web Dashboard

`trading_scanner.webapp` is a small FastAPI app for viewing pipeline/account
status and adjusting configuration. See `webapp.py` for details.

```bash
TRADING_SCANNER_DASHBOARD_PASSWORD=<pick-a-password> \
PYTHONPATH=src python -m trading_scanner.webapp
```

## Tests

```bash
PYTHONPATH=src pytest
```

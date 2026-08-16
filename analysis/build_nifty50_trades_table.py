"""Builds/refreshes ``nifty50_trades`` -- a standalone copy of every
``trades`` row for symbols in ``config/nifty50_symbols.txt``, so Nifty50-
only backtesting/analysis can query one small, clean table instead of
filtering the full (220-symbol) ``trades`` table by hand every time.

2026-08-14: built at the user's explicit request to isolate Nifty50
analysis from the rest of the (much noisier, smaller/mid-cap-heavy)
symbol universe -- see the session's futures+options hedge analysis this
same day for why: Nifty50 names showed a much thinner right tail (biggest
single-trade winner only +22.6%, vs +41.3% across the full universe),
so mixing the two skews conclusions either direction depending which
dominates the sample.

This is a **snapshot copy**, not a live view -- it will drift out of sync
as new trades close. Re-run this script whenever you want it refreshed
(safe to re-run any time: drops and rebuilds the table from scratch, a few
hundred KB of data, sub-second).

Nifty50 coverage note: 49 of the current 50 Nifty50 constituents are
covered -- Tata Motors and LTIMindtree are not in this deployment's
tracked 220-symbol universe (``config/symbols.txt``) at all, so they have
no trade history to copy. Add them there first (and let the system trade
them for a while) if you want full 50/50 coverage later.

Usage: python3 analysis/build_nifty50_trades_table.py [path-to-db]
       (defaults to data/skytrade.db, same convention as the other
       analysis/ scripts)
"""
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _REPO_ROOT / "data" / "skytrade.db"
_SYMBOLS_FILE = _REPO_ROOT / "config" / "nifty50_symbols.txt"

_CREATE_NIFTY50_TRADES_TABLE = """
CREATE TABLE nifty50_trades (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_timestamp TEXT NOT NULL,
    entry_price REAL NOT NULL,
    prediction_at_entry INTEGER NOT NULL,
    is_early_signal_flip INTEGER NOT NULL,
    exit_timestamp TEXT,
    exit_price REAL,
    pnl_percent REAL,
    status TEXT NOT NULL DEFAULT 'open',
    adx_at_entry REAL,
    regime_normalized_at_entry REAL,
    volatility_margin_at_entry REAL,
    volatility_filter_passed INTEGER,
    regime_filter_passed INTEGER,
    adx_filter_passed INTEGER
)
"""
# Mirrors trades' own columns exactly (see infrastructure/db/trades.py) --
# `id` here is kept equal to the source row's own id (not re-autoincremented)
# specifically so a row can always be traced back to its origin in `trades`.


def build(db_path: Path) -> None:
    symbols = [
        line.strip() for line in _SYMBOLS_FILE.read_text().splitlines() if line.strip()
    ]
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("DROP TABLE IF EXISTS nifty50_trades")
    cur.execute(_CREATE_NIFTY50_TRADES_TABLE)

    placeholders = ",".join("?" * len(symbols))
    cur.execute(
        f"""
        INSERT INTO nifty50_trades
        SELECT * FROM trades WHERE symbol IN ({placeholders})
        """,
        symbols,
    )
    con.commit()

    row_count = cur.execute("SELECT COUNT(*) FROM nifty50_trades").fetchone()[0]
    symbols_with_data = cur.execute(
        "SELECT COUNT(DISTINCT symbol) FROM nifty50_trades"
    ).fetchone()[0]
    missing = sorted(
        set(symbols)
        - {row[0] for row in cur.execute("SELECT DISTINCT symbol FROM nifty50_trades")}
    )
    con.close()

    print(f"nifty50_trades built from {db_path}")
    print(
        f"  {row_count} rows copied, "
        f"{symbols_with_data}/{len(symbols)} symbols have trade history"
    )
    if missing:
        print(f"  No trade history yet for: {missing}")


if __name__ == "__main__":
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_DB
    build(db_path)

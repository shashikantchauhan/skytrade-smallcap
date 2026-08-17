"""Centralized runtime configuration for the market scanner."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Settings required to run one market scan."""

    scan_interval_hours: int
    candle_interval: str
    candle_history: int
    symbols_file: Path
    logging_level: int
    turso_database_url: str | None
    turso_auth_token: str | None
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    index_symbol: str | None
    kite_api_key: str | None
    kite_api_secret: str | None
    # The kill switch for real order execution -- see application/
    # live_execution.py. Defaults fully OFF; every one of these must be
    # explicitly set to place a single real order. live_trading_symbols
    # empty means nothing is allowed regardless of live_trading_enabled --
    # there is no "all symbols" wildcard, on purpose.
    live_trading_enabled: bool
    live_trading_symbols: frozenset[str]
    live_trading_max_lots: int
    # Real, capital-gated futures paper account (application/futures_trading.py)
    # -- restricted to this allowlist (Nifty50 by default) rather than the
    # full 220-symbol universe, so the extra Kite margin-API call per signal
    # this account needs stays bounded. Empty file/no file -> nothing
    # trades on this book, same no-wildcard-default philosophy as
    # live_trading_symbols above.
    futures_paper_symbols_file: Path
    # 2026-08-17: distinguishes which deployment a Telegram message came
    # from -- e.g. this repo's own p-trade vs. the skytrade-smallcap fork,
    # which reuses the exact same bot/chat ID. Shown in every message
    # header (see infrastructure/telegram.py's _format_signal). Defaulted
    # here (unlike every other field above) so existing AppConfig(...)
    # call sites -- test fixtures mostly -- don't all need updating just
    # for this; load_config() below still sets it explicitly from env.
    notification_label: str = "Nifty50"


def load_config() -> AppConfig:
    """Load application settings from environment variables and safe defaults."""
    load_dotenv()
    return AppConfig(
        scan_interval_hours=_positive_int("TRADING_SCANNER_SCAN_INTERVAL_HOURS", 1),
        candle_interval=os.getenv("TRADING_SCANNER_CANDLE_INTERVAL", "1h"),
        candle_history=_positive_int("TRADING_SCANNER_CANDLE_HISTORY", 300),
        symbols_file=Path(os.getenv("TRADING_SCANNER_SYMBOLS_FILE", "config/symbols.txt")),
        logging_level=_logging_level(os.getenv("TRADING_SCANNER_LOGGING_LEVEL", "INFO")),
        turso_database_url=os.getenv("TRADING_SCANNER_TURSO_URL"),
        turso_auth_token=os.getenv("TRADING_SCANNER_TURSO_AUTH_TOKEN"),
        telegram_bot_token=os.getenv("TRADING_SCANNER_TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TRADING_SCANNER_TELEGRAM_CHAT_ID"),
        notification_label=os.getenv("TRADING_SCANNER_NOTIFICATION_LABEL", "Nifty50"),
        # NIFTY 50 -- broad NSE benchmark, not tied to any single sector.
        # Evaluated once per run and shown alongside every stock signal so you
        # can judge whether a signal lines up with the broader market or looks
        # like noise against it. Set to "" to disable index tracking entirely.
        index_symbol=os.getenv("TRADING_SCANNER_INDEX_SYMBOL", "^NSEI") or None,
        kite_api_key=os.getenv("TRADING_SCANNER_KITE_API_KEY"),
        kite_api_secret=os.getenv("TRADING_SCANNER_KITE_API_SECRET"),
        live_trading_enabled=_bool_flag("TRADING_SCANNER_LIVE_TRADING_ENABLED", default=False),
        live_trading_symbols=frozenset(
            s.strip()
            for s in os.getenv("TRADING_SCANNER_LIVE_TRADING_SYMBOLS", "").split(",")
            if s.strip()
        ),
        live_trading_max_lots=_positive_int("TRADING_SCANNER_LIVE_TRADING_MAX_LOTS", 1),
        futures_paper_symbols_file=Path(
            os.getenv(
                "TRADING_SCANNER_FUTURES_PAPER_SYMBOLS_FILE", "config/nifty50_symbols.txt"
            )
        ),
    )


def _positive_int(name: str, default: int) -> int:
    """Read a positive integer environment setting."""
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer; got {parsed}.")
    return parsed


def _bool_flag(name: str, default: bool) -> bool:
    """Explicit opt-in parsing for the live-trading kill switch -- only the
    exact string "true" (case-insensitive) turns it on; anything else
    (unset, "false", a typo) stays off. No implicit truthiness on a
    setting this consequential."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() == "true"


def _logging_level(value: str) -> int:
    """Convert a configured logging level name into a logging constant."""
    level = getattr(logging, value.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"TRADING_SCANNER_LOGGING_LEVEL is invalid: {value!r}.")
    return level

# ruff: noqa: E501 -- long lines below are almost all inside embedded
# HTML/CSS/JS template strings, which read worse wrapped than over-length.
"""Web dashboard: view the paper account live and control the pipeline.

Single-file FastAPI app, protected by a cookie-based login (one shared
password -- this is a personal tool, not a multi-user product). Sessions are
kept in an in-memory dict, so a dashboard restart logs everyone out -- fine
for a single-user tool. Reads directly from the same Turso database the
hourly pipeline writes to; no separate data layer.

Run with: `trading-scanner-dashboard` (see pyproject.toml), or directly:
    PYTHONPATH=src python -m trading_scanner.webapp
"""

import asyncio
import logging
import os
import secrets
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from fastapi import Cookie, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from kiteconnect import KiteConnect
from pydantic import BaseModel

from trading_scanner.application import futures_trading, paper_trading
from trading_scanner.application.options_analytics import enrich_trade
from trading_scanner.config.settings import load_config
from trading_scanner.domain.models import PaperPosition
from trading_scanner.infrastructure.db import (
    TursoFuturesPaperAccountRepository,
    TursoFuturesTradeRepository,
    TursoKiteSessionRepository,
    TursoOptionsTradeRepository,
    TursoPaperAccountRepository,
    TursoTradeRepository,
    create_turso_client,
)
from trading_scanner.infrastructure.kite import (
    KiteDerivativesChain,
    build_login_url,
    exchange_request_token,
)
from trading_scanner.infrastructure.kite import (
    get_last_prices as kite_get_last_prices,
)
from trading_scanner.infrastructure.yahoo import YahooProvider

_yahoo = YahooProvider()

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _REPO_ROOT / ".env"
_LOG_PATH = Path(os.getenv("TRADING_SCANNER_LOG_PATH", "/var/log/p-trade/signals.log"))
_BACKTEST_LOG_PATH = _LOG_PATH.with_name("derivatives-backtest.log")
_SESSION_COOKIE = "ptrade_session"
_SESSION_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days

app = FastAPI(title="SkyTrade dashboard")

# token -> {expiry, role, name}. In-memory: fine for a single-process
# personal tool; a restart just means logging in again.
_sessions: dict[str, dict] = {}


def _dashboard_password() -> str:
    password = os.getenv("TRADING_SCANNER_DASHBOARD_PASSWORD")
    if not password:
        raise HTTPException(
            status_code=500,
            detail="TRADING_SCANNER_DASHBOARD_PASSWORD is not set on the server.",
        )
    return password


def _viewer_credentials() -> dict[str, str]:
    """Named view-only logins, e.g. for a spouse/friend who should see the
    dashboard but never touch Kite login or trigger a pipeline run.

    ``TRADING_SCANNER_VIEWER_LOGINS="wife:somepassword,friend:otherpassword"``
    -- each name gets its own password so access can be revoked individually
    later without changing the admin password everyone else still uses.
    """
    raw = os.getenv("TRADING_SCANNER_VIEWER_LOGINS", "")
    result: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        name, password = entry.split(":", 1)
        if name.strip() and password:
            result[name.strip()] = password
    return result


def _authenticate(password: str) -> tuple[str, str] | None:
    """Returns (role, name) on a password match, checking the admin
    password first, then each named viewer -- None if it matches nothing."""
    if secrets.compare_digest(password, _dashboard_password()):
        return "admin", "admin"
    for name, viewer_password in _viewer_credentials().items():
        if secrets.compare_digest(password, viewer_password):
            return "viewer", name
    return None


def _require_session(ptrade_session: str | None = Cookie(default=None)) -> None:
    """API-route auth: 401 JSON if the session cookie is missing/expired.
    Allows both admin and viewer roles -- use ``_require_admin`` for routes
    that touch Kite or trigger the pipeline."""
    session = _sessions.get(ptrade_session or "")
    if session is None or session["expiry"] < time.time():
        raise HTTPException(status_code=401, detail="Not logged in.")


def _require_admin(ptrade_session: str | None = Cookie(default=None)) -> None:
    """Admin-only routes: Kite login/status, triggering the pipeline or a
    backtest, and config changes -- a viewer (e.g. a spouse checking in on
    the numbers) should never be able to touch any of these, both to keep
    Kite credentials private and because a second person clicking 'Kite
    login' can stomp the one active session the pipeline depends on (this
    happened once -- see the commit that added this check)."""
    session = _sessions.get(ptrade_session or "")
    if session is None or session["expiry"] < time.time():
        raise HTTPException(status_code=401, detail="Not logged in.")
    if session["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only.")


def _client():
    config = load_config()
    if not config.turso_database_url:
        raise HTTPException(status_code=500, detail="TRADING_SCANNER_TURSO_URL is not set.")
    return create_turso_client(config.turso_database_url, config.turso_auth_token), config


def _decimal(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


async def _last_prices(positions: list[PaperPosition], client, config) -> dict[str, float]:
    """Fetch current market prices for open positions' symbols.

    Prefers a live Kite quote when a session is active -- Yahoo's
    ``get_last_prices`` downloads the last *daily close*, which during a
    trading session can lag the real price by a full day (the bug reported
    against the dashboard: showing yesterday's close while the live price
    had already moved). Falls back to Yahoo if Kite isn't configured or has
    no active session today. Both calls are blocking, so run off the event
    loop."""
    symbols = [p.symbol for p in positions]
    if not symbols:
        return {}
    if config.kite_api_key:
        repository = TursoKiteSessionRepository(client)
        await repository.ensure_schema()
        token_row = await repository.get_token()
        if token_row is not None:
            access_token, _obtained_at = token_row
            kite = KiteConnect(api_key=config.kite_api_key)
            kite.set_access_token(access_token)
            prices = await asyncio.to_thread(kite_get_last_prices, kite, symbols)
            if prices:
                return prices
    return await asyncio.to_thread(_yahoo.get_last_prices, symbols)


def _unrealized_pnl(position: PaperPosition, last_prices: dict[str, float]) -> dict:
    current_price = last_prices.get(position.symbol)
    if current_price is None:
        return {"current_price": None, "unrealized_pnl": None, "unrealized_pnl_pct": None}
    pnl = (Decimal(str(current_price)) - position.entry_price) * position.quantity
    return {
        "current_price": current_price,
        "unrealized_pnl": _decimal(pnl),
        "unrealized_pnl_pct": _decimal(pnl / position.capital_allocated * 100),
    }


@app.get("/", response_model=None)
async def index(request: Request) -> HTMLResponse | RedirectResponse:
    session = _sessions.get(request.cookies.get(_SESSION_COOKIE, ""))
    if session is None or session["expiry"] < time.time():
        return RedirectResponse("/login")
    return HTMLResponse(_PAGE)


@app.get("/login", response_class=HTMLResponse)
async def login_page() -> str:
    return _LOGIN_PAGE


class LoginRequest(BaseModel):
    password: str


@app.post("/login")
async def login(body: LoginRequest) -> JSONResponse:
    authenticated = _authenticate(body.password)
    if authenticated is None:
        raise HTTPException(status_code=401, detail="Wrong password.")
    role, name = authenticated
    token = secrets.token_urlsafe(32)
    _sessions[token] = {"expiry": time.time() + _SESSION_TTL_SECONDS, "role": role, "name": name}
    response = JSONResponse({"ok": True, "role": role})
    response.set_cookie(
        _SESSION_COOKIE,
        token,
        max_age=_SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
    )
    return response


@app.get("/api/me")
async def me(
    ptrade_session: str | None = Cookie(default=None), _: None = Depends(_require_session)
) -> JSONResponse:
    """Lets the dashboard's own JS know whether to show admin-only controls
    (Kite login/status, pipeline trigger, backtest trigger, config) --
    those routes are enforced server-side too via ``_require_admin``, this
    is just so a viewer's UI doesn't show buttons that would 403."""
    session = _sessions[ptrade_session]
    return JSONResponse({"role": session["role"], "name": session["name"]})


@app.post("/logout")
async def logout(ptrade_session: str | None = Cookie(default=None)) -> JSONResponse:
    _sessions.pop(ptrade_session or "", None)
    response = JSONResponse({"ok": True})
    response.delete_cookie(_SESSION_COOKIE)
    return response


@app.get("/kite/login")
async def kite_login(_: None = Depends(_require_admin)) -> RedirectResponse:
    """Send the user to Kite's own login page -- their Zerodha password is
    entered there, never on this server. Requires being logged into this
    dashboard first (so a stranger can't hijack the Kite session)."""
    config = load_config()
    if not config.kite_api_key:
        raise HTTPException(status_code=500, detail="TRADING_SCANNER_KITE_API_KEY is not set.")
    return RedirectResponse(build_login_url(config.kite_api_key))


@app.get("/kite/callback", response_class=HTMLResponse)
async def kite_callback(request: Request) -> str:
    """Kite redirects here after login with a one-time request_token, which
    is exchanged immediately for the day's access token and stored. Not
    behind _require_session -- Kite itself is the auth gate for this step,
    and the request_token is single-use/short-lived, so there's nothing
    sensitive to protect on this specific hop."""
    request_token = request.query_params.get("request_token")
    status_param = request.query_params.get("status")
    if status_param != "success" or not request_token:
        return "<p>Kite login failed or was cancelled. You can close this tab and try again.</p>"
    config = load_config()
    if not config.kite_api_key or not config.kite_api_secret:
        return "<p>Kite API key/secret not configured on the server.</p>"
    try:
        access_token, obtained_at = exchange_request_token(
            config.kite_api_key, config.kite_api_secret, request_token
        )
    except Exception as error:
        return f"<p>Failed to exchange Kite token: {error}</p>"
    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    try:
        repository = TursoKiteSessionRepository(client)
        await repository.ensure_schema()
        await repository.set_token(access_token, obtained_at)
    finally:
        await client.close()
    return (
        "<p>Kite login successful -- today's session is active. "
        '<a href="/">Back to dashboard</a></p>'
    )


@app.get("/api/kite-status")
async def kite_status(_: None = Depends(_require_admin)) -> JSONResponse:
    config = load_config()
    if not config.kite_api_key:
        return JSONResponse({"configured": False})
    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    try:
        repository = TursoKiteSessionRepository(client)
        await repository.ensure_schema()
        token_row = await repository.get_token()
    finally:
        await client.close()
    if token_row is None:
        return JSONResponse({"configured": True, "logged_in": False})
    return JSONResponse({"configured": True, "logged_in": True, "obtained_at": token_row[1]})


@app.get("/api/status")
async def status(_: None = Depends(_require_session)) -> JSONResponse:
    client, config = _client()
    try:
        repository = TursoPaperAccountRepository(client, paper_trading.INITIAL_CAPITAL)
        cash_balance = await repository.get_cash_balance()
        open_positions = await repository.get_open_positions()
        total_equity = cash_balance + sum(
            (p.capital_allocated for p in open_positions), start=Decimal("0")
        )
        position_size = max(total_equity / paper_trading.TARGET_SLOTS, paper_trading.MIN_POSITION_SIZE)

        last_prices = await _last_prices(list(open_positions), client, config)
        unrealized_total = sum(
            (
                (Decimal(str(last_prices[p.symbol])) - p.entry_price) * p.quantity
                for p in open_positions
                if p.symbol in last_prices
            ),
            start=Decimal("0"),
        )
        priced_count = sum(1 for p in open_positions if p.symbol in last_prices)
        total_equity_mtm = total_equity + unrealized_total

        return JSONResponse(
            {
                "cash_balance": _decimal(cash_balance),
                "total_equity": _decimal(total_equity),
                "open_position_count": len(open_positions),
                "target_slots": paper_trading.TARGET_SLOTS,
                "current_slot_size": _decimal(position_size),
                "pnl_since_start": _decimal(total_equity - paper_trading.INITIAL_CAPITAL),
                "pnl_since_start_pct": _decimal(
                    (total_equity - paper_trading.INITIAL_CAPITAL)
                    / paper_trading.INITIAL_CAPITAL
                    * 100
                ),
                # Mark-to-market: total_equity above only reflects capital
                # committed at entry, not what open positions are worth right
                # now. unrealized_pnl is None if no live price could be
                # fetched for any open symbol (market closed, Yahoo hiccup).
                "unrealized_pnl": _decimal(unrealized_total) if priced_count else None,
                "total_equity_mtm": _decimal(total_equity_mtm) if priced_count else None,
                "pnl_since_start_mtm": (
                    _decimal(total_equity_mtm - paper_trading.INITIAL_CAPITAL)
                    if priced_count
                    else None
                ),
                "pnl_since_start_mtm_pct": (
                    _decimal(
                        (total_equity_mtm - paper_trading.INITIAL_CAPITAL)
                        / paper_trading.INITIAL_CAPITAL
                        * 100
                    )
                    if priced_count
                    else None
                ),
                "last_run": _last_run_summary(),
            }
        )
    finally:
        await client.close()


@app.get("/api/positions")
async def positions(_: None = Depends(_require_session)) -> JSONResponse:
    client, config = _client()
    try:
        repository = TursoPaperAccountRepository(client, paper_trading.INITIAL_CAPITAL)
        open_positions = await repository.get_open_positions()
        last_prices = await _last_prices(list(open_positions), client, config)
        return JSONResponse(
            [
                {
                    "symbol": p.symbol,
                    "entry_timestamp": p.entry_timestamp.isoformat(),
                    "entry_price": _decimal(p.entry_price),
                    "quantity": p.quantity,
                    "capital_allocated": _decimal(p.capital_allocated),
                    **_unrealized_pnl(p, last_prices),
                }
                for p in sorted(open_positions, key=lambda p: p.entry_timestamp, reverse=True)
            ]
        )
    finally:
        await client.close()


async def _futures_last_prices(positions: list, client, config) -> dict[str, float]:
    """Live LTP for the futures leg of each open combo (NFO segment) --
    separate from ``_last_prices`` above, which fetches equity/index quotes.
    Kite-only (no Yahoo fallback for NFO); returns {} if no active Kite
    session, matching this dashboard's other best-effort quote fetches."""
    if not positions or not config.kite_api_key:
        return {}
    repository = TursoKiteSessionRepository(client)
    await repository.ensure_schema()
    token_row = await repository.get_token()
    if token_row is None:
        return {}
    access_token, _obtained_at = token_row
    kite = KiteConnect(api_key=config.kite_api_key)
    kite.set_access_token(access_token)
    keys = [f"NFO:{p.futures_tradingsymbol}" for p in positions]

    def _fetch() -> dict:
        try:
            return kite.ltp(keys)
        except Exception:
            return {}

    data = await asyncio.to_thread(_fetch)
    return {
        p.symbol: data[f"NFO:{p.futures_tradingsymbol}"]["last_price"]
        for p in positions
        if f"NFO:{p.futures_tradingsymbol}" in data
    }


def _futures_unrealized_pnl(position, last_prices: dict[str, float]) -> dict:
    current_price = last_prices.get(position.symbol)
    if current_price is None:
        return {"current_price": None, "unrealized_pnl": None, "unrealized_pnl_pct": None}
    current = Decimal(str(current_price))
    pnl = (
        (current - position.futures_entry_price) * position.lot_size
        if position.side == "long"
        else (position.futures_entry_price - current) * position.lot_size
    )
    return {
        "current_price": current_price,
        "unrealized_pnl": _decimal(pnl),
        "unrealized_pnl_pct": _decimal(pnl / position.margin_allocated * 100),
    }


def _futures_monthly_summary(open_positions: list, closed_positions: list, window_days: int = 30) -> dict:
    """Trailing-window (default 30 days) performance summary for the
    futures paper account -- opened/closed counts, win rate, total P&L,
    average margin per trade.

    2026-08-17: built at the user's explicit request to track a full
    month of real paper performance before deciding whether to fund this
    for real trading next month -- the dashboard only ever showed
    "right now" state (open positions, last 50 closed) with no rollup a
    non-technical read could use to decide "was this month good enough."

    ``trades_opened`` counts by entry_timestamp (still-open or already
    closed, whichever) so a trade that opens AND closes inside the window
    counts once, not twice. ``trades_closed``/win rate/P&L are scoped by
    exit_timestamp instead, since that's when a trade's outcome is
    actually known -- a trade opened just before the window and closed
    inside it counts toward the outcome stats even though its own entry
    falls outside ``trades_opened``'s count, which is intentional: it's
    real P&L realized inside this window either way.
    """
    window_start = datetime.now(UTC) - timedelta(days=window_days)

    def _aware(ts: datetime) -> datetime:
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)

    opened = [
        p
        for p in (list(open_positions) + list(closed_positions))
        if _aware(p.entry_timestamp) >= window_start
    ]
    closed_in_window = [
        p
        for p in closed_positions
        if p.exit_timestamp is not None and _aware(p.exit_timestamp) >= window_start
    ]
    wins = [p for p in closed_in_window if p.pnl_amount is not None and p.pnl_amount > 0]
    total_pnl = sum(
        (p.pnl_amount for p in closed_in_window if p.pnl_amount is not None), start=Decimal("0")
    )
    total_margin_opened = sum((p.margin_allocated for p in opened), start=Decimal("0"))
    return {
        "window_days": window_days,
        "window_start": window_start.isoformat(),
        "trades_opened": len(opened),
        "trades_closed": len(closed_in_window),
        "trades_still_open": sum(1 for p in opened if p.status == "open"),
        "wins": len(wins),
        "losses": len(closed_in_window) - len(wins),
        "win_rate_pct": (
            _decimal(Decimal(100 * len(wins)) / len(closed_in_window)) if closed_in_window else None
        ),
        "total_pnl": _decimal(total_pnl),
        "avg_margin_per_trade": (
            _decimal(total_margin_opened / len(opened)) if opened else None
        ),
    }


@app.get("/api/futures-paper")
async def futures_paper(_: None = Depends(_require_session)) -> JSONResponse:
    """The real, capital-gated Nifty50 futures paper account (see
    application/futures_trading.py) -- separate book from the cash paper
    account above, own margin-based capital pool, own eligibility track
    record. Not shadow-tracking (that's /api/derivatives-shadow, uncapped,
    every symbol); this is only the trades that actually cleared the
    55%-win-rate bar and a real Kite margin check."""
    client, config = _client()
    try:
        repository = TursoFuturesPaperAccountRepository(client, futures_trading.FUTURES_INITIAL_CAPITAL)
        await repository.ensure_schema()
        cash_balance = await repository.get_cash_balance()
        open_positions = list(await repository.get_open_positions())
        # 500, not 50 -- large enough to cover several months at current
        # real trade volume (see _futures_monthly_summary, which needs
        # every closed trade inside the trailing window, not just the
        # most recent handful the plain "recent closed" list below shows).
        recent_closed = list(await repository.get_recent_closed_positions(500))
        last_prices = await _futures_last_prices(open_positions, client, config)
        total_margin_allocated = sum(
            (p.margin_allocated for p in open_positions), start=Decimal("0")
        )
        return JSONResponse(
            {
                "cash_balance": _decimal(cash_balance),
                "total_equity": _decimal(cash_balance + total_margin_allocated),
                "monthly_summary": _futures_monthly_summary(open_positions, recent_closed),
                "open_positions": [
                    {
                        "symbol": p.symbol,
                        "side": p.side,
                        "entry_timestamp": p.entry_timestamp.isoformat(),
                        "futures_entry_price": _decimal(p.futures_entry_price),
                        "futures_tradingsymbol": p.futures_tradingsymbol,
                        "hedge_tradingsymbol": p.hedge_tradingsymbol,
                        "lot_size": p.lot_size,
                        "margin_allocated": _decimal(p.margin_allocated),
                        **_futures_unrealized_pnl(p, last_prices),
                    }
                    for p in sorted(open_positions, key=lambda p: p.entry_timestamp, reverse=True)
                ],
                "recent_closed": [
                    {
                        "symbol": p.symbol,
                        "side": p.side,
                        "entry_timestamp": p.entry_timestamp.isoformat(),
                        "futures_entry_price": _decimal(p.futures_entry_price),
                        "exit_timestamp": p.exit_timestamp.isoformat() if p.exit_timestamp else None,
                        "futures_exit_price": _decimal(p.futures_exit_price),
                        "margin_allocated": _decimal(p.margin_allocated),
                        "pnl_amount": _decimal(p.pnl_amount),
                    }
                    # Display list stays capped at 50 like before -- the
                    # full 500-row fetch above is only for the monthly
                    # summary's window math, not meant to render as a table.
                    for p in recent_closed[:50]
                ],
            }
        )
    finally:
        await client.close()


@app.get("/api/symbols")
async def symbols(_: None = Depends(_require_session)) -> JSONResponse:
    """Symbols with at least one closed BUY trade, for the dashboard's filter
    dropdown -- alongside each one's own win rate/trade count so the dropdown
    can show something useful without a second round trip per symbol."""
    client, config = _client()
    try:
        repository = TursoTradeRepository(client)
        all_trades = await repository.get_trades(None, config.candle_interval)
        by_symbol: dict[str, list] = {}
        for t in all_trades:
            if t.side.value == "buy" and t.status == "closed":
                by_symbol.setdefault(t.symbol, []).append(t)
        rows = []
        for symbol, trades_ in sorted(by_symbol.items()):
            wins = sum(1 for t in trades_ if t.pnl_percent is not None and t.pnl_percent > 0)
            rows.append(
                {
                    "symbol": symbol,
                    "trade_count": len(trades_),
                    "win_rate": _decimal(Decimal(100 * wins) / len(trades_)),
                }
            )
        return JSONResponse(rows)
    finally:
        await client.close()


@app.get("/api/trades")
async def trades(
    limit: int = 50, symbol: str | None = None, _: None = Depends(_require_session)
) -> JSONResponse:
    client, config = _client()
    try:
        repository = TursoTradeRepository(client)
        all_trades = await repository.get_trades(symbol, config.candle_interval)
        # Win rate stays BUY-only -- matches the paper-trading eligibility
        # gate exactly (see application/paper_trading.py). SELL trades are
        # never tradeable in the NSE cash market, so folding them into this
        # number would make it not match what eligibility actually uses.
        closed_buys = [
            t for t in all_trades if t.side.value == "buy" and t.status == "closed"
        ]
        # The visible table, though, shows both sides -- SELL rows are real
        # backtest results too (see application/backtest.py), just never
        # became real/paper positions; hiding them was the actual bug.
        closed_all = [t for t in all_trades if t.status == "closed"]
        closed_all.sort(key=lambda t: t.exit_timestamp or t.entry_timestamp, reverse=True)
        # A symbol-filtered view shows its full backtested history rather
        # than just the dashboard's default recent-N window.
        recent = closed_all if symbol else closed_all[:limit]
        wins = sum(1 for t in closed_buys if t.pnl_percent is not None and t.pnl_percent > 0)
        return JSONResponse(
            {
                "overall_win_rate": _decimal(
                    Decimal(100 * wins) / len(closed_buys) if closed_buys else None
                ),
                "closed_buy_count": len(closed_buys),
                "recent": [
                    {
                        "symbol": t.symbol,
                        "side": t.side.value,
                        "entry_timestamp": t.entry_timestamp.isoformat(),
                        "entry_price": _decimal(t.entry_price),
                        "exit_timestamp": t.exit_timestamp.isoformat() if t.exit_timestamp else None,
                        "exit_price": _decimal(t.exit_price),
                        "pnl_percent": _decimal(t.pnl_percent),
                    }
                    for t in recent
                ],
            }
        )
    finally:
        await client.close()


def _moneyness(option_type: str, strike: Decimal, underlying_price: Decimal) -> str:
    """ATM/ITM/OTM label for display -- the strike chosen is always the
    *nearest* one to the underlying's price at entry (see
    ``KiteDerivativesChain.nearest_atm_option``), so it's close to ATM by
    construction, but listed strikes are spaced in fixed increments (e.g.
    every 50 or 100 rupees), so the nearest one can still land a step into
    ITM or OTM territory -- this makes that visible instead of implying
    every trade is exactly at-the-money."""
    if strike == underlying_price:
        return "ATM"
    if option_type == "CE":
        return "ITM" if strike < underlying_price else "OTM"
    return "ITM" if strike > underlying_price else "OTM"


def _options_greeks_payload(trade) -> dict | None:
    """Implied volatility + delta/theta/gamma/vega at entry (and exit, if
    closed) for one options trade -- see ``application/
    options_analytics.py``. None if the underlying computation couldn't
    resolve (e.g. a stale/implausible stored premium) -- the row still
    renders, just without this extra detail, rather than breaking the
    whole derivatives tab over one bad historical row.
    """
    try:
        result = enrich_trade(
            trade.option_type,
            trade.strike,
            trade.expiry,
            trade.entry_timestamp,
            trade.underlying_price_at_entry,
            trade.entry_premium,
            trade.exit_timestamp,
            trade.underlying_price_at_exit,
            trade.exit_premium,
        )
    except Exception:
        logging.getLogger(__name__).warning(
            "Greeks computation failed for %s -- omitting from this row.",
            trade.option_tradingsymbol, exc_info=True,
        )
        return None
    return result


def _derivatives_summary_payload(options_trades: list, futures_trades: list) -> dict:
    """Shared by the live shadow-tracking endpoint and the current-month
    backtest endpoint -- same shape, different ``source`` filter upstream.

    One leg per signal (see ``application/signal_pipeline.py``'s
    ``_open_derivatives_shadow``): a futures position hedged by an option
    at the opposite delta (``primary_futures`` + ``hedge_options``).
    ``directional_options`` and ``hedge_futures`` are legacy/always empty
    going forward -- earlier versions of this feature also tracked a naked
    directional option (dropped after review) and, briefly, hedged that
    option with a future too (a mistake, reverted). Kept here only so any
    rows already written under those schemes don't silently vanish from the
    API shape.
    """

    def _options_summary(purpose: str) -> dict:
        closed = [t for t in options_trades if t.purpose == purpose and t.status == "closed"]
        wins = sum(1 for t in closed if t.pnl_percent is not None and t.pnl_percent > 0)
        return {
            "closed_count": len(closed),
            "win_rate": _decimal(Decimal(100 * wins) / len(closed)) if closed else None,
            "total_pnl": _decimal(sum((t.pnl_amount or Decimal("0") for t in closed), Decimal("0"))),
        }

    def _futures_summary(purpose: str) -> dict:
        closed = [t for t in futures_trades if t.purpose == purpose and t.status == "closed"]
        wins = sum(1 for t in closed if t.pnl_percent is not None and t.pnl_percent > 0)
        return {
            "closed_count": len(closed),
            "win_rate": _decimal(Decimal(100 * wins) / len(closed)) if closed else None,
            "total_pnl": _decimal(sum((t.pnl_amount or Decimal("0") for t in closed), Decimal("0"))),
        }

    return {
        "directional_options": _options_summary("directional"),
        "hedge_futures": _futures_summary("hedge"),
        "primary_futures": _futures_summary("primary"),
        "hedge_options": _options_summary("hedge"),
        "recent_options": [
            {
                "symbol": t.symbol,
                "option_type": t.option_type,
                "purpose": t.purpose,
                "tradingsymbol": t.option_tradingsymbol,
                "entry_timestamp": t.entry_timestamp.isoformat(),
                "underlying_price_at_entry": _decimal(t.underlying_price_at_entry),
                "moneyness": _moneyness(t.option_type, t.strike, t.underlying_price_at_entry),
                "strike": _decimal(t.strike),
                "entry_premium": _decimal(t.entry_premium),
                "exit_timestamp": t.exit_timestamp.isoformat() if t.exit_timestamp else None,
                "exit_premium": _decimal(t.exit_premium),
                "lot_size": t.lot_size,
                "pnl_percent": _decimal(t.pnl_percent),
                "pnl_amount": _decimal(t.pnl_amount),
                "status": t.status,
                "greeks": _options_greeks_payload(t),
            }
            for t in sorted(options_trades, key=lambda t: t.entry_timestamp, reverse=True)[:30]
        ],
        "recent_futures": [
            {
                "symbol": t.symbol,
                "side": t.side,
                "purpose": t.purpose,
                "tradingsymbol": t.futures_tradingsymbol,
                "entry_timestamp": t.entry_timestamp.isoformat(),
                "entry_price": _decimal(t.entry_price),
                "exit_timestamp": t.exit_timestamp.isoformat() if t.exit_timestamp else None,
                "exit_price": _decimal(t.exit_price),
                "lot_size": t.lot_size,
                "pnl_percent": _decimal(t.pnl_percent),
                "pnl_amount": _decimal(t.pnl_amount),
                "status": t.status,
            }
            for t in sorted(futures_trades, key=lambda t: t.entry_timestamp, reverse=True)[:30]
        ],
    }


@app.get("/api/derivatives-shadow")
async def derivatives_shadow(symbol: str | None = None, _: None = Depends(_require_session)) -> JSONResponse:
    """Live forward shadow-tracking summary (source='live') -- analysis
    only, never a real order (see application/options_shadow.py,
    futures_shadow.py). Current-month backtest results live separately at
    /api/derivatives-backtest so they never dilute this live win rate."""
    client, _config = _client()
    try:
        options_repository = TursoOptionsTradeRepository(client)
        futures_repository = TursoFuturesTradeRepository(client)
        options_trades = await options_repository.get_trades(symbol, source="live")
        futures_trades = await futures_repository.get_trades(symbol, source="live")
        return JSONResponse(_derivatives_summary_payload(options_trades, futures_trades))
    finally:
        await client.close()


@app.get("/api/derivatives-backtest")
async def derivatives_backtest(
    symbol: str | None = None, _: None = Depends(_require_session)
) -> JSONResponse:
    """Current-month options/futures backtest summary (source='backtest')
    -- see application/derivatives_backtest.py. Trigger a run via POST
    /api/trigger-backtest."""
    client, _config = _client()
    try:
        options_repository = TursoOptionsTradeRepository(client)
        futures_repository = TursoFuturesTradeRepository(client)
        options_trades = await options_repository.get_trades(symbol, source="backtest")
        futures_trades = await futures_repository.get_trades(symbol, source="backtest")
        return JSONResponse(_derivatives_summary_payload(options_trades, futures_trades))
    finally:
        await client.close()


@app.get("/api/margin-benefit")
async def margin_benefit(
    symbol: str, _: None = Depends(_require_admin)
) -> JSONResponse:
    """Live margin required for the symbol's open futures position + its
    hedge option vs. holding the future alone, using Kite's own
    basket-margin API -- not a guessed percentage (see
    ``KiteDerivativesChain.margin_benefit``'s docstring for why the naive
    version of this got the wrong answer at first). Requires an active
    Kite session and an open primary-future position for the symbol."""
    config = load_config()
    if not config.kite_api_key:
        raise HTTPException(status_code=400, detail="Kite is not configured.")
    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    try:
        kite_session_repository = TursoKiteSessionRepository(client)
        await kite_session_repository.ensure_schema()
        token_row = await kite_session_repository.get_token()
        if token_row is None:
            raise HTTPException(status_code=400, detail="No active Kite session.")
        access_token, _obtained_at = token_row

        futures_repository = TursoFuturesTradeRepository(client)
        options_repository = TursoOptionsTradeRepository(client)
        await futures_repository.ensure_schema()
        await options_repository.ensure_schema()
        primary_future = await futures_repository.get_open_trade(symbol, purpose="primary")
        if primary_future is None:
            raise HTTPException(
                status_code=404, detail=f"No open primary futures position for {symbol}."
            )
        hedge_option_type = "PE" if primary_future.side == "long" else "CE"
        hedge_option = await options_repository.get_open_trade(symbol, hedge_option_type, "hedge")

        kite = KiteConnect(api_key=config.kite_api_key)
        kite.set_access_token(access_token)
        derivatives_chain = KiteDerivativesChain(kite)
        legs = [(primary_future.futures_tradingsymbol, "BUY", primary_future.lot_size)]
        if hedge_option is not None:
            legs.append((hedge_option.option_tradingsymbol, "BUY", hedge_option.lot_size))
        result = await asyncio.to_thread(derivatives_chain.margin_benefit, legs)
        if result is None:
            raise HTTPException(status_code=502, detail="Kite margin lookup failed.")
        return JSONResponse(
            {
                "symbol": symbol,
                "has_hedge": hedge_option is not None,
                **result,
            }
        )
    finally:
        await client.close()


class ConfigUpdate(BaseModel):
    capital: str | None = None
    slots: str | None = None
    min_position: str | None = None


@app.get("/api/config")
async def get_config(_: None = Depends(_require_admin)) -> JSONResponse:
    return JSONResponse(
        {
            "capital": str(paper_trading.INITIAL_CAPITAL),
            "slots": paper_trading.TARGET_SLOTS,
            "min_position": str(paper_trading.MIN_POSITION_SIZE),
        }
    )


@app.post("/api/config")
async def update_config(update: ConfigUpdate, _: None = Depends(_require_admin)) -> JSONResponse:
    """Rewrite the relevant lines in .env. Takes effect on the *next* pipeline
    run/dashboard restart -- this process's own already-imported constants
    are not changed live, since paper_trading.py reads them once at import."""
    fields = {
        "capital": "TRADING_SCANNER_PAPER_CAPITAL",
        "slots": "TRADING_SCANNER_PAPER_SLOTS",
        "min_position": "TRADING_SCANNER_PAPER_MIN_POSITION",
    }
    updates: dict[str, str] = {}
    for field, env_key in fields.items():
        value = getattr(update, field)
        if value is None:
            continue
        try:
            Decimal(value)
        except InvalidOperation as error:
            raise HTTPException(status_code=400, detail=f"{field} must be numeric.") from error
        updates[env_key] = value
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")
    _write_env_updates(updates)
    return JSONResponse({"updated": updates, "note": "Takes effect on the next pipeline run."})


@app.post("/api/trigger")
async def trigger(_: None = Depends(_require_admin)) -> JSONResponse:
    """Kick off one manual pipeline run in the background, same command cron uses."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = _LOG_PATH.open("a")
    subprocess.Popen(
        [sys.executable, "-m", "trading_scanner.signals"],
        cwd=str(_REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )
    return JSONResponse({"triggered": True})


@app.post("/api/trigger-backtest")
async def trigger_backtest(_: None = Depends(_require_admin)) -> JSONResponse:
    """Kick off one manual current-month derivatives backtest in the
    background (see application/derivatives_backtest.py). Requires an
    active Kite session (uses historical data, not live LTP)."""
    _BACKTEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = _BACKTEST_LOG_PATH.open("a")
    subprocess.Popen(
        [sys.executable, "-m", "trading_scanner.derivatives_backtest_cli"],
        cwd=str(_REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )
    return JSONResponse({"triggered": True})


@app.get("/api/logs")
async def logs(lines: int = 200, _: None = Depends(_require_session)) -> JSONResponse:
    if not _LOG_PATH.exists():
        return JSONResponse({"lines": []})
    tail = subprocess.run(
        ["tail", "-n", str(lines), str(_LOG_PATH)], capture_output=True, text=True
    )
    return JSONResponse({"lines": tail.stdout.splitlines()})


def _last_run_summary() -> dict | None:
    if not _LOG_PATH.exists():
        return None
    result = subprocess.run(
        ["grep", "-n", "Signal pipeline \\(started\\|finished\\)", str(_LOG_PATH)],
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    return {
        "status": "finished" if "finished" in last_line else "started",
        "raw": last_line.split(":", 1)[-1].strip() if ":" in last_line else last_line,
    }


def _write_env_updates(updates: dict[str, str]) -> None:
    existing_lines = _ENV_PATH.read_text(encoding="utf-8").splitlines() if _ENV_PATH.exists() else []
    remaining = dict(updates)
    new_lines: list[str] = []
    for line in existing_lines:
        key = line.split("=", 1)[0] if "=" in line else None
        if key in remaining:
            new_lines.append(f"{key}={remaining.pop(key)}")
        else:
            new_lines.append(line)
    for key, value in remaining.items():
        new_lines.append(f"{key}={value}")
    _ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_PAGE = (_TEMPLATES_DIR / "dashboard.html").read_text(encoding="utf-8")

_LOGIN_PAGE = (_TEMPLATES_DIR / "login.html").read_text(encoding="utf-8")


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("TRADING_SCANNER_DASHBOARD_PORT", "8000")))


if __name__ == "__main__":
    main()

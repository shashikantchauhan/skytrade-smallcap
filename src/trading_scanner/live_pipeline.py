"""Always-on live pipeline: builds candles from Kite's WebSocket tick feed
and evaluates signals the moment each hourly bar closes, replacing the
hourly-cron/Historical-Data-API pipeline (``signals.py``) as the source of
truth during market hours.

Why this exists: Zerodha's own guidance (confirmed live and via their dev
forum, see ``application/signal_pipeline.py``'s ``_evaluate_symbol``
docstring) is that the Historical Data API isn't meant for live signals --
it can lag the current session's candles by hours. Their documented fix:
build your own candles from the WebSocket tick feed. That's what this
module does. ``signals.py``'s download-based pipeline still exists, for
backfill, catch-up after downtime, and the dashboard's manual trigger --
just no longer the thing driving live trading during market hours.

Run as a systemd service (``p-trade-live``), not cron -- it needs to stay
connected continuously through the trading session, not spin up once an
hour. See ``deploy/p-trade-live.service``.

KiteTicker (pykiteconnect) is not asyncio-native -- it runs its own
Tornado-based event loop in a background thread (``connect(threaded=True)``)
and delivers ticks via a callback on that thread. The bridge to this
module's asyncio code is a plain ``queue.Queue``: the tick callback just
enqueues (cheap, thread-safe), and an asyncio task drains it.
"""

import asyncio
import logging
import queue
from datetime import UTC, datetime
from decimal import Decimal

from kiteconnect import KiteConnect, KiteTicker

from trading_scanner.alpha_engine import AlphaEngine
from trading_scanner.application.futures_trading import FUTURES_INITIAL_CAPITAL
from trading_scanner.application.paper_trading import (
    INITIAL_CAPITAL,
    stop_loss_price,
    trailing_stop_price,
)
from trading_scanner.application.signal_pipeline import (
    _ENGINE_SETTINGS,
    _close_paper_position,
    _collect_and_open_ranked_positions,
    _evaluate_from_stored_candles,
    _notify_kite_expired_once_per_day,
    _process_symbol,
)
from trading_scanner.application.symbols import SymbolLoader, SymbolLoadError
from trading_scanner.config.settings import AppConfig, load_config
from trading_scanner.domain.models import Candle
from trading_scanner.infrastructure.db import (
    TursoCandleRepository,
    TursoEngineStateRepository,
    TursoFuturesPaperAccountRepository,
    TursoFuturesTradeRepository,
    TursoKiteSessionRepository,
    TursoLiveOrderRepository,
    TursoOptionsTradeRepository,
    TursoPaperAccountRepository,
    TursoSignalRepository,
    TursoTradeRepository,
    create_turso_client,
)
from trading_scanner.infrastructure.kite import (
    KiteDerivativesChain,
    KiteInstrumentMap,
    KiteOrderExecutor,
)
from trading_scanner.infrastructure.kite_ticker import (
    CandleAggregator,
    bucket_start,
    is_market_hours,
)
from trading_scanner.infrastructure.telegram import LoggingNotifier, TelegramNotifier

logger = logging.getLogger(__name__)

# How often the boundary-check loop wakes up to see whether the current
# hourly bucket has closed. 10s is frequent enough that a bar is finalized
# within seconds of closing, not expensive enough to matter against an
# hour-long bucket.
_BOUNDARY_CHECK_SECONDS = 10

# How often to check whether a newer Kite access token has appeared (e.g.
# the admin re-logged in after a token expired) -- lets this long-running
# process pick up a fresh login without a manual restart.
_TOKEN_REFRESH_CHECK_SECONDS = 120

# Symbols processed concurrently when a bucket closes -- mirrors
# signal_pipeline.py's own concurrency bound for the same reason (network/
# DB I/O bound, not CPU bound).
_MAX_CONCURRENT_SYMBOLS = 12

# How often the tick-level stop-loss's open-positions cache is refreshed
# from the DB -- see _open_positions_cache's docstring. A few seconds of
# staleness is harmless; querying on every tick would not be.
_POSITIONS_CACHE_REFRESH_SECONDS = 5

# 2026-08-17: found by hand after a real 4+ hour outage on the parent
# deployment -- a stale daily token made KiteTicker fail its own reconnect
# loop with repeated 403s, exhaust its retries ("supervisor will recreate
# it" -- nothing actually does), and then _token_refresh_loop's in-place
# disconnect+reconnect *also* silently failed to restore any tick flow
# (logged "reconnecting", delivered zero ticks or callbacks of any kind
# for 2.5+ hours after). Only a full process restart (a fresh KiteTicker
# from run_forever's own cold-start path) actually recovered. This
# watchdog is the guaranteed fallback for exactly that: regardless of
# *why* the ticker went dead, if no tick or (re)connect has landed in
# this long during market hours, raise out of the gather in run_forever
# so its outer while loop tears down and reconnects from scratch -- the
# one path proven to work, without needing an actual systemd/process
# restart. Ported here 2026-08-18 alongside the same outage's other fix
# (_run_until_first_exit below).
_TICKER_STALE_SECONDS = 600
_TICKER_WATCHDOG_CHECK_SECONDS = 60


def _build_notifier(config: AppConfig):
    if config.telegram_bot_token and config.telegram_chat_id:
        return TelegramNotifier(config.telegram_bot_token, config.telegram_chat_id)
    return LoggingNotifier()


class LiveTickerPipeline:
    """Owns the KiteTicker connection, tick-to-candle aggregation, and
    firing signal evaluation on each bucket close. One instance per
    process; ``run_forever()`` is the entry point and does not return under
    normal operation."""

    def __init__(self, config: AppConfig, symbols: list[str]) -> None:
        self._config = config
        self._symbols = symbols
        self._engine = AlphaEngine(**_ENGINE_SETTINGS)
        self._tick_queue: queue.Queue = queue.Queue()
        self._aggregators: dict[int, CandleAggregator] = {}
        self._token_to_symbol: dict[int, str] = {}
        self._current_bucket: datetime | None = None
        self._ticker: KiteTicker | None = None
        self._access_token: str | None = None
        self._paper_account_lock = asyncio.Lock()
        self._client = None
        self._repos: dict = {}
        self._notifier = None
        # symbol -> entry_price for every currently-open paper position.
        # Refreshed from the DB every _POSITIONS_CACHE_REFRESH_SECONDS
        # (see _refresh_positions_cache_loop) rather than on every tick --
        # a DB round trip per tick would be far too slow across 220
        # symbols. Staleness is bounded and harmless: close_position() is
        # idempotent (returns None if nothing's open), so a stale entry
        # here only risks one redundant, no-op close attempt, never a
        # double-close or double-charge.
        self._open_positions_cache: dict[str, Decimal] = {}
        # symbol -> highest price seen since entry, for the trailing stop
        # (see application/paper_trading.py's trailing_stop_price). Same
        # staleness tolerance as _open_positions_cache above -- refreshed
        # alongside it, and additionally bumped in-memory (+ persisted)
        # the moment a tick makes a new high, so the trail tightens in
        # real time rather than only every _POSITIONS_CACHE_REFRESH_SECONDS.
        self._peak_price_cache: dict[str, Decimal] = {}
        # Set for real in _setup_repositories (needs the config's symbols
        # file loaded) -- empty here just means "not set up yet."
        self._futures_paper_symbols: frozenset[str] = frozenset()
        # Last time a tick arrived or the ticker (re)connected -- see
        # _ticker_watchdog_loop and _TICKER_STALE_SECONDS above.
        self._last_tick_at: datetime = datetime.now(UTC)

    async def _setup_repositories(self) -> None:
        if not self._config.turso_database_url:
            raise RuntimeError("TRADING_SCANNER_TURSO_URL is required.")
        self._client = create_turso_client(
            self._config.turso_database_url, self._config.turso_auth_token
        )
        self._repos = {
            "candle": TursoCandleRepository(self._client),
            "signal": TursoSignalRepository(self._client),
            "engine_state": TursoEngineStateRepository(self._client),
            "trade": TursoTradeRepository(self._client),
            "paper_account": TursoPaperAccountRepository(self._client, INITIAL_CAPITAL),
            "kite_session": TursoKiteSessionRepository(self._client),
            "options_trade": TursoOptionsTradeRepository(self._client),
            "futures_trade": TursoFuturesTradeRepository(self._client),
            "live_order": TursoLiveOrderRepository(self._client),
            "futures_paper_account": TursoFuturesPaperAccountRepository(
                self._client, FUTURES_INITIAL_CAPITAL
            ),
        }
        for repo in self._repos.values():
            await repo.ensure_schema()
        self._notifier = _build_notifier(self._config)
        # Real, capital-gated futures paper account -- restricted to this
        # allowlist (Nifty50 by default), not the full symbol universe. See
        # AppConfig.futures_paper_symbols_file / application/futures_trading.py.
        try:
            self._futures_paper_symbols = frozenset(
                SymbolLoader().load(self._config.futures_paper_symbols_file)
            )
        except SymbolLoadError:
            logger.warning(
                "Futures paper symbols file missing/empty (%s) -- futures paper "
                "trading is off this run.",
                self._config.futures_paper_symbols_file,
            )
            self._futures_paper_symbols = frozenset()

    async def _resolve_instrument_tokens(self, kite: KiteConnect) -> None:
        instrument_map = KiteInstrumentMap(kite)
        self._token_to_symbol = {}
        for symbol in self._symbols:
            try:
                token = await asyncio.to_thread(instrument_map.resolve, symbol)
                self._token_to_symbol[token] = symbol
                self._aggregators[token] = CandleAggregator()
            except Exception:
                logger.warning("Could not resolve instrument token for %s -- skipping.", symbol)
        logger.info(
            "Resolved %d/%d symbols to instrument tokens.",
            len(self._token_to_symbol), len(self._symbols),
        )

    async def _get_access_token(self) -> str | None:
        token_row = await self._repos["kite_session"].get_token()
        return token_row[0] if token_row else None

    def _on_ticks(self, ws, ticks) -> None:  # noqa: ANN001 -- kiteconnect's own callback signature
        self._last_tick_at = datetime.now(UTC)
        self._tick_queue.put(ticks)

    def _on_connect(self, ws, response) -> None:  # noqa: ANN001
        # Reset the watchdog clock here too, not just on ticks -- a fresh
        # connect can legitimately take a few seconds before the first tick
        # lands, and this must not look stale in that gap.
        self._last_tick_at = datetime.now(UTC)
        tokens = list(self._token_to_symbol.keys())
        logger.info("KiteTicker connected -- subscribing to %d instruments.", len(tokens))
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)

    def _on_close(self, ws, code, reason) -> None:  # noqa: ANN001
        logger.warning("KiteTicker closed: %s %s", code, reason)

    def _on_error(self, ws, code, reason) -> None:  # noqa: ANN001
        logger.error("KiteTicker error: %s %s", code, reason)

    def _on_reconnect(self, ws, attempts_count) -> None:  # noqa: ANN001
        logger.warning("KiteTicker reconnecting (attempt %d)...", attempts_count)

    def _on_noreconnect(self, ws) -> None:  # noqa: ANN001
        logger.error("KiteTicker exhausted reconnect attempts -- supervisor will recreate it.")

    def _connect_ticker(self, access_token: str) -> None:
        self._access_token = access_token
        ticker = KiteTicker(api_key=self._config.kite_api_key, access_token=access_token)
        ticker.on_ticks = self._on_ticks
        ticker.on_connect = self._on_connect
        ticker.on_close = self._on_close
        ticker.on_error = self._on_error
        ticker.on_reconnect = self._on_reconnect
        ticker.on_noreconnect = self._on_noreconnect
        ticker.connect(threaded=True)
        self._ticker = ticker

    def _disconnect_ticker(self) -> None:
        if self._ticker is not None:
            try:
                self._ticker.close()
            except Exception:
                logger.warning("Error closing KiteTicker (ignoring, moving on).", exc_info=True)
            self._ticker = None

    async def _drain_ticks_loop(self) -> None:
        """Consumes ticks pushed by the (threaded) KiteTicker callback and
        feeds them into the right symbol's aggregator. ``queue.Queue.get``
        is blocking, so it runs via ``asyncio.to_thread`` rather than
        polling/spinning. Also checks each tick against the open-positions
        cache for a stop-loss breach -- this is what makes the stop
        tick-level rather than only checked at the hourly bucket close."""
        while True:
            ticks = await asyncio.to_thread(self._tick_queue.get)
            for tick in ticks:
                token = tick.get("instrument_token")
                aggregator = self._aggregators.get(token)
                if aggregator is None:
                    continue
                price = tick.get("last_price")
                if price is None:
                    continue
                volume = tick.get("volume_traded", 0) or 0
                price_decimal = Decimal(str(price))
                aggregator.add_tick(datetime.now(UTC), price_decimal, int(volume))
                await self._check_stop_loss(token, price_decimal)

    async def _check_stop_loss(self, token: int, price: Decimal) -> None:
        """Tick-level price checks for an open position, in order: hard
        stop-loss first (unconditional downside cap), then the trailing
        stop (only once activated -- see ``trailing_stop_price``'s
        docstring for why a high activation threshold is what makes this
        help rather than hurt)."""
        symbol = self._token_to_symbol.get(token)
        if symbol is None:
            return
        entry_price = self._open_positions_cache.get(symbol)
        if entry_price is None:
            return

        if price <= stop_loss_price(entry_price):
            self._close_symbol_caches(symbol)
            logger.warning(
                "Stop-loss breached for %s: price=%s <= stop=%s (entry=%s) -- force-closing.",
                symbol, price, stop_loss_price(entry_price), entry_price,
            )
            await _close_paper_position(
                symbol, datetime.now(UTC), price,
                self._repos["paper_account"], self._repos["signal"], self._notifier,
                self._paper_account_lock,
            )
            return

        peak_price = self._peak_price_cache.get(symbol, entry_price)
        if price > peak_price:
            peak_price = price
            self._peak_price_cache[symbol] = peak_price
            # Only a DB write when price genuinely makes a new high, not
            # every tick -- infrequent in practice, and this is what makes
            # the trail survive a mid-day process restart instead of
            # resetting to entry_price.
            await self._repos["paper_account"].update_peak_price(symbol, peak_price)

        trail_stop = trailing_stop_price(entry_price, peak_price)
        if trail_stop is not None and price <= trail_stop:
            self._close_symbol_caches(symbol)
            logger.warning(
                "Trailing stop breached for %s: price=%s <= trail=%s (peak=%s, entry=%s) -- "
                "force-closing.",
                symbol, price, trail_stop, peak_price, entry_price,
            )
            await _close_paper_position(
                symbol, datetime.now(UTC), price,
                self._repos["paper_account"], self._repos["signal"], self._notifier,
                self._paper_account_lock,
            )

    def _close_symbol_caches(self, symbol: str) -> None:
        # Drop from both caches immediately -- a burst of ticks can arrive
        # for the same symbol before the next cache refresh, and neither
        # check must fire more than once for the same position.
        self._open_positions_cache.pop(symbol, None)
        self._peak_price_cache.pop(symbol, None)

    async def _refresh_positions_cache_loop(self) -> None:
        """Keeps ``_open_positions_cache`` in sync with the DB every few
        seconds -- not on every tick (see ``_open_positions_cache``'s own
        docstring for why), and not hooked into every individual
        open/close call site (which would be fragile -- easy to miss one).
        Polling the DB for ground truth is simpler and safe given
        ``close_position``'s idempotency."""
        while True:
            try:
                positions = await self._repos["paper_account"].get_open_positions()
                self._open_positions_cache = {p.symbol: p.entry_price for p in positions}
                self._peak_price_cache = {
                    p.symbol: (p.peak_price if p.peak_price is not None else p.entry_price)
                    for p in positions
                }
            except Exception:
                logger.warning(
                    "Failed to refresh open-positions cache (will retry).", exc_info=True
                )
            await asyncio.sleep(_POSITIONS_CACHE_REFRESH_SECONDS)

    async def _boundary_loop(self) -> None:
        """Wakes up periodically; when the current hourly bucket has
        rolled over, finalizes it into real Candle rows and fires signal
        evaluation for every symbol that had ticks."""
        while True:
            await asyncio.sleep(_BOUNDARY_CHECK_SECONDS)
            now = datetime.now(UTC)
            new_bucket = bucket_start(now)
            if self._current_bucket is None:
                self._current_bucket = new_bucket
                for aggregator in self._aggregators.values():
                    aggregator.start(new_bucket)
                continue
            if new_bucket > self._current_bucket:
                closed_bucket = self._current_bucket
                await self._finalize_bucket(closed_bucket)
                self._current_bucket = new_bucket
                for aggregator in self._aggregators.values():
                    aggregator.start(new_bucket)

    async def _finalize_bucket(self, bucket: datetime) -> None:
        candles: dict[str, Candle] = {}
        for token, aggregator in self._aggregators.items():
            ohlcv = aggregator.finalize()
            if ohlcv is None:
                continue
            symbol = self._token_to_symbol[token]
            open_, high, low, close, volume = ohlcv
            candles[symbol] = Candle(
                symbol=symbol, timestamp=bucket,
                open=open_, high=high, low=low, close=close, volume=volume,
            )
        if not candles:
            logger.info(
                "Bucket %s closed with no ticks for any symbol -- nothing to evaluate.", bucket
            )
            return
        logger.info(
            "Bucket %s closed: %d/%d symbols had ticks. Evaluating...",
            bucket, len(candles), len(self._aggregators),
        )
        await self._process_closed_candles(candles)

    async def _process_closed_candles(self, candles: dict[str, Candle]) -> None:
        kite = KiteConnect(api_key=self._config.kite_api_key)
        kite.set_access_token(self._access_token)
        derivatives_chain = KiteDerivativesChain(kite)
        order_executor = KiteOrderExecutor(kite)

        index_result = None
        if self._config.index_symbol and self._config.index_symbol in candles:
            index_candle = candles[self._config.index_symbol]
            await self._repos["candle"].upsert_candles(
                self._config.index_symbol, self._config.candle_interval, [index_candle]
            )
            evaluated = await _evaluate_from_stored_candles(
                self._config.index_symbol, self._config, self._engine,
                self._repos["candle"], self._repos["engine_state"],
            )
            index_result = evaluated[0] if evaluated is not None else None

        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SYMBOLS)

        # Every symbol whose hourly bucket just closed lands in this same
        # batch (``candles``) -- exactly the "one scan cycle" ranking
        # needs a full set of candidates for (see
        # application/signal_pipeline.py's _collect_and_open_ranked_positions).
        # So this is now two passes, same shape as the cron path
        # (run_signal_pipeline): (1) evaluate every symbol concurrently,
        # no trade/paper/futures writes yet; (2) rank this batch's BUY
        # candidates (cash) and BUY+SELL candidates (futures, Nifty50-only)
        # and open positions for them strongest-first; (3) run the rest of
        # each symbol's bookkeeping/notifications concurrently, with both
        # outcomes already decided so they aren't redecided per-symbol,
        # unranked, first-come-first-served like before.
        evaluated_by_symbol: dict[str, tuple] = {}

        async def _evaluate_one(symbol: str, candle: Candle) -> None:
            async with semaphore:
                try:
                    await self._repos["candle"].upsert_candles(
                        symbol, self._config.candle_interval, [candle]
                    )
                    evaluated_by_symbol[symbol] = await _evaluate_from_stored_candles(
                        symbol, self._config, self._engine,
                        self._repos["candle"], self._repos["engine_state"],
                    )
                except Exception:
                    logger.exception("Unexpected exception evaluating closed candle for %s", symbol)
                    evaluated_by_symbol[symbol] = None

        await asyncio.gather(*(
            _evaluate_one(symbol, candle)
            for symbol, candle in candles.items()
            if symbol != self._config.index_symbol
        ))

        paper_notes, futures_notes = await _collect_and_open_ranked_positions(
            evaluated_by_symbol, self._config, self._repos["trade"], self._repos["paper_account"],
            self._paper_account_lock, derivatives_chain, self._repos["futures_paper_account"],
            self._futures_paper_symbols,
        )

        async def _process_one(symbol: str) -> None:
            evaluated = evaluated_by_symbol.get(symbol)
            if evaluated is None:
                return
            async with semaphore:
                try:
                    await _process_symbol(
                        symbol, self._config, None, self._engine,
                        self._repos["candle"], self._repos["signal"], self._repos["engine_state"],
                        self._repos["trade"], self._repos["paper_account"], self._notifier,
                        index_result, self._paper_account_lock,
                        derivatives_chain,
                        self._repos["options_trade"], self._repos["futures_trade"],
                        precomputed_evaluation=evaluated,
                        order_executor=order_executor,
                        live_order_repository=self._repos["live_order"],
                        precomputed_paper_note=paper_notes.get(symbol),
                        futures_account_repository=self._repos["futures_paper_account"],
                        futures_paper_symbols=self._futures_paper_symbols,
                        precomputed_futures_note=futures_notes.get(symbol),
                    )
                except Exception:
                    logger.exception("Unexpected exception processing closed candle for %s", symbol)

        await asyncio.gather(*(
            _process_one(symbol)
            for symbol in candles
            if symbol != self._config.index_symbol
        ))

    async def _token_refresh_loop(self) -> None:
        """Detects a newer Kite login (the admin re-logging in via the
        dashboard) and reconnects the ticker with it, so this long-running
        process never needs a manual restart to pick up the daily token."""
        while True:
            await asyncio.sleep(_TOKEN_REFRESH_CHECK_SECONDS)
            try:
                token = await self._get_access_token()
            except Exception:
                logger.warning(
                    "Token refresh check failed (Turso hiccup?) -- will retry.", exc_info=True
                )
                continue
            if token and token != self._access_token:
                logger.info("Detected a new Kite access token -- reconnecting ticker.")
                self._disconnect_ticker()
                self._connect_ticker(token)

    async def _ticker_watchdog_loop(self) -> None:
        """Guaranteed fallback if the ticker goes dead and stays dead --
        see _TICKER_STALE_SECONDS above for the 2026-08-17 incident this
        exists for. Deliberately doesn't try to fix the ticker itself
        (_connect_ticker/_disconnect_ticker's in-place hot-swap is exactly
        what failed silently that day); it just raises, which propagates
        out of run_forever's gather and into _run()'s own try/except,
        which already does the one thing proven to work: tear this
        pipeline instance down and call run_forever() again from scratch,
        including a brand new KiteTicker.

        Also fixes a second gap from the same incident: this run never got
        a Telegram alert that the Kite session needed attention, because
        _notify_kite_expired_once_per_day was only ever wired into the old
        download-based run_signal_pipeline() path, not this always-on one
        -- so staleness here was invisible except in the server's own log.
        Reused directly (same once-per-calendar-day dedup via
        kite_session_repository.expiry_notified_date) rather than
        reinventing it."""
        while True:
            await asyncio.sleep(_TICKER_WATCHDOG_CHECK_SECONDS)
            if not is_market_hours(datetime.now(UTC)):
                continue
            stale_for = (datetime.now(UTC) - self._last_tick_at).total_seconds()
            if stale_for > _TICKER_STALE_SECONDS:
                try:
                    await _notify_kite_expired_once_per_day(
                        self._repos["kite_session"], self._notifier
                    )
                except Exception:
                    logger.exception("Failed to send ticker-stale notification")
                raise RuntimeError(
                    f"KiteTicker appears dead: no tick or (re)connect in "
                    f"{stale_for:.0f}s during market hours -- forcing a full reconnect."
                )

    async def run_forever(self) -> None:
        await self._setup_repositories()
        try:
            while True:
                now = datetime.now(UTC)
                if not is_market_hours(now):
                    logger.info("Outside market hours -- sleeping.")
                    await asyncio.sleep(60)
                    continue

                access_token = await self._get_access_token()
                if not access_token:
                    logger.warning("No Kite session yet -- waiting for admin login.")
                    await asyncio.sleep(60)
                    continue

                kite = KiteConnect(api_key=self._config.kite_api_key)
                kite.set_access_token(access_token)
                try:
                    await self._resolve_instrument_tokens(kite)
                except Exception:
                    logger.exception("Failed to resolve instrument tokens -- retrying shortly.")
                    await asyncio.sleep(60)
                    continue
                if not self._token_to_symbol:
                    logger.error(
                        "No symbols resolved to instrument tokens -- nothing to trade. "
                        "Retrying shortly."
                    )
                    await asyncio.sleep(60)
                    continue

                # Prime the stop-loss cache immediately -- a mid-day
                # restart (token refresh, deploy, crash-recover) must not
                # leave already-open positions unprotected until the first
                # periodic refresh fires.
                try:
                    positions = await self._repos["paper_account"].get_open_positions()
                    self._open_positions_cache = {p.symbol: p.entry_price for p in positions}
                    self._peak_price_cache = {
                        p.symbol: (p.peak_price if p.peak_price is not None else p.entry_price)
                        for p in positions
                    }
                except Exception:
                    logger.warning(
                        "Failed to prime open-positions cache on startup.", exc_info=True
                    )

                self._connect_ticker(access_token)
                self._current_bucket = None
                # Reset here too (not just in _on_connect) so a freshly
                # entered loop iteration always gets the watchdog's full
                # _TICKER_STALE_SECONDS grace period, even if _on_connect's
                # own callback is delayed or never fires at all.
                self._last_tick_at = datetime.now(UTC)
                try:
                    await _run_until_first_exit((
                        self._drain_ticks_loop(),
                        self._boundary_loop(),
                        self._token_refresh_loop(),
                        self._refresh_positions_cache_loop(),
                        self._ticker_watchdog_loop(),
                        self._run_until_market_close(),
                    ))
                finally:
                    self._disconnect_ticker()
        finally:
            if self._client is not None:
                await self._client.close()

    async def _run_until_market_close(self) -> None:
        """Sleeps until market close, then raises to unwind the other
        loops (gather) cleanly and let ``run_forever``'s outer loop go back
        to sleeping until tomorrow's open."""
        while is_market_hours(datetime.now(UTC)):
            await asyncio.sleep(30)
        raise _MarketClosed


class _MarketClosed(Exception):
    """Internal control-flow signal -- market hours ended, unwind this
    session's loops and let run_forever's outer loop take over."""


async def _run_until_first_exit(coros) -> None:
    """Run several long-running loops concurrently; the moment ANY one of
    them returns or raises, cancel the rest and propagate that one's
    outcome (raising if it raised).

    2026-08-18: replaces a plain ``asyncio.gather(*coros)`` here, found on
    the parent deployment to be responsible for a real ~2 hour silent
    outage. ``gather()`` without ``return_exceptions=True`` propagates the
    FIRST exception to its caller the moment one coroutine raises -- but
    it does NOT cancel the other still-running coroutines; they keep
    executing as orphaned background tasks the caller no longer has any
    reference to or control over. Every one of ``run_forever``'s crash-
    and-retry cycles was therefore leaking its previous cycle's other
    loops (ticker drain, boundary, token refresh, positions-cache
    refresh, watchdog) instead of tearing them down -- confirmed live on
    the parent: after 5 crashes in ~40 minutes, the process had
    accumulated that many orphaned copies of each loop, all mutating the
    same ``self`` (ticker, caches, DB client/repos) concurrently and
    unpredictably, which is what actually produced the frozen-but-not-
    crashed state systemd saw as healthy (near-zero CPU, sockets stuck in
    CLOSE-WAIT) -- including the watchdog itself being one of the leaked,
    no-longer-effective copies.
    """
    tasks = [asyncio.ensure_future(coro) for coro in coros]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    # Surface the first-finished task's outcome -- if it raised, re-raise
    # here so run_forever's own try/finally (ticker disconnect) and _run's
    # except clauses see the same exception they always have.
    (first,) = done
    first.result()


async def _run(config: AppConfig) -> None:
    symbols = SymbolLoader().load(config.symbols_file)
    logger.info("Loaded %d symbols for live ticker pipeline", len(symbols))
    while True:
        # A fresh LiveTickerPipeline every cycle, not one reused instance
        # -- belt-and-suspenders alongside _run_until_first_exit above:
        # even if some future loop leaks a task, it can no longer share
        # mutable state (ticker, caches, DB client) with the new cycle's
        # instance.
        pipeline = LiveTickerPipeline(config, symbols)
        try:
            await pipeline.run_forever()
        except _MarketClosed:
            logger.info("Market closed -- pipeline will resume tomorrow.")
            await asyncio.sleep(60)
        except Exception:
            logger.exception("Live pipeline crashed -- restarting in 30s.")
            await asyncio.sleep(30)


def main() -> None:
    config = load_config()
    logging.basicConfig(level=config.logging_level, format="%(asctime)s %(levelname)s: %(message)s")
    if not config.kite_api_key:
        logger.error("TRADING_SCANNER_KITE_API_KEY is required for the live ticker pipeline.")
        return
    logger.info("Live ticker pipeline starting")
    try:
        asyncio.run(_run(config))
    except SymbolLoadError as error:
        logger.error("%s", error)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Live ticker pipeline stopped")


if __name__ == "__main__":
    main()

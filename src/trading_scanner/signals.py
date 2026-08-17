"""Command-line entry point for the hourly multi-symbol signal pipeline."""

import asyncio
import logging

from trading_scanner.application.paper_trading import INITIAL_CAPITAL
from trading_scanner.application.signal_pipeline import run_signal_pipeline
from trading_scanner.application.symbols import SymbolLoader, SymbolLoadError
from trading_scanner.config.settings import AppConfig, load_config
from trading_scanner.infrastructure.db import (
    TursoCandleRepository,
    TursoEngineStateRepository,
    TursoFuturesTradeRepository,
    TursoKiteSessionRepository,
    TursoLiveOrderRepository,
    TursoOptionsTradeRepository,
    TursoPaperAccountRepository,
    TursoSignalRepository,
    TursoTradeRepository,
    create_turso_client,
)
from trading_scanner.infrastructure.telegram import LoggingNotifier, TelegramNotifier


def _build_notifier(config: AppConfig):
    """Use Telegram when configured; otherwise fall back to console logging."""
    if config.telegram_bot_token and config.telegram_chat_id:
        return TelegramNotifier(
            config.telegram_bot_token, config.telegram_chat_id, config.notification_label
        )
    return LoggingNotifier()


async def _run(config: AppConfig) -> None:
    logger = logging.getLogger(__name__)
    if not config.turso_database_url:
        raise RuntimeError(
            "TRADING_SCANNER_TURSO_URL is required to run the signal pipeline."
        )

    symbols = SymbolLoader().load(config.symbols_file)
    logger.info("Loaded %d symbols", len(symbols))

    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    try:
        candle_repository = TursoCandleRepository(client)
        signal_repository = TursoSignalRepository(client)
        engine_state_repository = TursoEngineStateRepository(client)
        trade_repository = TursoTradeRepository(client)
        paper_account_repository = TursoPaperAccountRepository(client, INITIAL_CAPITAL)
        kite_session_repository = TursoKiteSessionRepository(client)
        options_trade_repository = TursoOptionsTradeRepository(client)
        futures_trade_repository = TursoFuturesTradeRepository(client)
        live_order_repository = TursoLiveOrderRepository(client)
        await candle_repository.ensure_schema()
        await signal_repository.ensure_schema()
        await engine_state_repository.ensure_schema()
        await trade_repository.ensure_schema()
        await paper_account_repository.ensure_schema()
        await kite_session_repository.ensure_schema()
        await options_trade_repository.ensure_schema()
        await futures_trade_repository.ensure_schema()
        await live_order_repository.ensure_schema()

        notifier = _build_notifier(config)
        await run_signal_pipeline(
            config,
            symbols,
            candle_repository,
            signal_repository,
            engine_state_repository,
            trade_repository,
            paper_account_repository,
            notifier,
            kite_session_repository,
            options_trade_repository,
            futures_trade_repository,
            live_order_repository,
        )
    finally:
        await client.close()


def main() -> None:
    """Load configuration and run one hourly ingestion + signal pass."""
    config = load_config()
    logging.basicConfig(level=config.logging_level, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Signal pipeline started")

    try:
        asyncio.run(_run(config))
    except SymbolLoadError as error:
        logger.error("%s", error)
    except Exception:
        logger.exception("Unexpected exception during signal pipeline")
    finally:
        logger.info("Signal pipeline finished")


if __name__ == "__main__":
    main()

"""One-shot current-month options/futures backtest.

Run on demand (a dashboard button), not on the hourly pipeline schedule.
Replays this month's already-*closed* equity trades (from ``trades``, the
same table ``paper_trading.py`` scores) against the options/futures
contracts that would have shadowed each one, using Kite's historical data
API instead of live LTP (see ``KiteDerivativesChain.historical_premium``).

Only the current month is reliable: Kite's NFO instrument dump only lists
currently-live contracts (see ``infrastructure/kite.py``'s
``KiteDerivativesChain`` docstring) -- once a contract expires it drops out
of that dump and its ``instrument_token`` can no longer be resolved, so
earlier months can't be backtested this way. Trades whose entry falls
before the current calendar month are skipped.

Every inserted row is tagged ``source="backtest"`` (see
``domain.models.OptionsShadowTrade``/``FuturesShadowTrade``) so it never
mixes into the forward shadow-tracking win-rate stats
``signal_pipeline.py`` accumulates live (``source="live"``).

Same entry rules and PnL math as ``options_shadow.py``/``futures_shadow.py``
(futures long/short, hedged by an option at the opposite delta), deliberately
not imported from there -- this replays whole already-closed trades in one
shot rather than reacting to one signal at a time, so the two don't share a
code path.
"""

import logging
from datetime import date, datetime
from decimal import Decimal

from trading_scanner.domain.models import FuturesShadowTrade, OptionsShadowTrade, SignalSide, Trade
from trading_scanner.infrastructure.db import (
    TursoFuturesTradeRepository,
    TursoOptionsTradeRepository,
    TursoTradeRepository,
)
from trading_scanner.infrastructure.kite import KiteDerivativesChain

logger = logging.getLogger(__name__)

# Mirrors signal_pipeline.py's _HEDGE_OTM_PCT -- the hedge option leg
# targets a strike this far OTM instead of ATM, so it doesn't cancel out
# most of the primary position's profit (see that constant's docstring for
# the full reasoning, confirmed against Kite's own margin-benefit numbers).
_HEDGE_OTM_PCT = Decimal("0.05")


async def run_current_month_backtest(
    trade_repository: TursoTradeRepository,
    derivatives_chain: KiteDerivativesChain,
    options_trade_repository: TursoOptionsTradeRepository,
    futures_trade_repository: TursoFuturesTradeRepository,
    interval: str,
) -> list[str]:
    """Backtests every closed equity trade whose entry falls in the current
    calendar month. Returns one human-readable note per simulated leg
    written (best-effort -- a missing contract or missing historical
    premium silently skips that leg, matching the live shadow-tracking
    functions' own error handling).

    Clears every previous source='backtest' row first -- this is a full
    replay, not an incremental update, and without clearing first, running
    it more than once (or after the shadow-tracking logic changes) just
    keeps appending on top of stale rows, mixing old and new results with
    no way to tell them apart.
    """
    await options_trade_repository.delete_backtest_trades()
    await futures_trade_repository.delete_backtest_trades()
    month_start = date.today().replace(day=1)
    trades = await trade_repository.get_trades(None, interval)
    notes: list[str] = []
    for trade in trades:
        if trade.status != "closed" or trade.exit_timestamp is None or trade.exit_price is None:
            continue
        if trade.entry_timestamp.date() < month_start:
            continue
        notes.extend(await _backtest_one_trade(
            trade, derivatives_chain, options_trade_repository, futures_trade_repository
        ))
    return notes


async def _backtest_one_trade(
    trade: Trade,
    derivatives_chain: KiteDerivativesChain,
    options_trade_repository: TursoOptionsTradeRepository,
    futures_trade_repository: TursoFuturesTradeRepository,
) -> list[str]:
    assert trade.exit_timestamp is not None and trade.exit_price is not None
    notes: list[str] = []

    # 2026-08-14: was comparing trade.side.value (SignalSide is a StrEnum
    # whose values are lowercase, "buy"/"sell" -- see domain/models.py)
    # against the uppercase literals "BUY"/"SELL", which never matched --
    # every trade silently fell through to the else branch and was
    # backtested as a SHORT future regardless of its real side. Found by
    # hand-checking this module's output against known BUY signals during
    # a Nifty50 analysis session; comparing against the enum member
    # directly instead of a string literal makes this class of mismatch
    # impossible to reintroduce.
    futures_side = "long" if trade.side == SignalSide.BUY else "short"
    future_note = await _backtest_future(
        trade.symbol, futures_side, "primary",
        trade.entry_timestamp, trade.exit_timestamp,
        derivatives_chain, futures_trade_repository,
    )
    if future_note is not None:
        notes.append(future_note)
        hedge_type = "PE" if futures_side == "long" else "CE"
        # PE is OTM below spot, CE is OTM above spot -- see _HEDGE_OTM_PCT.
        hedge_strike_target = (
            trade.entry_price * (1 - _HEDGE_OTM_PCT)
            if futures_side == "long"
            else trade.entry_price * (1 + _HEDGE_OTM_PCT)
        )
        hedge_note = await _backtest_option(
            trade.symbol, hedge_type, "hedge",
            trade.entry_timestamp, trade.entry_price, trade.exit_timestamp, trade.exit_price,
            derivatives_chain, options_trade_repository,
            strike_target_price=hedge_strike_target,
        )
        if hedge_note:
            notes.append(hedge_note)
    return notes


async def _backtest_option(
    symbol: str,
    option_type: str,
    purpose: str,
    entry_timestamp: datetime,
    entry_underlying: Decimal,
    exit_timestamp: datetime,
    exit_underlying: Decimal,
    derivatives_chain: KiteDerivativesChain,
    options_trade_repository: TursoOptionsTradeRepository,
    strike_target_price: Decimal | None = None,
) -> str | None:
    try:
        target = strike_target_price if strike_target_price is not None else entry_underlying
        contract = derivatives_chain.nearest_atm_option(symbol, option_type, float(target))
        if contract is None:
            return None
        entry_premium = derivatives_chain.historical_premium(
            contract["instrument_token"], entry_timestamp
        )
        exit_premium = derivatives_chain.historical_premium(
            contract["instrument_token"], exit_timestamp
        )
        if entry_premium is None or exit_premium is None:
            return None
        entry_premium_d = Decimal(str(entry_premium))
        exit_premium_d = Decimal(str(exit_premium))
        lot_size = int(contract["lot_size"])
        pnl_amount = (exit_premium_d - entry_premium_d) * lot_size
        pnl_percent = (exit_premium_d - entry_premium_d) / entry_premium_d * 100
        trade = OptionsShadowTrade(
            symbol=symbol,
            option_type=option_type,
            purpose=purpose,
            option_tradingsymbol=contract["tradingsymbol"],
            strike=Decimal(str(contract["strike"])),
            expiry=str(contract["expiry"]),
            lot_size=lot_size,
            entry_timestamp=entry_timestamp,
            underlying_price_at_entry=entry_underlying,
            entry_premium=entry_premium_d,
            exit_timestamp=exit_timestamp,
            underlying_price_at_exit=exit_underlying,
            exit_premium=exit_premium_d,
            pnl_amount=pnl_amount,
            pnl_percent=pnl_percent,
            status="closed",
            source="backtest",
        )
        await options_trade_repository.insert_backtest_trade(trade)
        return (
            f"backtest options({purpose}): {contract['tradingsymbol']} {option_type} "
            f"pnl={pnl_percent:.2f}%"
        )
    except Exception:
        logger.exception("Backtest option leg failed for %s %s (%s)", symbol, option_type, purpose)
        return None


async def _backtest_future(
    symbol: str,
    side: str,
    purpose: str,
    entry_timestamp: datetime,
    exit_timestamp: datetime,
    derivatives_chain: KiteDerivativesChain,
    futures_trade_repository: TursoFuturesTradeRepository,
) -> str | None:
    try:
        contract = derivatives_chain.nearest_future(symbol)
        if contract is None:
            return None
        # Use the future's own historical price, not the underlying equity
        # price -- futures trade at a basis to spot, so pricing this off
        # ``entry_price``/``exit_price`` (equity fill prices) would
        # understate/overstate PnL versus what a real futures trade saw.
        future_entry_price = derivatives_chain.historical_premium(
            contract["instrument_token"], entry_timestamp
        )
        future_exit_price = derivatives_chain.historical_premium(
            contract["instrument_token"], exit_timestamp
        )
        if future_entry_price is None or future_exit_price is None:
            return None
        entry_price_d = Decimal(str(future_entry_price))
        exit_price_d = Decimal(str(future_exit_price))
        lot_size = int(contract["lot_size"])
        if side == "long":
            pnl_amount = (exit_price_d - entry_price_d) * lot_size
        else:
            pnl_amount = (entry_price_d - exit_price_d) * lot_size
        pnl_percent = pnl_amount / (entry_price_d * lot_size) * 100
        trade = FuturesShadowTrade(
            symbol=symbol,
            side=side,
            futures_tradingsymbol=contract["tradingsymbol"],
            expiry=str(contract["expiry"]),
            lot_size=lot_size,
            entry_timestamp=entry_timestamp,
            entry_price=entry_price_d,
            exit_timestamp=exit_timestamp,
            exit_price=exit_price_d,
            pnl_amount=pnl_amount,
            pnl_percent=pnl_percent,
            status="closed",
            source="backtest",
            purpose=purpose,
        )
        await futures_trade_repository.insert_backtest_trade(trade)
        return (
            f"backtest futures({purpose}): {side} {contract['tradingsymbol']} "
            f"pnl={pnl_percent:.2f}%"
        )
    except Exception:
        logger.exception("Backtest futures leg failed for %s (%s)", symbol, side)
        return None

"""Futures paper account: simulated real-money futures+hedge combo positions,
capped by real margin instead of full notional.

Sibling to ``paper_trading.py``, not a replacement -- separate capital pool
(see ``domain/models.py``'s ``FuturesPaperPosition`` docstring for why),
separate eligibility track record, separate slot math. Where the cash paper
account is long-only (NSE cash market can't short) and sizes each position
off ``total_equity / TARGET_SLOTS`` of *notional*, this account can take
both long and short (futures are the real short mechanism, see
``application/futures_shadow.py``) and sizes off ``total_equity /
TARGET_SLOTS`` of *margin* -- the actual SPAN+exposure Kite's basket-margin
API says the futures+hedge combo together will block, netted against the
account's existing positions (``KiteDerivativesChain.margin_benefit``'s
``combined_margin``), not a guessed percentage of the futures leg's full
notional.

``application/futures_shadow.py`` remains uncapped and analysis-only by its
own design (shadow-tracks *every* signal, no capital gate at all) -- this
module is a new, additional gate that decides whether a signal *also*
becomes a tracked, capital-constrained futures paper position, exactly
mirroring how ``paper_trading.py`` sits alongside the strategy's raw
``trades`` bookkeeping for the cash side.

2026-08-14: wired into ``signal_pipeline.py``'s live scan loop via
``open_futures_paper_position``/``close_futures_paper_position`` below,
restricted to the futures-paper symbol allowlist (``AppConfig.
futures_paper_symbols_file`` -- Nifty50 by default, see ``config/
nifty50_symbols.txt``) so the margin-API-per-signal latency this module's
docstring used to flag stays bounded to 49 symbols, not the full 220.
Reuses the eligibility bar from this module (55% win rate, this exact
symbol+side's own closed trades, ``MIN_CLOSED_TRADES`` minimum) --
unchanged from when this module was first built.
"""

import asyncio
import logging
import os
from datetime import datetime
from decimal import Decimal

from dotenv import load_dotenv

from trading_scanner.domain.models import FuturesPaperPosition, SignalSide
from trading_scanner.domain.ports import FuturesPaperAccountRepository, TradeRepository
from trading_scanner.infrastructure.kite import KiteDerivativesChain

load_dotenv()

# Deliberately separate .env keys from the cash paper account's -- see this
# module's docstring on why the two books are not shared.
FUTURES_INITIAL_CAPITAL = Decimal(os.getenv("TRADING_SCANNER_FUTURES_CAPITAL", "400000"))
FUTURES_TARGET_SLOTS = int(os.getenv("TRADING_SCANNER_FUTURES_SLOTS", "16"))
FUTURES_MIN_SLOT_MARGIN = Decimal(os.getenv("TRADING_SCANNER_FUTURES_MIN_MARGIN", "15000"))
# Never deploy 100% of a slot's margin budget -- MTM losses on positions
# already open can raise real maintenance-margin requirements intraday
# (a margin call), so headroom is kept unallocated on purpose. 25% mirrors
# the kind of buffer Zerodha itself recommends holding above the bare SPAN+
# exposure figure for exactly this reason.
FUTURES_MARGIN_BUFFER_PCT = Decimal(os.getenv("TRADING_SCANNER_FUTURES_MARGIN_BUFFER", "0.25"))

MIN_WIN_RATE = Decimal("55")
MIN_CLOSED_TRADES = 5


async def is_eligible(
    symbol: str, side: SignalSide, interval: str, trade_repository: TradeRepository
) -> bool:
    """Return whether a symbol's own track record on this side clears the bar.

    Unlike the cash paper account's BUY-only check, futures can take either
    side, so eligibility is checked against the matching side's own closed
    trades -- a symbol whose edge is only proven long should not
    automatically also get a green light to short it, and vice versa.
    """
    trades = await trade_repository.get_trades(symbol, interval)
    closed = [trade for trade in trades if trade.side == side and trade.status == "closed"]
    if len(closed) < MIN_CLOSED_TRADES:
        return False
    wins = sum(1 for trade in closed if trade.pnl_percent is not None and trade.pnl_percent > 0)
    return Decimal(100 * wins) / len(closed) >= MIN_WIN_RATE


async def try_open_futures_position(
    symbol: str,
    side: str,
    entry_timestamp: datetime,
    futures_entry_price: Decimal,
    futures_tradingsymbol: str,
    hedge_tradingsymbol: str,
    hedge_entry_price: Decimal,
    lot_size: int,
    derivatives_chain: KiteDerivativesChain,
    futures_account_repository: FuturesPaperAccountRepository,
) -> FuturesPaperPosition | None:
    """Open a futures+hedge combo paper position if this account's own
    capital (sized off live margin, not notional) allows it.

    One open combo per symbol, mirroring the cash paper account. Margin is
    priced fresh via Kite's real basket-margin API for this exact combo
    (see ``KiteDerivativesChain.margin_benefit``) -- if that call fails
    (no live Kite session, API hiccup), this returns None rather than
    guessing a margin figure, matching ``try_open_futures_position`` in
    ``futures_shadow.py``'s own best-effort/silent-on-failure convention.
    """
    open_positions = await futures_account_repository.get_open_positions()
    if any(position.symbol == symbol for position in open_positions):
        return None

    cash_balance = await futures_account_repository.get_cash_balance()
    total_equity = cash_balance + sum(
        (position.margin_allocated for position in open_positions), start=Decimal("0")
    )
    slot_budget = max(total_equity / FUTURES_TARGET_SLOTS, FUTURES_MIN_SLOT_MARGIN)

    futures_transaction = "BUY" if side == "long" else "SELL"
    # margin_benefit calls the blocking kiteconnect client -- run off the
    # event loop, matching webapp.py's own asyncio.to_thread usage of it.
    margin_result = await asyncio.to_thread(
        derivatives_chain.margin_benefit,
        [
            (futures_tradingsymbol, futures_transaction, lot_size),
            # The hedge option is always bought (paying premium) regardless
            # of which side the future is on -- see signal_pipeline.py's
            # _open_derivatives_shadow, this mirrors the same structure.
            (hedge_tradingsymbol, "BUY", lot_size),
        ],
    )
    if margin_result is None:
        return None
    required_margin = Decimal(str(margin_result["combined_margin"])) * (
        1 + FUTURES_MARGIN_BUFFER_PCT
    )
    if required_margin > slot_budget or required_margin > cash_balance:
        return None

    position = FuturesPaperPosition(
        symbol=symbol,
        side=side,
        entry_timestamp=entry_timestamp,
        futures_entry_price=futures_entry_price,
        futures_tradingsymbol=futures_tradingsymbol,
        hedge_tradingsymbol=hedge_tradingsymbol,
        lot_size=lot_size,
        margin_allocated=required_margin,
        hedge_entry_price=hedge_entry_price,
    )
    await futures_account_repository.open_position(position)
    return position


# Same OTM target as signal_pipeline.py's own hedge leg (_HEDGE_OTM_PCT) and
# derivatives_backtest.py's copy -- duplicated rather than imported to avoid
# a circular import (signal_pipeline.py is the one calling into this
# module, not the other way around); see either of those two for the full
# reasoning on why OTM instead of ATM. 5% (2026-08-17, was 2%): user
# preference for a wider/cheaper hedge, less downside protection but a
# smaller premium drag on the combo's real P&L.
_HEDGE_OTM_PCT = Decimal("0.05")


async def open_futures_paper_position(
    symbol: str,
    side: SignalSide,
    entry_timestamp: datetime,
    market_price: Decimal,
    interval: str,
    derivatives_chain: KiteDerivativesChain,
    trade_repository: TradeRepository,
    futures_account_repository: FuturesPaperAccountRepository,
) -> str | None:
    """One call per signal, mirroring ``signal_pipeline._open_derivatives_shadow``'s
    shape but for the real, capital-gated futures paper account instead of
    the uncapped analysis-only shadow: resolves the nearest futures
    contract and its OTM hedge option, checks this exact symbol+side's own
    55%-win-rate track record (``is_eligible``), then hands off to
    ``try_open_futures_position`` for the real margin check.

    ``market_price`` (the underlying/equity price) is used ONLY to target
    the hedge option's OTM strike, never as the futures leg's own entry
    price -- a futures contract trades at its own basis to spot (confirmed
    with real data this session: on real August trades, cash and futures
    P&L on the *same* signal diverged by up to tens of thousands of rupees,
    sometimes even opposite signs). The futures leg's real entry price is
    this contract's own live LTP (``derivatives_chain.ltp``), fetched here
    -- using the equity price as a stand-in would silently defeat the
    entire point of a *futures* paper account.

    Caller (``signal_pipeline.py``) is responsible for only calling this
    for symbols on the futures-paper allowlist (see ``AppConfig.
    futures_paper_symbols_file`` -- Nifty50 today) -- this function itself
    is symbol-agnostic, same as every other function in this module.

    Best-effort: returns None (nothing opened) on missing eligibility, a
    missing futures/options contract, a missing live futures quote, or a
    margin-check failure, matching ``try_open_futures_position``'s own
    silent-on-failure convention. Never raises into the caller.
    """
    if not await is_eligible(symbol, side, interval, trade_repository):
        return None
    futures_contract = derivatives_chain.nearest_future(symbol)
    if futures_contract is None:
        return None
    futures_ltp = derivatives_chain.ltp(f"NFO:{futures_contract['tradingsymbol']}")
    if futures_ltp is None:
        return None
    futures_entry_price = Decimal(str(futures_ltp))
    futures_side = "long" if side == SignalSide.BUY else "short"
    hedge_option_type = "PE" if side == SignalSide.BUY else "CE"
    # PE is OTM below spot, CE is OTM above spot -- see _HEDGE_OTM_PCT.
    # Deliberately still keyed off market_price (spot), not futures_entry_price
    # -- OTM hedge strikes are conventionally set relative to the underlying.
    hedge_strike_target = (
        market_price * (1 - _HEDGE_OTM_PCT)
        if side == SignalSide.BUY
        else market_price * (1 + _HEDGE_OTM_PCT)
    )
    hedge_contract = derivatives_chain.nearest_atm_option(
        symbol, hedge_option_type, float(hedge_strike_target)
    )
    if hedge_contract is None:
        return None
    # 2026-08-17: required, not best-effort -- the hedge's own entry
    # premium has to be known now to ever compute a real combined P&L at
    # close (see close_futures_paper_position). Opening a combo we can't
    # later account for properly isn't better than not opening it.
    hedge_ltp = derivatives_chain.ltp(f"NFO:{hedge_contract['tradingsymbol']}")
    if hedge_ltp is None:
        return None
    position = await try_open_futures_position(
        symbol,
        futures_side,
        entry_timestamp,
        futures_entry_price,
        futures_contract["tradingsymbol"],
        hedge_contract["tradingsymbol"],
        Decimal(str(hedge_ltp)),
        int(futures_contract["lot_size"]),
        derivatives_chain,
        futures_account_repository,
    )
    if position is None:
        return None
    # 2026-08-17: the hedge leg's own tradingsymbol/entry premium wasn't
    # shown here before -- only the futures leg was, even though a real
    # hedge position was genuinely opened alongside it (real premium
    # spent). Both legs now shown explicitly. " | " not "; " as the
    # internal separator -- this whole string is one segment of a Signal.
    # rationale that itself gets split on ";" (see infrastructure/
    # telegram.py's _format_entry), so a "; " here would fracture this
    # note across multiple bullets instead of staying one coherent line.
    return (
        f"futures-paper: opened {futures_side} {futures_contract['tradingsymbol']} "
        f"@ ₹{futures_entry_price} | hedge: bought {hedge_contract['tradingsymbol']} "
        f"@ ₹{hedge_ltp} | margin=₹{position.margin_allocated:.0f}"
    )


async def close_futures_paper_position(
    symbol: str,
    exit_timestamp: datetime,
    derivatives_chain: KiteDerivativesChain,
    futures_account_repository: FuturesPaperAccountRepository,
) -> str | None:
    """Close whatever ``open_futures_paper_position`` opened for ``symbol``,
    if anything did. None (no-op) if this symbol has no open combo in the
    futures paper account -- most signals never clear eligibility/margin in
    the first place, so this is the common case, not an error.

    Exit price is this contract's own live LTP (``derivatives_chain.ltp``),
    same reasoning as ``open_futures_paper_position``'s entry price -- NOT
    the underlying equity price the caller's cash-side exit uses. Using the
    equity price here would silently price every futures exit off the
    wrong instrument (this was a real bug, fixed 2026-08-14, caught by the
    user noticing futures P&L wasn't tracking the underlying's real move --
    which is exactly correct behavior once this used the real futures
    price; it just wasn't, until this fix).

    2026-08-17: also fetches the hedge option's own exit LTP so the
    reported P&L is the real combo's (futures leg + hedge leg), not just
    the futures leg alone -- see ``TursoFuturesPaperAccountRepository.
    close_position``. Best-effort on the hedge quote specifically (an
    illiquid far/near-OTM option can have no live quote at exit): falls
    back to futures-only P&L rather than leaving the position stuck open
    over one bad quote, but still closes for real either way.
    """
    contract = derivatives_chain.nearest_future(symbol)
    if contract is None:
        return None
    futures_ltp = derivatives_chain.ltp(f"NFO:{contract['tradingsymbol']}")
    if futures_ltp is None:
        return None
    open_positions = await futures_account_repository.get_open_positions()
    open_position = next((p for p in open_positions if p.symbol == symbol), None)
    if open_position is None:
        return None
    hedge_ltp = derivatives_chain.ltp(f"NFO:{open_position.hedge_tradingsymbol}")
    if hedge_ltp is None:
        logging.getLogger(__name__).warning(
            "No live quote for hedge leg %s closing %s -- P&L will be futures-only.",
            open_position.hedge_tradingsymbol, symbol,
        )
    position = await futures_account_repository.close_position(
        symbol, exit_timestamp, Decimal(str(futures_ltp)),
        hedge_exit_price=Decimal(str(hedge_ltp)) if hedge_ltp is not None else None,
    )
    if position is None:
        return None
    pnl_percent = position.pnl_amount / position.margin_allocated * 100
    # " | " internally, same reasoning as open_futures_paper_position's note.
    hedge_note = (
        f" | hedge: {position.hedge_tradingsymbol} @ ₹{position.hedge_exit_price} "
        f"(entry ₹{position.hedge_entry_price})"
        if position.hedge_exit_price is not None
        else f" | hedge: {position.hedge_tradingsymbol} (no live quote at close, futures-only P&L)"
    )
    return (
        f"futures-paper: closed {position.side} {position.futures_tradingsymbol} "
        f"@ ₹{position.futures_exit_price}{hedge_note} | "
        f"pnl=₹{position.pnl_amount:.0f} ({pnl_percent:.2f}%)"
    )

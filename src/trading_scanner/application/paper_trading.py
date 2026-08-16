"""Paper-trading account: long-only simulated real-money positions.

NSE cash market does not allow short selling for multi-day (delivery) holds
-- only intraday MIS positions can be short, squared off same day. This
strategy's average holding period is ~3.5 days, so SELL/short signals can
never be executed as real cash-market positions; the paper account only ever
opens a position on a BUY entry. SELL signals still notify (see
``signal_pipeline.py``) but are informational only.

Two gates decide whether a BUY entry actually becomes a paper position:

1. **Eligibility**: the symbol's own closed-trade, BUY-only win rate (see
   ``application/backtest.py``/``signal_pipeline.py``'s trade bookkeeping)
   must be at least ``MIN_WIN_RATE``, and it must have at least
   ``MIN_CLOSED_TRADES`` closed BUY trades to compute a meaningful rate from.
   A symbol with no track record yet, or a poor one, is skipped -- still
   notified, just tagged as not paper-traded.
2. **Capacity**: the account only has ``INITIAL_CAPITAL`` to work with, split
   into ``TARGET_SLOTS`` dynamically-sized slots. If the cash balance can't
   cover one more slot, the entry is skipped and tagged accordingly rather
   than silently dropped.

``TARGET_SLOTS`` (32) matches real signal demand: Little's Law
(concurrent positions needed ~= entries/day x average holding period),
computed only over symbols that actually clear the eligibility bar above
(ineligible symbols never reach ``try_open_position`` at all, so they don't
count toward real capacity demand). ``INITIAL_CAPITAL`` (Rs 8,00,000) is
sized so 32 slots at the resulting ~Rs 25,000/slot fully covers that demand
with no capital-driven skips under normal conditions.

Slot size is **dynamic**, not fixed: every entry recomputes
``total_equity / TARGET_SLOTS``, where total_equity is cash plus all open
positions' allocated capital. As the account compounds profit week over
week, each slot grows proportionally -- no manual re-tuning needed. A floor
(``MIN_POSITION_SIZE``, Rs 25,000) keeps the flat per-trade DP charge
(~Rs 18, sell-side only) under ~5% of an average winning trade's profit;
below that floor, flat fees start eating a disproportionate share of returns.
"""

import os
from datetime import datetime
from decimal import Decimal

from dotenv import load_dotenv

from trading_scanner.domain.models import PaperPosition, SignalSide
from trading_scanner.domain.ports import PaperAccountRepository, TradeRepository

# Loaded here (not just in config/settings.py) because the constants below
# are read from the environment at import time, which can happen before
# signals.py's main() gets around to calling load_config(). Safe to call
# more than once -- dotenv never overwrites an already-set env var.
load_dotenv()

# Overridable via .env so the dashboard's config editor can adjust sizing
# without a code change/redeploy -- defaults match the Little's Law sizing
# derived above (32 slots, Rs 8,00,000, Rs 25,000 floor).
INITIAL_CAPITAL = Decimal(os.getenv("TRADING_SCANNER_PAPER_CAPITAL", "800000"))
TARGET_SLOTS = int(os.getenv("TRADING_SCANNER_PAPER_SLOTS", "32"))
MIN_POSITION_SIZE = Decimal(os.getenv("TRADING_SCANNER_PAPER_MIN_POSITION", "25000"))
MIN_WIN_RATE = Decimal("55")
MIN_CLOSED_TRADES = 5

# The strategy has no price-based risk control of its own -- a losing
# position only closes when the model's opposite signal eventually fires,
# however far price has moved by then. application/stop_loss_replay.py
# validated this threshold against every real historical trade's actual
# candle-by-candle path (2026-08-13, post corrupted-candle cleanup):
# BUY expectancy 0.194% -> 1.099%, SELL expectancy -15.27% -> +0.99% at a
# 3% cap. This only ever fires *before* the strategy's own exit signal --
# if the signal comes first, nothing changes; see live_pipeline.py's
# tick-level stop check.
STOP_LOSS_PCT = Decimal(os.getenv("TRADING_SCANNER_STOP_LOSS_PCT", "3"))


def stop_loss_price(entry_price: Decimal) -> Decimal:
    """The price at which an open BUY position should be force-closed,
    instead of waiting for the strategy's own opposite signal. Long-only,
    so the stop is always below entry."""
    return entry_price * (1 - STOP_LOSS_PCT / 100)


# 2026-08-14: like STOP_LOSS_PCT, validated against every real historical
# trade's actual candle-by-candle path (application/profit_protection_
# replay.py, 6,292 closed BUY trades) before deploying, not just intuition.
#
# The strategy's profit is extremely concentrated in a small number of big
# winners -- the top 10% of trades produce ~70% of all-time total profit,
# and just 37 trades (0.6%) that ever ran past +15% are worth 10.6% of it
# alone. A LOW activation threshold (tested 3-5%) trims that tail short and
# *reduces* total return (-24% to -35% vs no trailing stop at all) because
# it fires on ordinary trades that were never going to run big, converting
# some small losses into small wins while clipping the rare huge winners
# that actually drive returns.
#
# A HIGH activation threshold avoids this: it never engages on a trade
# until price has already moved up TRAILING_STOP_ACTIVATION_PCT from entry
# (well past what a typical trade reaches), so it's a no-op for the ~90% of
# trades that stay small, and only protects gains already banked on the
# rare big runners -- without capping how far they can still run. Tested
# 10/15/20/25% activation x 2/3/5% trail, all beat the no-trailing-stop
# baseline; activate=15%/trail=3% chosen as a solidly-supported middle of
# that range rather than the single best (most extreme) grid cell, to
# avoid overfitting a coarse grid search to one dataset snapshot.
TRAILING_STOP_ACTIVATION_PCT = Decimal(
    os.getenv("TRADING_SCANNER_TRAILING_STOP_ACTIVATION_PCT", "15")
)
TRAILING_STOP_TRAIL_PCT = Decimal(os.getenv("TRADING_SCANNER_TRAILING_STOP_TRAIL_PCT", "3"))


def trailing_stop_price(entry_price: Decimal, peak_price: Decimal) -> Decimal | None:
    """The trailing-stop exit price for an open BUY position, or None if
    the trail hasn't activated yet (price never reached
    ``TRAILING_STOP_ACTIVATION_PCT`` above entry).

    ``peak_price`` is the highest price seen since entry (see
    ``PaperPosition.peak_price``) -- the trail is anchored to that, not the
    current price, so it only ever tightens as a position makes new highs
    and never chases price back down. Checked *after* the hard stop-loss
    (see ``stop_loss_price``) and *before* the strategy's own exit signal
    -- see ``live_pipeline.py``'s tick-level check.
    """
    if peak_price < entry_price * (1 + TRAILING_STOP_ACTIVATION_PCT / 100):
        return None
    return peak_price * (1 - TRAILING_STOP_TRAIL_PCT / 100)


async def is_eligible(symbol: str, interval: str, trade_repository: TradeRepository) -> bool:
    """Return whether a symbol's BUY-only track record clears the paper-trading bar.

    Long-only, so only BUY-side closed trades count -- a symbol whose edge is
    entirely on the SELL side is still not tradeable here.
    """
    win_rate = await _buy_only_win_rate(symbol, interval, trade_repository)
    return win_rate is not None and win_rate >= MIN_WIN_RATE


async def try_open_position(
    symbol: str,
    entry_timestamp: datetime,
    entry_price: Decimal,
    paper_account_repository: PaperAccountRepository,
) -> PaperPosition | None:
    """Open a paper position sized off current total equity if capital allows.

    Slot size is recomputed fresh on every call from total_equity /
    TARGET_SLOTS (floored at MIN_POSITION_SIZE) so the account scales
    proportionally as profit compounds in, without a hardcoded slot size
    going stale. Returns None (no position opened) if the remaining cash
    balance can't cover one more slot -- the caller is responsible for
    notifying that the signal was skipped for lack of capital, not silently
    dropping it.
    """
    cash_balance = await paper_account_repository.get_cash_balance()
    open_positions = await paper_account_repository.get_open_positions()
    total_equity = cash_balance + sum(
        (position.capital_allocated for position in open_positions), start=Decimal("0")
    )
    position_size = max(total_equity / TARGET_SLOTS, MIN_POSITION_SIZE)

    if cash_balance < position_size:
        return None
    quantity = int(position_size / entry_price)
    if quantity < 1:
        return None
    position = PaperPosition(
        symbol=symbol,
        entry_timestamp=entry_timestamp,
        entry_price=entry_price,
        quantity=quantity,
        capital_allocated=quantity * entry_price,
    )
    await paper_account_repository.open_position(position)
    return position


async def _buy_only_win_rate(
    symbol: str, interval: str, trade_repository: TradeRepository
) -> Decimal | None:
    """Compute the closed BUY-only win rate, or None if too few trades exist."""
    trades = await trade_repository.get_trades(symbol, interval)
    closed_buys = [
        trade for trade in trades if trade.side == SignalSide.BUY and trade.status == "closed"
    ]
    if len(closed_buys) < MIN_CLOSED_TRADES:
        return None
    wins = sum(
        1 for trade in closed_buys if trade.pnl_percent is not None and trade.pnl_percent > 0
    )
    return Decimal(100 * wins) / len(closed_buys)

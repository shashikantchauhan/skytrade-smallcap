from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class SignalSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Candle:
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    side: SignalSide
    strategy: str
    timestamp: datetime
    price: Decimal
    rationale: str
    # Drives Telegram formatting only (see infrastructure/telegram.py) --
    # not persisted, not used for fingerprint/dedup, purely "which kind of
    # message is this so the notifier can header/emoji it distinctly":
    #   "entry"      -- a fresh BUY/SELL strategy signal
    #   "exit"       -- the strategy itself closing a position (informational
    #                    only when no paper position was open for it)
    #   "paper_exit" -- an actual simulated-capital paper position closing,
    #                    the one with real P&L
    category: str = "entry"

    @property
    def fingerprint(self) -> str:
        return f"{self.strategy}:{self.symbol}:{self.side}:{self.timestamp.isoformat()}"


@dataclass(frozen=True, slots=True)
class Trade:
    """A recorded entry (and, once closed, exit) for win-rate/backtest tracking.

    ``side`` follows AlphaEngine's own semantics: BUY is a long entry (profit
    when price rises), SELL is a short entry (profit when price falls) --
    confirmed against the Pine script's own backtest helper, which computes
    long profit as ``exit - entry`` and short profit as ``entry - exit``.
    """

    symbol: str
    side: SignalSide
    entry_timestamp: datetime
    entry_price: Decimal
    prediction_at_entry: int
    is_early_signal_flip: bool
    exit_timestamp: datetime | None = None
    exit_price: Decimal | None = None
    pnl_percent: Decimal | None = None
    status: str = "open"  # "open" | "closed"
    # Feature snapshot at entry, for training a future ranking/meta-labeling
    # model (see application/backtest.py) -- all optional so existing rows
    # (and any caller not yet passing them) remain valid.
    adx_at_entry: float | None = None
    regime_normalized_at_entry: float | None = None
    volatility_margin_at_entry: float | None = None
    volatility_filter_passed: bool | None = None
    regime_filter_passed: bool | None = None
    adx_filter_passed: bool | None = None


@dataclass(frozen=True, slots=True)
class PaperPosition:
    """A simulated real-money position in the paper-trading account.

    Long-only: NSE cash market doesn't allow short selling for multi-day
    holds, so paper positions only ever come from BUY entries (SELL signals
    stay informational -- see ``application/paper_trading.py``). Distinct
    from ``Trade`` (which records the strategy's own raw backtest scoring
    for every symbol regardless of tradability); this tracks actual capital
    committed and returned by the paper account.
    """

    symbol: str
    entry_timestamp: datetime
    entry_price: Decimal
    quantity: int
    capital_allocated: Decimal
    exit_timestamp: datetime | None = None
    exit_price: Decimal | None = None
    pnl_amount: Decimal | None = None
    status: str = "open"  # "open" | "closed"
    # Highest price seen since entry -- starts at entry_price, only ever
    # moves up. Drives the trailing stop (see
    # ``application/paper_trading.trailing_stop_price``): validated
    # 2026-08-14 against every real historical trade's candle-by-candle
    # path (``application/profit_protection_replay.py``) that a trail only
    # helps once activated well past the median trade's move (>=10-15%) --
    # a low activation threshold clips the strategy's rare big winners and
    # *reduces* total return (top 10% of trades produce ~70% of all
    # profit). None until the live tick-level check starts tracking it.
    peak_price: Decimal | None = None


@dataclass(frozen=True, slots=True)
class FuturesPaperPosition:
    """A simulated real-money futures+hedge combo position in the futures
    paper account (see ``application/futures_trading.py``).

    Distinct from ``FuturesShadowTrade`` (which shadow-tracks *every*
    BUY/SELL signal, uncapped, analysis-only): this only exists for combos
    that actually cleared the futures paper account's own capital gate --
    sized against ``margin_allocated`` (live SPAN+exposure margin for the
    futures+hedge combo together, from ``KiteDerivativesChain.
    margin_benefit``'s ``combined_margin``, not the futures leg's full
    notional), a plus-buffer figure of which is deducted from the account's
    own capital pool. Own book, separate from the cash paper account's
    capital (see NOTES.md's futures-capital-gate roadmap for why: simpler
    to reason about, and the two are already tracked in separate tables
    with no shared accounting today).
    """

    symbol: str
    side: str  # "long" | "short"
    entry_timestamp: datetime
    futures_entry_price: Decimal
    futures_tradingsymbol: str
    hedge_tradingsymbol: str
    lot_size: int
    margin_allocated: Decimal
    # 2026-08-17: the hedge option's own entry/exit premium -- it's always
    # bought (long), paying real premium, so its own price move is part of
    # the combo's real economics, not just a margin-sizing input. Before
    # this, pnl_amount only ever reflected the futures leg alone, while the
    # combo was *sized* (margin) as if the hedge mattered -- inconsistent,
    # and understated the real cost/benefit of the hedge.
    hedge_entry_price: Decimal | None = None
    hedge_exit_price: Decimal | None = None
    exit_timestamp: datetime | None = None
    futures_exit_price: Decimal | None = None
    pnl_amount: Decimal | None = None
    status: str = "open"  # "open" | "closed"


@dataclass(frozen=True, slots=True)
class OptionsShadowTrade:
    """A hypothetical options trade shadowing a BUY/SELL signal.

    Two distinct purposes, both analysis-only, never a real order:

    - ``purpose="directional"``: buying an option as a standalone
      directional bet -- a CALL for a BUY signal, a PUT for a SELL signal
      (NSE cash market has no short selling, so a SELL can never become a
      real/paper equity position -- see ``paper_trading.py`` -- this is
      what a synthetic short via options would have looked like instead).
    - ``purpose="hedge"``: a protective option bought alongside a shadow
      futures position (see ``FuturesShadowTrade``) -- a PUT hedging a long
      future, a CALL hedging a short future.

    ``option_type`` is Kite's own convention: ``"CE"`` (call) or ``"PE"``
    (put). Entirely separate from the paper account's capital.
    """

    symbol: str
    option_type: str  # "CE" | "PE"
    purpose: str  # "directional" | "hedge"
    option_tradingsymbol: str
    strike: Decimal
    expiry: str
    lot_size: int
    entry_timestamp: datetime
    underlying_price_at_entry: Decimal
    entry_premium: Decimal
    exit_timestamp: datetime | None = None
    underlying_price_at_exit: Decimal | None = None
    exit_premium: Decimal | None = None
    pnl_amount: Decimal | None = None
    pnl_percent: Decimal | None = None
    status: str = "open"  # "open" | "closed"
    source: str = "live"  # "live" (forward shadow-tracking) | "backtest" (current-month replay)


@dataclass(frozen=True, slots=True)
class FuturesShadowTrade:
    """A hypothetical futures trade shadowing a BUY/SELL signal.

    Two distinct purposes, two independent strategies tracked side by side
    for comparison (never a real order, both analysis-only):

    - ``purpose="primary"``: the futures position *is* the trade --
      ``side="long"`` for a BUY signal, ``side="short"`` for a SELL signal
      (the real short mechanism a real broker connection would use, unlike
      equity, where the NSE cash market has no short selling). Paired with
      an ``OptionsShadowTrade`` hedge (``purpose="hedge"``) opened at the
      same time: a PUT for a long future, a CALL for a short future.
    - ``purpose="hedge"``: the reverse structure -- the directional option
      (``OptionsShadowTrade``, ``purpose="directional"``) is the trade, and
      *this* future is what hedges it, opposite-delta to the option: a
      short future hedging a bought CALL, a long future hedging a bought
      PUT.

    Both purposes can be open for the same symbol at once, hence every
    lookup/close is scoped by ``(symbol, purpose)``, not just symbol.
    """

    symbol: str
    side: str  # "long" | "short"
    futures_tradingsymbol: str
    expiry: str
    lot_size: int
    entry_timestamp: datetime
    entry_price: Decimal
    purpose: str = "primary"  # "primary" | "hedge"
    exit_timestamp: datetime | None = None
    exit_price: Decimal | None = None
    pnl_amount: Decimal | None = None
    pnl_percent: Decimal | None = None
    status: str = "open"  # "open" | "closed"
    source: str = "live"  # "live" (forward shadow-tracking) | "backtest" (current-month replay)


@dataclass(frozen=True, slots=True)
class LiveOrderLeg:
    """One real order placed on Zerodha -- see ``application/
    live_execution.py`` for the full basket-entry/exit flow this is a
    record of. Distinct from ``OptionsShadowTrade``/``FuturesShadowTrade``
    (which are always hypothetical, ``source`` never real) -- every row
    here corresponds to an actual ``place_order`` call and a real fill (or
    a real rejection/cancellation).

    ``basket_id`` groups the legs of one entry/exit together (e.g. the
    option leg and the futures leg of one hedged-futures basket) so a
    partial failure -- one leg filled, the other didn't -- is queryable as
    a single unit instead of two unrelated rows.
    """

    basket_id: str
    symbol: str
    purpose: str  # "primary" (the futures leg) | "hedge" (the option leg)
    tradingsymbol: str
    transaction_type: str  # "BUY" | "SELL"
    quantity: int
    order_id: str
    status: str  # Kite's own order status: "COMPLETE" | "REJECTED" | "CANCELLED" | ...
    placed_at: datetime
    average_price: Decimal | None = None
    rejection_reason: str | None = None

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_scanner.application import futures_trading
from trading_scanner.domain.models import SignalSide, Trade


class FakeFuturesPaperAccountRepository:
    """In-memory FuturesPaperAccountRepository fake, mirrors
    test_signal_pipeline.py's FakePaperAccountRepository."""

    def __init__(self, cash_balance=Decimal("400000")) -> None:
        self._cash_balance = cash_balance
        self.opened = []

    async def get_cash_balance(self) -> Decimal:
        return self._cash_balance

    async def open_position(self, position) -> None:
        self.opened.append(position)
        self._cash_balance -= position.margin_allocated

    async def close_position(
        self, symbol, exit_timestamp, futures_exit_price, hedge_exit_price=None
    ):
        matching = [p for p in self.opened if p.symbol == symbol and p.status == "open"]
        if not matching:
            return None
        position = matching[-1]
        futures_pnl = (
            (futures_exit_price - position.futures_entry_price) * position.lot_size
            if position.side == "long"
            else (position.futures_entry_price - futures_exit_price) * position.lot_size
        )
        hedge_pnl = (
            (hedge_exit_price - position.hedge_entry_price) * position.lot_size
            if hedge_exit_price is not None and position.hedge_entry_price is not None
            else Decimal("0")
        )
        pnl_amount = futures_pnl + hedge_pnl
        self._cash_balance += position.margin_allocated + pnl_amount
        index = self.opened.index(position)
        self.opened[index] = replace(
            position, exit_timestamp=exit_timestamp, futures_exit_price=futures_exit_price,
            hedge_exit_price=hedge_exit_price, pnl_amount=pnl_amount, status="closed",
        )
        return self.opened[index]

    async def get_open_positions(self):
        return [p for p in self.opened if p.status == "open"]

    async def get_recent_closed_positions(self, limit):
        closed = [p for p in self.opened if p.status == "closed"]
        return sorted(closed, key=lambda p: p.exit_timestamp, reverse=True)[:limit]


class FakeTradeRepository:
    def __init__(self, trades) -> None:
        self._trades = trades

    async def open_trade(self, interval, trade) -> None:
        raise NotImplementedError

    async def close_open_trade(self, symbol, interval, side, exit_timestamp, exit_price) -> None:
        raise NotImplementedError

    async def abandon_open_trade(self, symbol, interval, side) -> None:
        raise NotImplementedError

    async def get_trades(self, symbol, interval):
        return [t for t in self._trades if symbol is None or t.symbol == symbol]


class FakeDerivativesChain:
    """Stands in for KiteDerivativesChain -- margin_benefit is used by
    try_open_futures_position directly; nearest_future/nearest_atm_option
    are used by open_futures_paper_position's contract resolution."""

    def __init__(
        self,
        combined_margin: Decimal | None,
        has_future: bool = True,
        has_hedge_option: bool = True,
        futures_ltp: float | None = 2910.0,
        hedge_ltp: float | None = None,
    ):
        self._combined_margin = combined_margin
        self._has_future = has_future
        self._has_hedge_option = has_hedge_option
        self._futures_ltp = futures_ltp
        # Defaults to futures_ltp (old behavior: every contract priced the
        # same) unless a test cares about the two legs moving differently.
        self._hedge_ltp = hedge_ltp if hedge_ltp is not None else futures_ltp

    def ltp(self, exchange_tradingsymbol):
        if exchange_tradingsymbol.endswith(("PE", "CE")):
            return self._hedge_ltp
        return self._futures_ltp

    def margin_benefit(self, legs):
        if self._combined_margin is None:
            return None
        return {
            "primary_only_margin": float(self._combined_margin) * 1.5,
            "combined_margin": float(self._combined_margin),
            "margin_benefit": float(self._combined_margin) * 0.5,
        }

    def nearest_future(self, symbol):
        if not self._has_future:
            return None
        return {
            "tradingsymbol": f"{symbol.removesuffix('.NS')}26AUGFUT",
            "lot_size": 500,
            "instrument_token": 111,
            "expiry": "2026-08-25",
        }

    def nearest_atm_option(self, symbol, option_type, underlying_price):
        if not self._has_hedge_option:
            return None
        name = symbol.removesuffix(".NS")
        strike = int(underlying_price)
        return {
            "tradingsymbol": f"{name}26AUG{strike}{option_type}",
            "lot_size": 500,
            "instrument_token": 222,
            "strike": underlying_price,
        }


def _closed_trade(symbol, side, win=True):
    return Trade(
        symbol=symbol, side=side, entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        entry_price=Decimal("100"), prediction_at_entry=5, is_early_signal_flip=False,
        exit_timestamp=datetime(2026, 1, 2, tzinfo=UTC),
        exit_price=Decimal("110") if win else Decimal("90"),
        pnl_percent=Decimal("10") if win else Decimal("-10"), status="closed",
    )


@pytest.mark.asyncio
async def test_is_eligible_checks_the_matching_side_only():
    trades = [_closed_trade("RELIANCE.NS", SignalSide.BUY) for _ in range(5)]
    repo = FakeTradeRepository(trades)
    assert await futures_trading.is_eligible("RELIANCE.NS", SignalSide.BUY, "60minute", repo)
    assert not await futures_trading.is_eligible("RELIANCE.NS", SignalSide.SELL, "60minute", repo)


@pytest.mark.asyncio
async def test_try_open_opens_when_margin_fits_the_slot_budget():
    account = FakeFuturesPaperAccountRepository(cash_balance=Decimal("400000"))
    # Default slot budget: max(400000/16, 15000) = 25000. A 15000 combined
    # margin, plus the 25% buffer, is 18750 -- comfortably under that.
    chain = FakeDerivativesChain(combined_margin=Decimal("15000"))
    position = await futures_trading.try_open_futures_position(
        "RELIANCE.NS", "long", datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "RELIANCE26AUGFUT", "RELIANCE26AUG2800PE", Decimal("50"), 500, chain, account,
    )
    assert position is not None
    assert position.margin_allocated == Decimal("15000") * (
        1 + futures_trading.FUTURES_MARGIN_BUFFER_PCT
    )
    assert account.opened == [position]


@pytest.mark.asyncio
async def test_try_open_skips_when_margin_exceeds_slot_budget():
    account = FakeFuturesPaperAccountRepository(cash_balance=Decimal("400000"))
    # Way beyond any reasonable slot budget for this account size.
    chain = FakeDerivativesChain(combined_margin=Decimal("1000000"))
    position = await futures_trading.try_open_futures_position(
        "RELIANCE.NS", "long", datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "RELIANCE26AUGFUT", "RELIANCE26AUG2800PE", Decimal("50"), 500, chain, account,
    )
    assert position is None
    assert account.opened == []


@pytest.mark.asyncio
async def test_try_open_skips_when_margin_api_fails():
    account = FakeFuturesPaperAccountRepository()
    chain = FakeDerivativesChain(combined_margin=None)
    position = await futures_trading.try_open_futures_position(
        "RELIANCE.NS", "long", datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "RELIANCE26AUGFUT", "RELIANCE26AUG2800PE", Decimal("50"), 500, chain, account,
    )
    assert position is None


@pytest.mark.asyncio
async def test_try_open_skips_a_symbol_already_open():
    account = FakeFuturesPaperAccountRepository()
    chain = FakeDerivativesChain(combined_margin=Decimal("10000"))
    first = await futures_trading.try_open_futures_position(
        "RELIANCE.NS", "long", datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "RELIANCE26AUGFUT", "RELIANCE26AUG2800PE", Decimal("50"), 500, chain, account,
    )
    assert first is not None
    second = await futures_trading.try_open_futures_position(
        "RELIANCE.NS", "short", datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "RELIANCE26AUGFUT", "RELIANCE26AUG3000CE", Decimal("50"), 500, chain, account,
    )
    assert second is None


# --- open_futures_paper_position / close_futures_paper_position ---
# The orchestrator that resolves real contracts (nearest_future/
# nearest_atm_option) and checks eligibility before handing off to
# try_open_futures_position -- this is what signal_pipeline.py actually
# calls per BUY/SELL signal.


@pytest.mark.asyncio
async def test_open_futures_paper_position_skips_when_not_eligible():
    # Only 5 BUY trades, all wins -> 100% win rate, so make it a losing
    # track record instead to fail the 55% bar cleanly.
    trades = [_closed_trade("RELIANCE.NS", SignalSide.BUY, win=False) for _ in range(5)]
    repo = FakeTradeRepository(trades)
    account = FakeFuturesPaperAccountRepository()
    chain = FakeDerivativesChain(combined_margin=Decimal("10000"))

    note = await futures_trading.open_futures_paper_position(
        "RELIANCE.NS", SignalSide.BUY, datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "60minute", chain, repo, account,
    )

    assert note is None
    assert account.opened == []


@pytest.mark.asyncio
async def test_open_futures_paper_position_opens_a_long_for_buy_when_eligible():
    trades = [_closed_trade("RELIANCE.NS", SignalSide.BUY, win=True) for _ in range(5)]
    repo = FakeTradeRepository(trades)
    account = FakeFuturesPaperAccountRepository()
    chain = FakeDerivativesChain(combined_margin=Decimal("10000"))

    note = await futures_trading.open_futures_paper_position(
        "RELIANCE.NS", SignalSide.BUY, datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "60minute", chain, repo, account,
    )

    assert note is not None
    assert len(account.opened) == 1
    assert account.opened[0].side == "long"
    assert account.opened[0].futures_tradingsymbol == "RELIANCE26AUGFUT"
    # PE hedges a long future -- see _HEDGE_OTM_PCT's docstring.
    assert "PE" in account.opened[0].hedge_tradingsymbol


@pytest.mark.asyncio
async def test_open_futures_paper_position_opens_a_short_for_sell_when_eligible():
    trades = [_closed_trade("RELIANCE.NS", SignalSide.SELL, win=True) for _ in range(5)]
    repo = FakeTradeRepository(trades)
    account = FakeFuturesPaperAccountRepository()
    chain = FakeDerivativesChain(combined_margin=Decimal("10000"))

    note = await futures_trading.open_futures_paper_position(
        "RELIANCE.NS", SignalSide.SELL, datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "60minute", chain, repo, account,
    )

    assert note is not None
    assert account.opened[0].side == "short"
    # CE hedges a short future.
    assert "CE" in account.opened[0].hedge_tradingsymbol


@pytest.mark.asyncio
async def test_open_futures_paper_position_skips_when_no_futures_contract():
    trades = [_closed_trade("RELIANCE.NS", SignalSide.BUY, win=True) for _ in range(5)]
    repo = FakeTradeRepository(trades)
    account = FakeFuturesPaperAccountRepository()
    chain = FakeDerivativesChain(combined_margin=Decimal("10000"), has_future=False)

    note = await futures_trading.open_futures_paper_position(
        "RELIANCE.NS", SignalSide.BUY, datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "60minute", chain, repo, account,
    )

    assert note is None
    assert account.opened == []


@pytest.mark.asyncio
async def test_open_futures_paper_position_skips_when_no_hedge_option():
    trades = [_closed_trade("RELIANCE.NS", SignalSide.BUY, win=True) for _ in range(5)]
    repo = FakeTradeRepository(trades)
    account = FakeFuturesPaperAccountRepository()
    chain = FakeDerivativesChain(combined_margin=Decimal("10000"), has_hedge_option=False)

    note = await futures_trading.open_futures_paper_position(
        "RELIANCE.NS", SignalSide.BUY, datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "60minute", chain, repo, account,
    )

    assert note is None
    assert account.opened == []


@pytest.mark.asyncio
async def test_close_futures_paper_position_closes_the_open_combo():
    # Entry and exit use SEPARATE chain instances with different live LTPs
    # -- the futures leg's own price, not the equity market_price passed
    # to open_futures_paper_position (which is only used for the hedge
    # strike target -- see that function's docstring). Hedge leg also
    # moves (decays from 50 to 45, a realistic OTM-option-losing-value
    # scenario) -- 2026-08-17: pnl_amount is the real *combined* P&L now,
    # not the futures leg alone (see FuturesPaperPosition's docstring).
    trades = [_closed_trade("RELIANCE.NS", SignalSide.BUY, win=True) for _ in range(5)]
    repo = FakeTradeRepository(trades)
    account = FakeFuturesPaperAccountRepository()
    entry_chain = FakeDerivativesChain(
        combined_margin=Decimal("10000"), futures_ltp=2900.0, hedge_ltp=50.0
    )
    await futures_trading.open_futures_paper_position(
        "RELIANCE.NS", SignalSide.BUY, datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "60minute", entry_chain, repo, account,
    )

    exit_chain = FakeDerivativesChain(
        combined_margin=Decimal("10000"), futures_ltp=2950.0, hedge_ltp=45.0
    )
    note = await futures_trading.close_futures_paper_position(
        "RELIANCE.NS", datetime(2026, 2, 5, tzinfo=UTC), exit_chain, account,
    )

    assert note is not None
    assert account.opened[0].status == "closed"
    futures_pnl = (Decimal("2950") - Decimal("2900")) * 500
    # the hedge lost value -- real cost, now counted:
    hedge_pnl = (Decimal("45") - Decimal("50")) * 500
    assert account.opened[0].pnl_amount == futures_pnl + hedge_pnl


@pytest.mark.asyncio
async def test_close_futures_paper_position_is_a_noop_when_nothing_open():
    account = FakeFuturesPaperAccountRepository()
    chain = FakeDerivativesChain(combined_margin=Decimal("10000"))

    note = await futures_trading.close_futures_paper_position(
        "RELIANCE.NS", datetime(2026, 2, 5, tzinfo=UTC), chain, account,
    )

    assert note is None

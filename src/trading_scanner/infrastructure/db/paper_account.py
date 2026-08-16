"""The long-only cash paper-trading account: cash balance and positions."""

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

import libsql_client

from trading_scanner.domain.models import PaperPosition
from trading_scanner.infrastructure.db._shared import add_column_if_missing

_CREATE_PAPER_ACCOUNT_TABLE = """
CREATE TABLE IF NOT EXISTS paper_account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash_balance REAL NOT NULL
)
"""

_CREATE_PAPER_POSITIONS_TABLE = """
CREATE TABLE IF NOT EXISTS paper_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    entry_timestamp TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    capital_allocated REAL NOT NULL,
    exit_timestamp TEXT,
    exit_price REAL,
    pnl_amount REAL,
    status TEXT NOT NULL DEFAULT 'open'
)
"""


class TursoPaperAccountRepository:
    """Persists the paper-trading account's cash balance and positions.

    One account only -- ``paper_account`` always holds a single row
    (``id = 1``), initialized with ``initial_capital`` the first time
    ``get_cash_balance`` runs and left untouched on every call after that.
    """

    def __init__(self, client: libsql_client.Client, initial_capital: Decimal) -> None:
        self._client = client
        self._initial_capital = initial_capital

    async def ensure_schema(self) -> None:
        """Create the paper_account and paper_positions tables if missing."""
        await self._client.execute(_CREATE_PAPER_ACCOUNT_TABLE)
        await self._client.execute(_CREATE_PAPER_POSITIONS_TABLE)
        # 2026-08-14: trailing stop's high-water mark (see domain/models.py's
        # PaperPosition.peak_price docstring) -- migrates any already-deployed
        # table forward.
        await add_column_if_missing(self._client, "paper_positions", "peak_price", "REAL")

    async def get_cash_balance(self) -> Decimal:
        result = await self._client.execute("SELECT cash_balance FROM paper_account WHERE id = 1")
        if not result.rows:
            await self._client.execute(
                "INSERT INTO paper_account (id, cash_balance) VALUES (1, ?)",
                [float(self._initial_capital)],
            )
            return self._initial_capital
        return Decimal(str(result.rows[0][0]))

    async def open_position(self, position: PaperPosition) -> None:
        """Record a new open position and deduct its capital from cash."""
        await self.get_cash_balance()  # Ensures the account row exists.
        await self._client.execute(
            """
            INSERT INTO paper_positions
                (symbol, entry_timestamp, entry_price, quantity, capital_allocated, status,
                 peak_price)
            VALUES (?, ?, ?, ?, ?, 'open', ?)
            """,
            [
                position.symbol,
                position.entry_timestamp.isoformat(),
                float(position.entry_price),
                position.quantity,
                float(position.capital_allocated),
                float(position.entry_price),  # peak starts at entry, only ever moves up
            ],
        )
        await self._client.execute(
            "UPDATE paper_account SET cash_balance = cash_balance - ? WHERE id = 1",
            [float(position.capital_allocated)],
        )

    async def close_position(
        self, symbol: str, exit_timestamp: datetime, exit_price: Decimal
    ) -> PaperPosition | None:
        """Close the most recent open position for symbol, crediting cash back."""
        result = await self._client.execute(
            """
            SELECT id, entry_timestamp, entry_price, quantity, capital_allocated
            FROM paper_positions
            WHERE symbol = ? AND status = 'open'
            ORDER BY entry_timestamp DESC LIMIT 1
            """,
            [symbol],
        )
        if not result.rows:
            return None
        position_id, entry_timestamp, entry_price, quantity, capital_allocated = result.rows[0]
        entry_price = Decimal(str(entry_price))
        capital_allocated = Decimal(str(capital_allocated))
        pnl_amount = (exit_price - entry_price) * quantity
        proceeds = capital_allocated + pnl_amount
        await self._client.execute(
            """
            UPDATE paper_positions
            SET exit_timestamp = ?, exit_price = ?, pnl_amount = ?, status = 'closed'
            WHERE id = ?
            """,
            [exit_timestamp.isoformat(), float(exit_price), float(pnl_amount), position_id],
        )
        await self._client.execute(
            "UPDATE paper_account SET cash_balance = cash_balance + ? WHERE id = 1",
            [float(proceeds)],
        )
        return PaperPosition(
            symbol=symbol,
            entry_timestamp=datetime.fromisoformat(entry_timestamp),
            entry_price=entry_price,
            quantity=quantity,
            capital_allocated=capital_allocated,
            exit_timestamp=exit_timestamp,
            exit_price=exit_price,
            pnl_amount=pnl_amount,
            status="closed",
        )

    async def get_open_positions(self) -> Sequence[PaperPosition]:
        result = await self._client.execute(
            """
            SELECT symbol, entry_timestamp, entry_price, quantity, capital_allocated, peak_price
            FROM paper_positions WHERE status = 'open'
            """
        )
        return [
            PaperPosition(
                symbol=row[0],
                entry_timestamp=datetime.fromisoformat(row[1]),
                entry_price=Decimal(str(row[2])),
                quantity=row[3],
                capital_allocated=Decimal(str(row[4])),
                # Rows opened before this column existed have no stored peak
                # yet -- fall back to entry_price (a correct, conservative
                # starting point: the trailing stop simply hasn't observed
                # any move above entry yet for them either way).
                peak_price=Decimal(str(row[5])) if row[5] is not None else Decimal(str(row[2])),
            )
            for row in result.rows
        ]

    async def update_peak_price(self, symbol: str, peak_price: Decimal) -> None:
        await self._client.execute(
            "UPDATE paper_positions SET peak_price = ? WHERE symbol = ? AND status = 'open'",
            [float(peak_price), symbol],
        )

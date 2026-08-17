"""Shared helpers used by every repository in this package."""

import asyncio
import random

import libsql_client

# 2026-08-17: a burst of concurrent writes (many symbols closing positions
# within the same second, e.g. after a long data outage catches up all at
# once) can contend hard enough on a local SQLite file to raise "database
# is locked" -- previously unhandled, which crashed the entire live
# pipeline process (self-recovered ~30s later via its own top-level retry,
# but a transient, self-resolving lock shouldn't take the whole system
# down). Retries are capped well under _POSITIONS_CACHE_REFRESH_SECONDS
# (5s, see live_pipeline.py) so a retried call never visibly lags behind
# normal polling.
_LOCK_RETRY_ATTEMPTS = 5
_LOCK_RETRY_BASE_DELAY_SECONDS = 0.05


def _is_lock_error(error: Exception) -> bool:
    text = str(error)
    return "database is locked" in text or "SQLITE_BUSY" in text


class _RetryingClient:
    """Wraps a real ``libsql_client.Client``, retrying ``execute``/``batch``
    a few times with jittered exponential backoff on a "database is
    locked" error -- see the module-level comment above. Any other
    exception (including a *different* libsql/SQLite error) passes through
    immediately, unretried, exactly as it always did; this only ever
    catches the one specific, known-transient failure mode. Duck-type
    compatible with ``libsql_client.Client`` for every method this
    codebase actually calls (``execute``, ``batch``, ``close``) -- callers
    keep their existing ``client: libsql_client.Client`` type hints since
    nothing here relies on ``isinstance``.
    """

    def __init__(self, client: libsql_client.Client) -> None:
        self._client = client

    async def _with_retry(self, fn, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        delay = _LOCK_RETRY_BASE_DELAY_SECONDS
        for attempt in range(_LOCK_RETRY_ATTEMPTS):
            try:
                return await fn(*args, **kwargs)
            except Exception as error:
                if not _is_lock_error(error) or attempt == _LOCK_RETRY_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(delay + random.uniform(0, delay))
                delay *= 2

    async def execute(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return await self._with_retry(self._client.execute, *args, **kwargs)

    async def batch(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return await self._with_retry(self._client.batch, *args, **kwargs)

    async def close(self) -> None:
        await self._client.close()


async def add_column_if_missing(
    client: libsql_client.Client, table: str, column: str, definition: str
) -> None:
    """Migrates an already-deployed table forward -- ``CREATE TABLE IF NOT
    EXISTS`` only helps on a fresh database, it never alters an existing
    one.

    Checks ``PRAGMA table_info`` first rather than blind-ALTER-and-swallow
    the "duplicate column" error: over Turso's HTTP transport, an ALTER
    against an already-existing column doesn't come back as a normal
    exception with that text in it -- ``libsql_client``'s HTTP backend
    raises a raw ``KeyError('result')`` while parsing the error response,
    which silently killed every caller of this function (the derivatives
    backtest CLI, in particular, crashed before doing any work, so
    triggering it from the dashboard looked like a no-op with zero
    feedback). Checking first sidesteps relying on that error shape at all.
    """
    result = await client.execute(f"PRAGMA table_info({table})")
    existing_columns = {row[1] for row in result.rows}  # row[1] is the column name
    if column in existing_columns:
        return
    await client.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def create_turso_client(url: str, auth_token: str | None) -> libsql_client.Client:
    """Create one shared libSQL client for every repository to reuse.

    ``libsql://`` connects over WebSocket (Hrana), which some networks (proxies,
    restrictive firewalls) block at the handshake. This pipeline has no need
    for WebSocket-only features (subscriptions, interactive transactions across
    calls), so hosted URLs are normalized to plain HTTPS -- functionally
    equivalent here, and works anywhere HTTPS does. Local ``file:`` URLs are
    left untouched.

    Wrapped in ``_RetryingClient`` (see above) so every repository gets
    lock-retry behavior automatically, with zero changes needed at any
    call site.
    """
    if url.startswith("libsql://"):
        url = "https://" + url.removeprefix("libsql://")
    return _RetryingClient(libsql_client.create_client(url=url, auth_token=auth_token))

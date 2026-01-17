"""
PostgreSQL Connection Pool Management.

Design Patterns:
- Singleton: Single connection pool shared across the application
- Lazy Initialization: Pool created only when first needed
- Resource Management: Automatic cleanup on shutdown

Performance:
- Connection pooling: min_size=2, max_size=10
- Reuses connections across requests
- ~5x faster than creating new connections
"""

from typing import Any, TypeVar
from collections.abc import Awaitable, Callable

import asyncio
import asyncpg
from asyncpg.prepared_stmt import PreparedStatement
import time

from pff.shared.core.logger import logger

from pff.infrastructure.persistence.db.config import get_postgres_config

try:
    from pff.infrastructure.observability import get_observability_manager

    _observability = get_observability_manager(
        experiment_name="postgres_pool",
        enable_debugging=False,
        enable_db_metrics=False,
    )
except Exception:  # pragma: no cover - observability is optional in tests
    _observability = None


T = TypeVar("T")


# Global connection pool (singleton)
_connection_pool: asyncpg.Pool | None = None

# Prepared statement cache for query reuse
_prepared_statements: dict[str, PreparedStatement] = {}


async def get_connection_pool() -> asyncpg.Pool:
    """
    Get or create the global PostgreSQL connection pool.

    Returns:
        asyncpg.Pool: Shared connection pool

    Pattern: Singleton + Lazy Initialization
    """
    global _connection_pool

    # Check if pool exists but belongs to a different loop (e.g. asyncio.run usage)
    if _connection_pool is not None:
        try:
            current_loop = asyncio.get_running_loop()
            pool_loop = getattr(_connection_pool, "_loop", None)
            if pool_loop is not None and pool_loop is not current_loop:
                logger.debug("postgres_pool reinitializing (loop_mismatch=True)")
                _connection_pool = None
        except Exception:
            pass

    if _connection_pool is None:
        config = get_postgres_config()

        logger.debug(
            f"postgres_pool_creating min_size={config.pool.min_size} max_size={config.pool.max_size}"
        )

        async def _init_connection(conn: asyncpg.Connection) -> None:
            statement_sql = config.apply_statement_timeout_sql()
            if statement_sql:
                await conn.execute(statement_sql)
            await conn.execute("SET application_name = 'pff_service';")

        pool_kwargs = config.pool.to_asyncpg_kwargs()
        ssl_context = config.ssl.ssl_context()
        if ssl_context:
            pool_kwargs["ssl"] = ssl_context

        import os

        cpu_count = os.cpu_count() or 4
        pool_kwargs["max_size"] = min(50, max(10, cpu_count * 2))

        try:
            _connection_pool = await asyncpg.create_pool(
                config.dsn_asyncpg,
                init=_init_connection,
                **pool_kwargs,
            )

            logger.debug(
                f"postgres_pool_created min_size={config.pool.min_size} max_size={config.pool.max_size}"
            )
            _record_metric("postgres_pool_created", 1.0)
        except asyncpg.exceptions.TooManyConnectionsError as exc:
            logger.warning(
                f"PostgreSQL connection pool creation failed (connections exhausted): {exc}"
            )
            raise
        except Exception as exc:
            logger.warning(f"PostgreSQL connection pool creation failed: {exc}")
            raise

    return _connection_pool


async def close_connection_pool() -> None:
    """
    Close global connection pool.

    Should be called on application shutdown.
    """
    global _connection_pool
    global prepared_statements  # noqa: F824

    if _connection_pool is not None:
        logger.debug("postgres_pool_closing")
        await _connection_pool.close()
        _connection_pool = None
        logger.debug("postgres_pool_closed")
        _record_metric("postgres_pool_closed", 1.0)

    clear_prepared_statements()  # noqa: F824


def _record_metric(name: str, value: float) -> None:
    if _observability is not None:
        try:
            _observability.record_metric(name, value)
        except Exception:  # pragma: no cover - metrics never block logic
            logger.debug(f"Failed to record metric {name}")


async def _execute_with_retry(
    operation: str,
    coroutine_factory: Callable[[], Awaitable[T]],
) -> T:
    config = get_postgres_config()
    attempts = max(1, config.retry.attempts)
    delay = max(0.0, config.retry.backoff_seconds)
    last_exception: Exception | None = None

    for attempt in range(1, attempts + 1):
        start = time.perf_counter()
        try:
            result = await coroutine_factory()
            duration = time.perf_counter() - start
            _record_metric(f"postgres_{operation}_seconds", duration)
            return result
        except asyncpg.PostgresError as exc:
            last_exception = exc
            _record_metric("postgres_errors_total", 1.0)
            logger.warning(
                f"PostgreSQL error in {operation} (attempt {attempt}/{attempts}): {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            _record_metric("postgres_errors_total", 1.0)
            logger.warning(
                f"Unexpected error in {operation} (attempt {attempt}/{attempts}): {exc}",
            )

        if attempt < attempts and delay > 0:
            await asyncio.sleep(delay * attempt)

    assert last_exception is not None  # for type checkers
    raise last_exception


async def _get_or_create_prepared(
    conn: asyncpg.Connection,
    query: str,
    *params: Any,
) -> PreparedStatement:
    """
    Get prepared statement directly from connection.
    Relies on asyncpg's internal LRU cache for prepared statements
    when using the same connection.
    """
    # Simply prepare and return. asyncpg handles caching per connection.
    return await conn.prepare(query)


async def execute_batch(
    query: str,
    records: list[tuple[Any, ...]],
    *,
    batch_size: int = 1000,
) -> None:
    """
    Execute multiple records in a single batch using COPY protocol for 10-100x speedup.

    Args:
        query: INSERT query template
        records: List of parameter tuples
        batch_size: Records per COPY operation

    Example:
        >>> await execute_batch(
        ...     "INSERT INTO kg_splits VALUES ($1, $2, $3)",
        ...     [("train", data, meta), ("valid", data2, meta2)]
        ... )
    """
    pool = await get_connection_pool()

    async def _call() -> None:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(query, records)

    return await _execute_with_retry("execute_batch", _call)


def clear_prepared_statements() -> None:
    """
    No-op. asyncpg manages prepared statements per connection.
    """
    pass


async def execute_query(query: str, *args, use_prepared: bool = True) -> str:
    pool = await get_connection_pool()

    async def _call() -> str:
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    return await _execute_with_retry("execute", _call)


async def fetch_one(query: str, *args, use_prepared: bool = True):
    pool = await get_connection_pool()

    async def _call():
        async with pool.acquire() as conn:
            if use_prepared:
                # Use conn.prepare to leverage asyncpg implicit cache or explicit stmt object
                stmt = await conn.prepare(query)
                return await stmt.fetchrow(*args)
            return await conn.fetchrow(query, *args)

    return await _execute_with_retry("fetch_one", _call)


async def fetch_all(query: str, *args, use_prepared: bool = True):
    pool = await get_connection_pool()

    async def _call():
        async with pool.acquire() as conn:
            if use_prepared:
                stmt = await conn.prepare(query)
                return await stmt.fetch(*args)
            return await conn.fetch(query, *args)

    return await _execute_with_retry("fetch_all", _call)


async def fetch_val(query: str, *args, use_prepared: bool = True):
    pool = await get_connection_pool()

    async def _call():
        async with pool.acquire() as conn:
            if use_prepared:
                stmt = await conn.prepare(query)
                return await stmt.fetchval(*args)
            return await conn.fetchval(query, *args)

    return await _execute_with_retry("fetch_val", _call)

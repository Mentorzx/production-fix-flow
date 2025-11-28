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

from typing import Awaitable, Callable, Optional, TypeVar

import asyncio
import asyncpg
from loguru import logger
import time

from pff.utils.db import get_postgres_config

try:
    from pff.utils.performance.observability import get_observability_manager

    _observability = get_observability_manager(
        experiment_name="postgres_pool",
        enable_debugging=False,
        enable_db_metrics=False,
    )
except Exception:  # pragma: no cover - observability is optional in tests
    _observability = None


T = TypeVar("T")


# Global connection pool (singleton)
_connection_pool: Optional[asyncpg.Pool] = None


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
                logger.warning("Detected global pool from previous loop. Reinitializing...")
                _connection_pool = None
        except Exception:
            pass

    if _connection_pool is None:
        config = get_postgres_config()

        logger.info(
            "Criando connection pool PostgreSQL...",
            extra={
                "min_size": config.pool.min_size,
                "max_size": config.pool.max_size,
            },
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

        _connection_pool = await asyncpg.create_pool(
            config.dsn_asyncpg,
            init=_init_connection,
            **pool_kwargs,
        )

        logger.success(
            "Connection pool criado",
            extra={
                "min_size": config.pool.min_size,
                "max_size": config.pool.max_size,
            },
        )
        _record_metric("postgres_pool_created", 1.0)

    return _connection_pool


async def close_connection_pool() -> None:
    """
    Close the global connection pool.

    Should be called on application shutdown.
    """
    global _connection_pool

    if _connection_pool is not None:
        logger.info("Fechando connection pool PostgreSQL...")
        await _connection_pool.close()
        _connection_pool = None
        logger.success("Connection pool fechado")
        _record_metric("postgres_pool_closed", 1.0)


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
                f"Erro PostgreSQL em {operation} (tentativa {attempt}/{attempts}): {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            _record_metric("postgres_errors_total", 1.0)
            logger.warning(
                f"Erro inesperado em {operation} (tentativa {attempt}/{attempts}): {exc}",
            )

        if attempt < attempts and delay > 0:
            await asyncio.sleep(delay * attempt)

    assert last_exception is not None  # for type checkers
    raise last_exception


async def execute_query(query: str, *args) -> str:
    """
    Execute a SQL query with parameters.

    Args:
        query: SQL query with $1, $2, etc. placeholders
        *args: Query parameters

    Returns:
        Result status string

    Example:
        >>> await execute_query("DELETE FROM kg_splits WHERE split_name = $1", "train")
    """
    pool = await get_connection_pool()

    async def _call() -> str:
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    return await _execute_with_retry("execute", _call)


async def fetch_one(query: str, *args):
    """
    Fetch a single row from a query.

    Args:
        query: SQL query with $1, $2, etc. placeholders
        *args: Query parameters

    Returns:
        Single row or None

    Example:
        >>> row = await fetch_one("SELECT * FROM kg_splits WHERE id = $1", 123)
    """
    pool = await get_connection_pool()

    async def _call():
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    return await _execute_with_retry("fetch_one", _call)


async def fetch_all(query: str, *args):
    """
    Fetch all rows from a query.

    Args:
        query: SQL query with $1, $2, etc. placeholders
        *args: Query parameters

    Returns:
        List of rows

    Example:
        >>> rows = await fetch_all("SELECT * FROM kg_splits WHERE split_name = $1", "train")
    """
    pool = await get_connection_pool()

    async def _call():
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    return await _execute_with_retry("fetch_all", _call)


async def fetch_val(query: str, *args):
    """
    Fetch a single value from a query.

    Args:
        query: SQL query with $1, $2, etc. placeholders
        *args: Query parameters

    Returns:
        Single value

    Example:
        >>> count = await fetch_val("SELECT COUNT(*) FROM kg_splits")
    """
    pool = await get_connection_pool()

    async def _call():
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    return await _execute_with_retry("fetch_val", _call)

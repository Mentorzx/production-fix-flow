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

from typing import Optional

import asyncpg
from loguru import logger

from pff.config import settings


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

    if _connection_pool is None:
        # Parse DATABASE_URL to asyncpg format
        database_url = settings.DATABASE_URL.replace(
            "postgresql://", ""
        ).replace("postgresql+asyncpg://", "")
        database_url = f"postgresql://{database_url}"

        logger.info("🔌 Criando connection pool PostgreSQL...")

        _connection_pool = await asyncpg.create_pool(
            database_url,
            min_size=2,  # Minimum connections
            max_size=10,  # Maximum connections
            command_timeout=60,  # 60s timeout
            # Connection pool options
            max_queries=50000,  # Max queries per connection before recycling
            max_inactive_connection_lifetime=300,  # 5min idle timeout
        )

        logger.success(f"✅ Connection pool criado (min=2, max=10)")

    return _connection_pool


async def close_connection_pool() -> None:
    """
    Close the global connection pool.

    Should be called on application shutdown.
    """
    global _connection_pool

    if _connection_pool is not None:
        logger.info("🔌 Fechando connection pool PostgreSQL...")
        await _connection_pool.close()
        _connection_pool = None
        logger.success("✅ Connection pool fechado")


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

    async with pool.acquire() as conn:
        result = await conn.execute(query, *args)

    return result


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

    async with pool.acquire() as conn:
        result = await conn.fetchrow(query, *args)

    return result


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

    async with pool.acquire() as conn:
        result = await conn.fetch(query, *args)

    return result


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

    async with pool.acquire() as conn:
        result = await conn.fetchval(query, *args)

    return result

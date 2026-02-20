"""
PostgresRepository - Base class for PostgreSQL Repository Pattern.

Centralizes the boilerplate shared by all Postgres-backed repositories:
    - Lazy connection pool initialization with event-loop mismatch detection
    - Double-checked locking for schema creation
    - Retry-on-missing-table via _execute_with_schema

Design Patterns Applied:
    - Template Method: subclasses override ``_create_schema`` to provide DDL.
    - Dependency Injection: pool and file_manager are injectable.
    - Lazy Init: pool created on first use via ``get_connection_pool``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import asyncpg

from pff.infrastructure.persistence.db.connection import get_connection_pool
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger


class PostgresRepository:
    """Base class for PostgreSQL repositories with lazy pool and schema management.

    Subclasses that need auto-created tables should override ``_create_schema``
    with their DDL statements.  Repositories that depend on externally-managed
    schemas (e.g. ``init-db.sql``) can leave the default no-op.

    Args:
        pool: Optional pre-existing asyncpg pool (for DI / testing).
        file_manager: Optional FileManager instance (for DI / testing).
    """

    def __init__(
        self,
        *,
        pool: Any | None = None,
        file_manager: FileManager | None = None,
    ) -> None:
        """Execute init.



        Args:

            pool: Optional input value.

            file_manager: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.pool: asyncpg.Pool | None = pool
        self._file_manager = file_manager or FileManager()
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Pool lifecycle
    # ------------------------------------------------------------------

    async def _ensure_pool(self) -> None:
        """Lazily acquire a connection pool and ensure schema exists.

        Detects event-loop mismatches (e.g. pool created in a different loop)
        and transparently re-creates the pool when necessary.
        """
        if self.pool is not None:
            try:
                current_loop = asyncio.get_running_loop()
                pool_loop = getattr(self.pool, "_loop", None)
                if (
                    isinstance(pool_loop, asyncio.AbstractEventLoop)
                    and pool_loop is not current_loop
                ):
                    self.pool = None
                    self._schema_ready = False
                    self._schema_lock = asyncio.Lock()
            except RuntimeError:
                pass

        if self.pool is None:
            self.pool = await get_connection_pool()

        await self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema lifecycle
    # ------------------------------------------------------------------

    async def _ensure_schema(self, *, force: bool = False) -> None:
        """Ensure required database tables/indexes exist (double-checked locking).

        Args:
            force: When True, re-runs DDL even if schema was already verified.
        """
        if self.pool is None:
            return
        if force:
            self._schema_ready = False
        if self._schema_ready:
            return

        async with self._schema_lock:
            if self._schema_ready:
                return
            async with self.pool.acquire() as conn:
                await self._create_schema(conn)
            self._schema_ready = True

    async def _create_schema(self, conn: Any) -> None:
        """Override in subclasses to execute DDL statements.

        Called inside a connection context after the double-checked lock
        has been acquired.  The default is a no-op (for repos that rely
        on externally-managed schemas).

        Args:
            conn: Active asyncpg connection.
        """

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    async def _execute_with_schema(
        self,
        operation: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """Execute *operation* with automatic schema recovery.

        If the first attempt raises ``UndefinedTableError``, the schema is
        re-created (``force=True``) and the operation is retried once.

        Args:
            operation: Async callable receiving an ``asyncpg.Connection``.

        Returns:
            Whatever *operation* returns.
        """
        await self._ensure_pool()
        assert self.pool is not None
        try:
            async with self.pool.acquire() as conn:
                return await operation(conn)
        except asyncpg.UndefinedTableError:
            logger.warning(f"{self.__class__.__name__} tables missing - recreating automatically.")
            await self._ensure_schema(force=True)
            async with self.pool.acquire() as conn:
                return await operation(conn)

"""Asynchronous LISTEN/NOTIFY helpers for PostgreSQL."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

from loguru import logger

PayloadHandler = Callable[[Optional[str]], Awaitable[None]]


async def _get_connection_pool():
    """Lazy import to avoid circular dependency with pff.db.connection."""

    from pff.db.connection import get_connection_pool  # Local import prevents circular import

    return await get_connection_pool()


async def notify_postgres(channel: str, payload: Optional[str] = None) -> None:
    """Publish a notification on the given PostgreSQL channel."""

    pool = await _get_connection_pool()
    async with pool.acquire() as conn:
        await conn.execute("SELECT pg_notify($1, $2)", channel, payload or "")

    logger.debug(f"🔔 Notified channel '{channel}'", extra={"payload": payload})


async def register_postgres_listener(channel: str, handler: PayloadHandler) -> None:
    """Register a coroutine handler for PostgreSQL LISTEN notifications."""

    pool = await _get_connection_pool()

    async def _invoke_handler(payload: Optional[str]) -> None:
        try:
            await handler(payload)
        except Exception as exc:  # pragma: no cover - handlers must handle own errors
            logger.error(f"Listener for channel '{channel}' falhou: {exc}")

    def _listener(connection, pid, ch, payload) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        loop.create_task(_invoke_handler(payload))

    await pool.add_listener(channel, _listener)
    logger.debug(f"👂 Registrado listener para canal '{channel}'")

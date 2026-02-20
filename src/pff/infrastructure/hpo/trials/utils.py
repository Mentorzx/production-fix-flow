"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/infrastructure/hpo/trials/utils.py

"""

from __future__ import annotations

import gc

from pff.infrastructure.persistence.db.connection import close_connection_pool
from pff.shared import logger
from pff.shared.acceleration.asyncio_runner import run_coroutine_sync


def cleanup_resources() -> None:
    """Cleanup database pool and trigger garbage collection."""
    try:
        run_coroutine_sync(close_connection_pool())
    except Exception as exc:
        logger.debug(f"Resource cleanup failed to close connection pool: {exc}")
    gc.collect()

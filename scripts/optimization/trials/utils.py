from __future__ import annotations

import asyncio
import gc
from typing import Any

from pff.db.connection import close_connection_pool
from pff.utils.core.logger import logger


def is_cuda_safe() -> bool:
    """
    Check if CUDA is safely available for use.

    Uses the global state from RotatEManager to avoid re-initialization
    attempts that could cause segfaults.
    """
    try:
        from pff.validators.rotate.manager import _CUDA_AVAILABLE

        if _CUDA_AVAILABLE is False:
            return False
        if _CUDA_AVAILABLE is True:
            return True
        return False
    except ImportError:
        return False


def cleanup_resources() -> None:
    """Cleanup resources on exit to prevent segfaults."""
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(close_connection_pool())
        loop.close()
    except Exception:
        pass
    gc.collect()

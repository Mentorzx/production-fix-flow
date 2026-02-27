"""Asyncio execution helpers for sync entrypoints.

Design Pattern: Adapter. Bridges synchronous callers (scripts/services) with async
coroutines without leaking event-loop management into business logic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from pff.shared.core.logging import logger

_R = TypeVar("_R")


def run_coroutine_sync(
    coro: Coroutine[Any, Any, _R], *, timeout_s: float | None = None
) -> _R:
    """Run a coroutine from synchronous code.

    If there is no running event loop, this uses `asyncio.run`. If called while an
    event loop is already running (e.g., notebooks or async orchestrators), the
    coroutine is executed in a dedicated thread via `asyncio.run` to avoid nested
    loop errors.

    Args:
        coro: Coroutine to execute.
        timeout_s: Optional timeout for the thread-backed execution path.

    Returns:
        The coroutine result.

    Raises:
        TimeoutError: If `timeout_s` is exceeded in the thread-backed path.
        Exception: Propagates any exception raised by the coroutine.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    def _run_in_thread() -> _R:
        """Execute coro in a fresh loop within the thread."""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="pff_asyncio_runner"
    ) as executor:
        future = executor.submit(_run_in_thread)
        try:
            return future.result(timeout=timeout_s)
        except TimeoutError:
            logger.warning(
                "Asyncio runner timeout exceeded; cancelling coroutine execution"
            )
            future.cancel()
            raise


def run_coroutine_in_new_loop(
    coro: Coroutine[Any, Any, _R],
    *,
    drain_pending_tasks: bool = True,
    timeout_s: float | None = None,
) -> _R:
    """Run a coroutine in a dedicated event loop and optionally drain pending tasks.

    This is useful for async pipelines that spawn background tasks which must be
    allowed to finish before the loop is closed (e.g., graceful shutdown hooks).

    When invoked from within an already-running event loop, the execution happens
    in a dedicated thread to prevent nested-loop errors.

    Args:
        coro: Coroutine to execute.
        drain_pending_tasks: If True, gather and await pending tasks after `coro`
            completes, using `return_exceptions=True`.
        timeout_s: Optional timeout for the thread-backed execution path.

    Returns:
        The coroutine result.

    Raises:
        TimeoutError: If `timeout_s` is exceeded in the thread-backed path.
        Exception: Propagates any exception raised by the coroutine.
    """

    def _run() -> _R:
        """Execute run.



        Returns:

            Return value produced by the callable.

        """

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            if drain_pending_tasks:
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            return result
        finally:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run()

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="pff_asyncio_runner"
    ) as executor:
        future = executor.submit(_run)
        try:
            return future.result(timeout=timeout_s)
        except TimeoutError:
            logger.warning(
                "Asyncio runner timeout exceeded; cancelling coroutine execution"
            )
            future.cancel()
            raise

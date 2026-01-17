"""
Tests for pff.shared.acceleration.asyncio_runner.

Validates the sync/async bridge used by scripts (e.g., HPO) to avoid leaking
event-loop management into business logic.
"""

from __future__ import annotations

import asyncio

import pytest

from pff.shared.acceleration.asyncio_runner import (
    run_coroutine_in_new_loop,
    run_coroutine_sync,
)


@pytest.mark.unit
def test_run_coroutine_sync_without_running_loop() -> None:
    """run_coroutine_sync executes coroutine via asyncio.run when no loop is running."""

    async def _coro() -> int:
        await asyncio.sleep(0)
        return 7

    assert run_coroutine_sync(_coro()) == 7


@pytest.mark.asyncio
async def test_run_coroutine_sync_with_running_loop() -> None:
    """run_coroutine_sync executes coroutine in a thread when a loop is running."""

    async def _coro() -> int:
        await asyncio.sleep(0)
        return 11

    assert run_coroutine_sync(_coro(), timeout_s=2.0) == 11


@pytest.mark.unit
def test_run_coroutine_in_new_loop_drains_pending_tasks() -> None:
    """run_coroutine_in_new_loop drains background tasks before closing the loop."""
    events: list[str] = []

    async def _background() -> None:
        await asyncio.sleep(0.01)
        events.append("done")

    async def _main() -> str:
        asyncio.create_task(_background())
        return "ok"

    assert run_coroutine_in_new_loop(_main(), drain_pending_tasks=True) == "ok"
    assert events == ["done"]

"""Async I/O primitives for file_manager package.

Provides consistent async file operations using aiofile.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiofile

from pff.shared.acceleration.asyncio_runner import (
    run_coroutine_sync as _run_coroutine_sync,
)


async def read_async_content(path: Path, *, chunk_size: int | None = None) -> bytes:
    """Read file content asynchronously using aiofile.

    Args:
        path: File path to read.
        chunk_size: Optional chunk size for reading.

    Returns:
        File content as bytes.
    """
    async with aiofile.async_open(path, "rb") as f:
        if chunk_size:
            chunks = []
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        return await f.read()  # type: ignore[no-any-return]


async def write_async_bytes(path: Path, data: bytes) -> None:
    """Write bytes to file asynchronously using aiofile.

    Args:
        path: Destination file path.
        data: Bytes to write.
    """
    async with aiofile.async_open(path, "wb") as f:
        await f.write(data)


async def write_async_text(
    path: Path, content: str, *, encoding: str = "utf-8"
) -> None:
    """Write text to file asynchronously using aiofile.

    Args:
        path: Destination file path.
        content: Text content to write.
        encoding: Text encoding (default utf-8).
    """
    async with aiofile.async_open(path, "w", encoding=encoding) as f:
        await f.write(content)


async def async_ensure_dir(path: Path) -> None:
    """Ensure parent directory exists for a file path (async).

    Uses asyncio.to_thread to avoid blocking.

    Args:
        path: File path whose parent directory should exist.
    """
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)


def run_coroutine_sync(coro):
    """Run a coroutine synchronously.

    Handles the case where we're already in an async context.

    Args:
        coro: Coroutine to run.

    Returns:
        Result of the coroutine.
    """
    return _run_coroutine_sync(coro)

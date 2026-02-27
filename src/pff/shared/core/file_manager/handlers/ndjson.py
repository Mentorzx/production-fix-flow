"""NDJSON/JSONL file handler."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

import polars as pl

from ..async_io import async_ensure_dir, read_async_content
from ..utils import ensure_dir
from .base import FileHandler
from .tabular_utils import read_tabular


class NDJSONHandler(FileHandler):
    """Handler for newline-delimited JSON files (.ndjson, .jsonl)."""

    def read(
        self, path: Path | io.BytesIO, **kwargs: Any
    ) -> pl.DataFrame | pl.LazyFrame:
        """Read NDJSON file into a Polars DataFrame.

        Args:
            path: NDJSON file path or in-memory buffer.
            **kwargs: Arguments forwarded to pl.read_ndjson.

        Returns:
            Polars DataFrame.
        """
        lazy = bool(kwargs.pop("lazy", False))
        streaming = kwargs.pop("streaming", None)

        return read_tabular(
            path,
            lazy=lazy,
            streaming=streaming,
            scan_fn=pl.scan_ndjson,
            read_fn=pl.read_ndjson,
            **kwargs,
        )

    def save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Save a Polars DataFrame as NDJSON."""
        ensure_dir(path)
        if isinstance(obj, pl.LazyFrame):
            obj.collect().write_ndjson(path, **kwargs)
        else:
            obj.write_ndjson(path, **kwargs)

    async def async_read(
        self, path: Path, **kwargs: Any
    ) -> pl.DataFrame | pl.LazyFrame:
        """Asynchronously read NDJSON file."""
        chunk_size = kwargs.pop("chunk_size", None)

        if kwargs.get("lazy") or kwargs.get("streaming"):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self.read(path, **kwargs))

        if chunk_size is not None:
            try:
                chunk_size = int(chunk_size)
            except (TypeError, ValueError):
                chunk_size = None

        content = await read_async_content(path, chunk_size=chunk_size)
        buffer = io.BytesIO(content)
        return self.read(buffer, **kwargs)

    async def async_save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Asynchronously save as NDJSON."""
        await async_ensure_dir(path)
        await asyncio.to_thread(self.save, obj, path, **kwargs)

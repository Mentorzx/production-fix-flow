"""Excel file handler (.xls, .xlsx)."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

import polars as pl

from ..async_io import async_ensure_dir
from ..utils import ensure_dir
from .base import FileHandler


class ExcelHandler(FileHandler):
    """Handler for Excel files using Polars."""

    def read(self, path: Path | io.BytesIO, **kwargs: Any) -> pl.DataFrame:
        """Read an Excel file into a Polars DataFrame.

        Args:
            path: Excel file path or in-memory buffer.
            **kwargs: Arguments forwarded to pl.read_excel.

        Returns:
            Polars DataFrame.
        """
        return pl.read_excel(path, **kwargs)  # type: ignore[no-any-return]

    def save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Save a Polars DataFrame to Excel file.

        Args:
            obj: Polars DataFrame to save.
            path: Destination file path.
            **kwargs: Arguments forwarded to write_excel.
        """
        ensure_dir(path)
        if isinstance(obj, pl.LazyFrame):
            obj.collect().write_excel(path, **kwargs)
        else:
            obj.write_excel(path, **kwargs)

    async def async_read(self, path: Path, **kwargs: Any) -> pl.DataFrame:
        """Async read delegates to sync implementation."""
        return self.read(path, **kwargs)

    async def async_save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Async save delegates to sync implementation."""
        await async_ensure_dir(path)
        await asyncio.to_thread(self.save, obj, path, **kwargs)

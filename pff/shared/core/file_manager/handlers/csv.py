"""CSV and TSV file handlers."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

import polars as pl
from polars.exceptions import ComputeError

from ...logging import logger
from ..async_io import async_ensure_dir, read_async_content
from ..utils import ensure_dir
from .base import FileHandler
from .tabular_utils import read_tabular


def _detect_csv_dialect(raw: bytes) -> tuple[str, str]:
    """Detect CSV separator and encoding from raw bytes."""
    from charset_normalizer import detect as detect_encoding

    result = detect_encoding(raw[:4096])
    encoding: str = (result.get("encoding") or "utf-8") if result else "utf-8"

    try:
        sample = raw[:4096].decode(encoding, errors="ignore")
    except Exception:
        sample = raw[:4096].decode("utf-8", errors="ignore")

    lines = sample.split("\n")[:5]

    comma_count = sum(line.count(",") for line in lines)
    semicolon_count = sum(line.count(";") for line in lines)
    tab_count = sum(line.count("\t") for line in lines)

    if tab_count >= comma_count and tab_count >= semicolon_count:
        sep = "\t"
    elif semicolon_count > comma_count:
        sep = ";"
    else:
        sep = ","

    return sep, encoding


class CSVHandler(FileHandler):
    """Handler for CSV and TSV files using Polars."""

    def read(self, path: Path | io.BytesIO, **kwargs: Any) -> pl.DataFrame | pl.LazyFrame:
        """Read a CSV file or buffer into a Polars DataFrame, with dialect fallback."""
        lazy = bool(kwargs.pop("lazy", False))
        streaming = kwargs.pop("streaming", None)

        if isinstance(path, Path) and path.suffix.lower() == ".tsv":
            kwargs["separator"] = "\t"
            kwargs.setdefault("has_header", False)
            kwargs.setdefault("truncate_ragged_lines", True)
            kwargs.setdefault("ignore_errors", True)

        if isinstance(path, io.BytesIO):
            raw = path.read()
            sep, encoding = _detect_csv_dialect(raw)
            path.seek(0)
            if "separator" not in kwargs:
                kwargs["separator"] = sep
            kwargs.setdefault("encoding", encoding)
            return pl.read_csv(path, **kwargs)
        else:
            kwargs.setdefault("encoding", "utf-8")

        try:
            return read_tabular(
                path,
                lazy=lazy,
                streaming=streaming,
                scan_fn=pl.scan_csv,
                read_fn=pl.read_csv,
                **kwargs,
            )
        except (ComputeError, pl.exceptions.PolarsError) as e:
            logger.warning(f"Initial CSV/TSV read failed: {e}")
            kwargs["truncate_ragged_lines"] = True
            kwargs["ignore_errors"] = True
            return read_tabular(
                path,
                lazy=False,
                streaming=streaming,
                scan_fn=pl.scan_csv,
                read_fn=pl.read_csv,
                **kwargs,
            )

    def save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Save a Polars DataFrame or LazyFrame as CSV."""
        ensure_dir(path)
        if isinstance(obj, pl.LazyFrame):
            obj.sink_csv(path, **kwargs)
        else:
            obj.write_csv(path, **kwargs)

    async def async_read(self, path: Path, **kwargs: Any) -> pl.DataFrame | pl.LazyFrame:
        """Asynchronously read a CSV file into a Polars DataFrame."""
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
        """Asynchronously save a Polars DataFrame as CSV."""
        await async_ensure_dir(path)
        await asyncio.to_thread(self.save, obj, path, **kwargs)

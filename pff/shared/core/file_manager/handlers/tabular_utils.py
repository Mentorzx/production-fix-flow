"""Shared tabular handler helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Callable

import polars as pl

from ..config import get_streaming_threshold_bytes


def read_tabular(
    path: Path | Any,
    *,
    lazy: bool,
    streaming: bool | None,
    scan_fn: Callable[..., pl.LazyFrame],
    read_fn: Callable[..., pl.DataFrame],
    **kwargs: Any,
) -> pl.DataFrame | pl.LazyFrame:
    """Read or scan a tabular source with streaming-aware defaults."""
    if isinstance(path, Path):
        if streaming is None:
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                size = None
            if size and size > get_streaming_threshold_bytes():
                streaming = True
        if lazy or streaming:
            scan = scan_fn(str(path), **kwargs)
            if lazy:
                return scan
            return scan.collect(engine="streaming")
    return read_fn(path, **kwargs)

"""Handler registry with thread-safe factory pattern.

This module provides:
- FileHandler base class
- All concrete handler implementations
- Thread-safe handler cache with get_handler()
- HANDLER_FACTORIES and SUPPORTED_EXTS constants
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from .arrow_ipc import ArrowIPCHandler
from .base import FileHandler
from .binary import BinHandler, NumPyHandler, PickleHandler
from .csv import CSVHandler
from .excel import ExcelHandler
from .json import JSONHandler
from .ndjson import NDJSONHandler
from .parquet import ParquetHandler
from .text import TextHandler
from .yaml import YAMLHandler
from .zstd import ZstdHandler

if TYPE_CHECKING:
    pass


HANDLER_FACTORIES: dict[str, type[FileHandler]] = {
    ".csv": CSVHandler,
    ".tsv": CSVHandler,
    ".parquet": ParquetHandler,
    ".pq": ParquetHandler,
    ".parq": ParquetHandler,
    ".arrow": ArrowIPCHandler,
    ".ipc": ArrowIPCHandler,
    ".feather": ArrowIPCHandler,
    ".ndjson": NDJSONHandler,
    ".jsonl": NDJSONHandler,
    ".json": JSONHandler,
    ".yaml": YAMLHandler,
    ".yml": YAMLHandler,
    ".txt": TextHandler,
    ".pkl": PickleHandler,
    ".xls": ExcelHandler,
    ".xlsx": ExcelHandler,
    ".bin": BinHandler,
    ".pt": BinHandler,
    ".npy": NumPyHandler,
    ".zst": ZstdHandler,
    ".zstd": ZstdHandler,
}


SUPPORTED_EXTS: frozenset[str] = frozenset(HANDLER_FACTORIES.keys())


_HANDLER_CACHE: dict[str, FileHandler] = {}
_HANDLER_LOCK = threading.Lock()


def get_handler(suffix: str) -> FileHandler | None:
    """Get or create a handler for the given suffix (thread-safe).

    Uses a global cache to avoid creating multiple instances of the same handler.
    Handlers are expected to be stateless.

    Args:
        suffix: File extension with dot (e.g., ".csv").

    Returns:
        Handler instance or None if extension is not supported.
    """
    suffix_lower = suffix.lower()

    if suffix_lower in _HANDLER_CACHE:
        return _HANDLER_CACHE[suffix_lower]

    with _HANDLER_LOCK:
        if suffix_lower in _HANDLER_CACHE:
            return _HANDLER_CACHE[suffix_lower]

        factory = HANDLER_FACTORIES.get(suffix_lower)
        if factory is None:
            return None

        handler = factory()
        _HANDLER_CACHE[suffix_lower] = handler
        return handler


def clear_handler_cache() -> None:
    """Clear the handler cache (useful for testing)."""
    with _HANDLER_LOCK:
        _HANDLER_CACHE.clear()


__all__ = [
    "FileHandler",
    "CSVHandler",
    "ParquetHandler",
    "JSONHandler",
    "YAMLHandler",
    "TextHandler",
    "BinHandler",
    "PickleHandler",
    "NumPyHandler",
    "ExcelHandler",
    "NDJSONHandler",
    "ZstdHandler",
    "ArrowIPCHandler",
    "HANDLER_FACTORIES",
    "SUPPORTED_EXTS",
    "get_handler",
    "clear_handler_cache",
]

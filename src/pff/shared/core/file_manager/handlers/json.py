"""JSON file handler using orjson for maximum performance."""

from __future__ import annotations

import io
import mmap
from pathlib import Path
from typing import Any

import orjson

from ..async_io import async_ensure_dir, read_async_content, write_async_bytes
from ..utils import ensure_dir
from .base import FileHandler


class JSONHandler(FileHandler):
    """Estado da arte JSON handler usando orjson e memory mapping.

    Performance:
    - Uses orjson for fastest JSON encoding/decoding
    - Uses memory mapping (mmap) for zero-copy file reading
    - Optimized for low-latency and high-throughput scenarios
    """

    def read(self, path: Path | io.BytesIO, **kwargs: Any) -> Any:
        """Deserialize JSON content using orjson + mmap for speed.

        Returns native Python types (dict, list, int, float, str, bool, None).
        """
        if isinstance(path, io.BytesIO):
            return orjson.loads(path.read())

        path_obj = Path(path)
        with open(path_obj, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return orjson.loads(memoryview(mm))

    def save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Serialize object to JSON file using orjson."""
        ensure_dir(path)

        path.write_bytes(orjson.dumps(obj))

    async def async_read(self, path: Path, **kwargs: Any) -> Any:
        """Asynchronously deserialize JSON content."""
        chunk_size = kwargs.get("chunk_size")

        raw = await read_async_content(path, chunk_size=chunk_size)
        return orjson.loads(raw)

    async def async_save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Asynchronously serialize object to JSON."""
        await async_ensure_dir(path)
        encoded = orjson.dumps(obj)
        await write_async_bytes(path, encoded)

    def load_bytes(self, raw: bytes, **kwargs: Any) -> Any:
        """Decode JSON from raw bytes directly."""
        return orjson.loads(raw)

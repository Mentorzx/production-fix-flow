"""JSON file handler using msgspec for maximum performance."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import msgspec

from .base import FileHandler
from ..utils import ensure_dir
from ..async_io import read_async_content, write_async_bytes, async_ensure_dir


class JSONHandler(FileHandler):
    """Estado da arte JSON handler usando msgspec para máxima performance.

    Performance:
    - Uses msgspec for fast JSON encoding/decoding
    - Memory-efficient compared to alternatives
    - Automatic key caching for repeated structures

    Maintains 100% API compatibility with previous implementation.
    """

    def __init__(self) -> None:
        """Create reusable encoder/decoder instances for better performance."""
        self._encoder = msgspec.json.Encoder()
        self._decoder = msgspec.json.Decoder()

    def read(self, path: Path | io.BytesIO, **kwargs: Any) -> Any:
        """Deserialize JSON content using msgspec for maximum performance.

        Returns native Python types (dict, list, int, float, str, bool, None).
        """
        if isinstance(path, io.BytesIO):
            return self._decoder.decode(path.read())
        return self._decoder.decode(Path(path).read_bytes())

    def save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Serialize object to JSON file using msgspec."""
        ensure_dir(path)
        path.write_bytes(self._encoder.encode(obj))

    async def async_read(self, path: Path, **kwargs: Any) -> Any:
        """Asynchronously deserialize JSON content using real async I/O."""
        chunk_size = kwargs.get("chunk_size")
        raw = await read_async_content(path, chunk_size=chunk_size)
        return self._decoder.decode(raw)

    async def async_save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Asynchronously serialize object to JSON using real async I/O."""
        await async_ensure_dir(path)
        encoded = self._encoder.encode(obj)
        await write_async_bytes(path, encoded)

    def load_bytes(self, raw: bytes, **kwargs: Any) -> Any:
        """Decode JSON from raw bytes directly."""
        return self._decoder.decode(raw)

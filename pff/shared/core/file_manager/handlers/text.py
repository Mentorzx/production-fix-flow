"""Text file handler with intelligent encoding detection."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from charset_normalizer import detect

from .base import FileHandler
from ..utils import ensure_dir
from ..async_io import read_async_content, write_async_text, async_ensure_dir


class TextHandler(FileHandler):
    """Estado da arte text handler com detecção de encoding inteligente.

    Uses charset-normalizer for encoding detection.
    """

    def read(self, path: Path | io.BytesIO, **kwargs: Any) -> str:
        """Read text content with intelligent encoding detection.

        Args:
            path: Text file path or in-memory buffer.
            **kwargs: Reserved for future options.

        Returns:
            Decoded text string.
        """
        if isinstance(path, io.BytesIO):
            raw = path.read()
        else:
            raw = Path(path).read_bytes()

        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            detection = detect(raw)
            encoding = detection.get("encoding") or "latin-1"
            return raw.decode(encoding, errors="ignore")

    def save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Save text content to a file, creating dirs if needed."""
        ensure_dir(path)
        path.write_text(str(obj), encoding="utf-8")

    async def async_read(self, path: Path, **kwargs: Any) -> str:
        """Asynchronously read text content with encoding detection."""
        raw = await read_async_content(path, chunk_size=kwargs.get("chunk_size"))

        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            detection = detect(raw)
            encoding = detection.get("encoding") or "latin-1"
            return raw.decode(encoding, errors="ignore")

    async def async_save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Asynchronously save text content."""
        await async_ensure_dir(path)
        await write_async_text(path, str(obj), encoding="utf-8")

    def load_bytes(self, raw: bytes, **kwargs: Any) -> str:
        """Decode text from raw bytes."""
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            detection = detect(raw)
            encoding = detection.get("encoding") or "latin-1"
            return raw.decode(encoding, errors="ignore")

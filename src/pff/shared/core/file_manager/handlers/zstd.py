"""Zstandard compressed file handler."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from ..async_io import async_ensure_dir
from ..utils import ensure_dir
from .base import FileHandler

try:
    import zstandard as zstd
except ImportError:
    zstd = None  # type: ignore[assignment]


def decompress_zstd_bytes(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> bytes:
    """Decompress a zstd file using streaming to manage memory.

    Args:
        path: Path to .zst/.zstd file.
        chunk_size: Size of chunks to read during decompression.

    Returns:
        Decompressed bytes.

    Raises:
        ImportError: If zstandard is not installed.
    """
    if zstd is None:
        raise ImportError("zstandard is required to read .zst/.zstd files")

    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        with dctx.stream_reader(f) as reader:
            output = bytearray()
            while True:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break
                output.extend(chunk)
    return bytes(output)


def decompress_zstd_to_file(
    source_path: Path, dest_path: Path, *, chunk_size: int = 16 * 1024 * 1024
) -> None:
    """Decompress a zstd file to a destination file using streaming.

    Args:
        source_path: Path to .zst/.zstd file.
        dest_path: Path to write decompressed content to.
        chunk_size: Size of chunks for streaming.
    """
    if zstd is None:
        raise ImportError("zstandard is required to read .zst/.zstd files")

    dctx = zstd.ZstdDecompressor()
    with open(source_path, "rb") as source, open(dest_path, "wb") as dest:
        with dctx.stream_reader(source) as reader:
            while True:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break
                dest.write(chunk)


def resolve_zstd_inner_suffix(path: Path, handler_kwargs: dict[str, Any]) -> str | None:
    """Extract inner extension for zstd-compressed files.

    Note: This function intentionally mutates handler_kwargs by removing 'inner_suffix'
    to prevent it from being passed through to inner handlers.

    Args:
        path: Path to the .zst/.zstd file.
        handler_kwargs: Kwargs dict that may contain 'inner_suffix' override.

    Returns:
        Inner file extension (e.g., ".json") or None.
    """
    override = handler_kwargs.pop("inner_suffix", None)
    if isinstance(override, str) and override:
        return override if override.startswith(".") else f".{override}"
    suffixes = path.suffixes
    if len(suffixes) >= 2 and suffixes[-1] in {".zst", ".zstd"}:
        return suffixes[-2]
    return None


class ZstdHandler(FileHandler):
    """Handler for Zstandard compressed files (.zst, .zstd)."""

    def read(self, path: Path | io.BytesIO, **kwargs: Any) -> Any:
        """Decompress and read zstd file content.

        Args:
            path: Path to zstd file or BytesIO buffer.
            inner_suffix: Override for inner file extension detection.
            **kwargs: Other args passed to inner handler.

        Returns:
            Decompressed and parsed content.
        """
        if zstd is None:
            raise ImportError("zstandard is required to read .zst/.zstd files")

        if isinstance(path, io.BytesIO):
            dctx = zstd.ZstdDecompressor()
            raw = dctx.decompress(path.read())
            inner_suffix = kwargs.pop("inner_suffix", None)
            if inner_suffix:
                from . import get_handler

                handler = get_handler(inner_suffix)
                if handler:
                    return handler.load_bytes(raw, **kwargs)
            return raw

        inner_suffix = resolve_zstd_inner_suffix(path, kwargs)
        raw = decompress_zstd_bytes(path)

        if inner_suffix:
            from . import get_handler

            handler = get_handler(inner_suffix)
            if handler:
                return handler.load_bytes(raw, **kwargs)
        return raw

    def save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Compress and save content as zstd file.

        Args:
            obj: Content to save (bytes or serializable object).
            path: Destination path.
            **kwargs: Compression options.
        """
        if zstd is None:
            raise ImportError("zstandard is required to write .zst/.zstd files")

        ensure_dir(path)

        if isinstance(obj, bytes):
            data = obj
        elif isinstance(obj, str):
            data = obj.encode("utf-8")
        else:
            import msgspec

            data = msgspec.json.encode(obj)

        cctx = zstd.ZstdCompressor(level=kwargs.get("level", 3))
        with path.open("wb") as f:
            f.write(cctx.compress(data))

    async def async_read(self, path: Path, **kwargs: Any) -> Any:
        """Async read delegates to sync implementation."""
        return self.read(path, **kwargs)

    async def async_save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Async save delegates to sync implementation."""
        await async_ensure_dir(path)
        self.save(obj, path, **kwargs)

    def load_bytes(self, raw: bytes, **kwargs: Any) -> Any:
        """Decompress and parse zstd bytes."""
        if zstd is None:
            raise ImportError("zstandard is required to read .zst/.zstd files")

        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(raw)

        inner_suffix = kwargs.pop("inner_suffix", None)
        if inner_suffix:
            from . import get_handler

            handler = get_handler(inner_suffix)
            if handler:
                return handler.load_bytes(decompressed, **kwargs)
        return decompressed

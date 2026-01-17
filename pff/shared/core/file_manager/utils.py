"""Shared utilities for file_manager package.

Contains helper functions used across multiple modules:
- Hashing and checksums
- Encoding detection
- Path manipulation
- JSON safety conversion
- Memory mapping
"""

from __future__ import annotations

import hashlib
import mmap
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from collections.abc import Iterator, Mapping

import msgspec
import orjson
from charset_normalizer import detect

from ..logger import logger
from .config import get_encoder_buffer_size


def compute_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute SHA256 hash of a file using chunked readinto to reduce allocations."""
    hasher = hashlib.sha256()
    buf = bytearray(chunk_size)
    mv = memoryview(buf)
    with path.open("rb") as f:
        while True:
            read_len = f.readinto(buf)
            if not read_len:
                break
            hasher.update(mv[:read_len])
    return hasher.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """Compute SHA256 hash of in-memory bytes."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def compute_sha256_buffer(data: bytes | bytearray | memoryview | mmap.mmap) -> str:
    """Compute SHA256 hash of a bytes-like buffer without extra copies."""
    hasher = hashlib.sha256()
    mv = data if isinstance(data, memoryview) else memoryview(data)
    hasher.update(mv)
    return hasher.hexdigest()


def detect_encoding_sample(path: Path, sample_size: int = 8192) -> str | None:
    """Detect file encoding by reading only a sample.

    Uses f.read() instead of read_bytes() to avoid loading entire file.

    Args:
        path: File path to sample.
        sample_size: Number of bytes to read for detection.

    Returns:
        Detected encoding string or None.
    """
    try:
        with path.open("rb") as f:
            sample = f.read(sample_size)
        if not sample:
            return None
        result = detect(sample)
        return result.get("encoding") if result else None
    except Exception:
        return None


def fast_suffix(name: str) -> str:
    """Extract lowercase file suffix from a filename.

    Faster than Path().suffix for simple extension extraction.

    Args:
        name: Filename or path string.

    Returns:
        Lowercase extension with dot (e.g., ".csv") or empty string.
    """
    if not name:
        return ""
    last_sep = name.rfind("/")
    dot = name.rfind(".")
    if dot == -1 or dot < last_sep:
        return ""
    return name[dot:].lower()


def ensure_dir(path: Path) -> None:
    """Ensure parent directory exists for a file path."""
    path.parent.mkdir(parents=True, exist_ok=True)


async def async_ensure_dir(path: Path) -> None:
    """Async version of ensure_dir using asyncio.to_thread."""
    import asyncio

    await asyncio.to_thread(ensure_dir, path)


@contextmanager
def memory_map_file(path: Path) -> Iterator[mmap.mmap]:
    """Context manager for memory-mapped file reading.

    Provides efficient access to large files without loading into RAM.

    Args:
        path: File path to memory map.

    Yields:
        Memory-mapped file object.
    """
    with path.open("rb") as f:
        if f.seek(0, 2) == 0:
            # Empty file - yield empty bytes view
            mm = mmap.mmap(-1, 0, access=mmap.ACCESS_READ)
            try:
                yield mm
            finally:
                mm.close()
        else:
            f.seek(0)
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                try:
                    if hasattr(mm, "madvise") and hasattr(mmap, "MADV_SEQUENTIAL"):
                        mm.madvise(mmap.MADV_SEQUENTIAL)
                except Exception:
                    pass
                yield mm


def make_json_safe(value: Any) -> Any:
    """Convert value to JSON-safe type, handling special types.

    Handles ruamel.yaml special types, Path objects, datetime, etc.
    Logs a debug message when converting unknown types to string.

    Args:
        value: Any Python value.

    Returns:
        JSON-serializable value.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float):
        return float(value) if type(value) is not float else value
    if isinstance(value, int):
        return int(value) if type(value) is not int else value
    if isinstance(value, str):
        return str(value) if type(value) is not str else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {
            key: make_json_safe(val)
            for key, val in value.__dict__.items()
            if not key.startswith("_")
        }
    logger.debug(
        f"Converting unknown type {type(value).__name__} to string for JSON safety"
    )
    return str(value)


def read_manifest(path: Path) -> dict[str, Any] | None:
    """Read manifest JSON, distinguishing missing files from corrupted content.

    Args:
        path: Path to manifest JSON file.

    Returns:
        Parsed manifest dict or None if not found/corrupted.
    """
    if not path.exists():
        return None
    try:
        data = orjson.loads(path.read_bytes())
        return data if isinstance(data, dict) else None
    except orjson.JSONDecodeError as exc:
        logger.warning(f"Corrupted manifest at {path}: {exc}")
        return None
    except OSError as exc:
        logger.warning(f"Failed to read manifest at {path}: {exc}")
        return None


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write manifest JSON to file.

    Args:
        path: Destination path.
        manifest: Manifest dictionary to write.
    """
    ensure_dir(path)
    path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))


def read_raw_bytes(raw_parquet_path: Path) -> bytes:
    """Read and concatenate all chunks from a RAW parquet file.

    Args:
        raw_parquet_path: Path to RAW layer parquet.

    Returns:
        Complete file bytes.
    """
    import pyarrow.parquet as pq

    table = pq.ParquetFile(raw_parquet_path)
    output = bytearray()
    columns = ["chunk_bytes"] if "chunk_bytes" in table.schema.names else ["chunk"]
    for batch in table.iter_batches(columns=columns):
        col = batch.column(0)
        for chunk in col:
            output.extend(chunk.as_py() or b"")
    return bytes(output)


def get_index_manifest_path(
    cache_root: Path,
    source_path: Path,
    stat_sig: tuple[int, int],
) -> Path:
    """Generate index manifest path for cache lookup.

    Args:
        cache_root: Root directory for parquet cache.
        source_path: Original source file path.
        stat_sig: Tuple of (mtime_ns, size_bytes).

    Returns:
        Path to manifest JSON file.
    """
    name_safe = source_path.name.replace(" ", "_")[:64]
    return cache_root / ".index" / f"{name_safe}_{stat_sig[0]}_{stat_sig[1]}.json"


# Msgspec encoder/decoder utilities
_MSGSPEC_TLS = threading.local()
_msgspec_encoder = msgspec.json.Encoder()
_msgspec_decoder = msgspec.json.Decoder()
_msgpack_encoder = msgspec.msgpack.Encoder()


def get_json_encoder() -> msgspec.json.Encoder:
    """Return a process-aware msgspec JSON encoder."""
    pid = os.getpid()
    encoder = getattr(_MSGSPEC_TLS, "json_encoder", None)
    encoder_pid = getattr(_MSGSPEC_TLS, "json_encoder_pid", None)
    if encoder is None or encoder_pid != pid:
        encoder = msgspec.json.Encoder()
        _MSGSPEC_TLS.json_encoder = encoder
        _MSGSPEC_TLS.json_encoder_pid = pid
    return encoder


def get_json_buffer() -> bytearray:
    """Return a reusable JSON buffer for msgspec encoding."""
    buffer = getattr(_MSGSPEC_TLS, "json_buffer", None)
    if buffer is None:
        buffer = bytearray(get_encoder_buffer_size())
        _MSGSPEC_TLS.json_buffer = buffer
    return buffer


def encode_json(obj: Any, *, encoder: msgspec.json.Encoder | None = None) -> bytes:
    """Encode object to JSON bytes using msgspec.

    Args:
        obj: Object to encode.
        encoder: Optional custom encoder.

    Returns:
        JSON bytes.
    """
    enc = encoder or _msgspec_encoder
    return enc.encode(obj)


def decode_json(data: bytes) -> Any:
    """Decode JSON bytes using msgspec.

    Args:
        data: JSON bytes.

    Returns:
        Decoded Python object.
    """
    return _msgspec_decoder.decode(data)


def encode_msgpack(obj: Any) -> bytes:
    """Encode object to msgpack bytes.

    Args:
        obj: Object to encode.

    Returns:
        Msgpack bytes.
    """
    return _msgpack_encoder.encode(obj)

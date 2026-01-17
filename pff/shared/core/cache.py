from __future__ import annotations

import atexit
import functools
import gzip
import importlib
import inspect
import os
import pickle
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from hashlib import blake2b, sha256
from multiprocessing.managers import DictProxy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, ParamSpec, Protocol, TypeVar, cast, overload
from collections.abc import Callable, Iterable
from urllib.parse import urlsplit

import orjson
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from ..core.logger import logger
from pff.config import CACHE_CONFIG_PATH, settings

try:
    import msgspec  # type: ignore

    MSGSPEC_AVAILABLE = True
except Exception:
    msgspec = None
    MSGSPEC_AVAILABLE = False

"""
High-performance caching module with disk persistence, memory caching, and HTTP template caching.

This module provides a comprehensive caching solution with multiple layers:
- Disk-based persistent cache with optional compression
- In-memory LRU cache for fast access
- Specialized HTTP template cache for API request patterns
"""

P = ParamSpec("P")
R = TypeVar("R")


def _load_cache_settings() -> dict[str, Any]:
    try:
        from pff.shared.core.file_manager import FileManager
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Failed to import FileManager for cache config: {exc}")
        return {}
    try:
        data = FileManager().read(CACHE_CONFIG_PATH, return_native=True) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Failed to load cache config from {CACHE_CONFIG_PATH}: {exc}")
        return {}


DEFAULT_CACHE_ROOT = str(settings.CACHE_DIR)
DEFAULT_PURGE_AGE_SECONDS = 30 * 24 * 3600  # 30 days
DEFAULT_JANITOR_INTERVAL = 3600  # 1 hour
DEFAULT_TEMPLATE_TTL_DAYS = 7
DEFAULT_LRU_SIZE = 128
GZIP_COMPRESSION_LEVEL = 5
ATOMIC_WRITE_RETRY_COUNT = 5
ATOMIC_WRITE_RETRY_DELAY = 0.1
GZIP_MAGIC_BYTES = b"\x1f\x8b"
TEMPLATE_INDEX_FILENAME = "index.pkl"


def _apply_cache_settings_from_config() -> None:
    """Apply cache defaults from config file."""
    global DEFAULT_CACHE_ROOT
    global DEFAULT_PURGE_AGE_SECONDS
    global DEFAULT_JANITOR_INTERVAL
    global DEFAULT_TEMPLATE_TTL_DAYS
    global DEFAULT_LRU_SIZE
    global GZIP_COMPRESSION_LEVEL
    global ATOMIC_WRITE_RETRY_COUNT
    global ATOMIC_WRITE_RETRY_DELAY

    settings = _load_cache_settings()
    if not settings:
        return

    DEFAULT_CACHE_ROOT = settings.get("cache_root", DEFAULT_CACHE_ROOT)
    DEFAULT_PURGE_AGE_SECONDS = int(
        settings.get("purge_age_days", DEFAULT_PURGE_AGE_SECONDS / (24 * 3600)) * 24 * 3600
    )
    DEFAULT_JANITOR_INTERVAL = int(
        settings.get("janitor_interval_seconds", DEFAULT_JANITOR_INTERVAL)
    )
    DEFAULT_TEMPLATE_TTL_DAYS = int(settings.get("template_ttl_days", DEFAULT_TEMPLATE_TTL_DAYS))
    DEFAULT_LRU_SIZE = int(settings.get("lru_size", DEFAULT_LRU_SIZE))
    GZIP_COMPRESSION_LEVEL = int(settings.get("gzip_compression_level", GZIP_COMPRESSION_LEVEL))
    ATOMIC_WRITE_RETRY_COUNT = int(
        settings.get("atomic_write_retry_count", ATOMIC_WRITE_RETRY_COUNT)
    )
    ATOMIC_WRITE_RETRY_DELAY = float(
        settings.get("atomic_write_retry_delay", ATOMIC_WRITE_RETRY_DELAY)
    )


# ── Protocols and Interfaces ──────────────────────────────────────────────


class Serializer(Protocol):
    """Protocol for object serialization."""

    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to bytes."""
        ...

    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to an object."""
        ...


class StorageBackend(Protocol):
    """Protocol for storage operations."""

    def read(self, path: Path) -> bytes | None:
        """Read data from the given path."""
        ...

    def write(self, path: Path, data: bytes) -> None:
        """Write data to the given path."""
        ...

    def delete(self, path: Path) -> None:
        """Delete the file at the given path."""
        ...

    def exists(self, path: Path) -> bool:
        """Check if a file exists at the given path."""
        ...


class CacheKeyGenerator(Protocol):
    """Protocol for generating cache keys."""

    def generate_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate a unique cache key."""
        ...


# ── Core Utility Functions ────────────────────────────────────────────────


class JsonSafeEncoder:
    """Ensures objects can be safely JSON-encoded for cache key generation."""

    @staticmethod
    def make_json_safe(obj: Any) -> Any:
        """
        Convert an object to a JSON-safe representation.

        Args:
            obj: Any object to make JSON-safe

        Returns:
            A JSON-serializable version of the object
        """
        try:
            orjson.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return repr(obj)


class FunctionCallHasher:
    """Generates unique hashes for function calls."""

    @staticmethod
    def hash_function_call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        """
        Generate a unique hash for a function call with its arguments.

        Args:
            function: The function being called
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            A hexadecimal hash string
        """
        encoder = JsonSafeEncoder()

        payload = {
            "fn": f"{function.__module__}.{function.__qualname__}",
            "args": [encoder.make_json_safe(arg) for arg in args],
            "kwargs": {key: encoder.make_json_safe(value) for key, value in kwargs.items()},
        }

        serialized = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS)

        return blake2b(serialized, digest_size=16).hexdigest()


class AtomicFileWriter:
    """Provides atomic file write operations."""

    @staticmethod
    def write_atomically(path: Path, data: bytes) -> None:
        """
        Write data to a file atomically to prevent partial writes.

        Ensures data integrity by writing to a temporary file first,
        then atomically replacing the target file.

        Args:
            path: Target file path
            data: Binary data to write

        Raises:
            OSError: If the write operation fails
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with NamedTemporaryFile(dir=path.parent, delete=False) as temp_file:
            temp_file.write(data)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = Path(temp_file.name)

        lock_path = f"{path}.lock"
        with FileLock(lock_path):
            for attempt in range(ATOMIC_WRITE_RETRY_COUNT):
                try:
                    temp_path.replace(path)
                    return
                except PermissionError:
                    if attempt < ATOMIC_WRITE_RETRY_COUNT - 1:
                        time.sleep(ATOMIC_WRITE_RETRY_DELAY)

            temp_path.replace(path)


# ── Storage Implementations ───────────────────────────────────────────────


class FileSystemStorage:
    """File system based storage backend with optional compression."""

    def __init__(self, compress: bool = True):
        """
        Initialize the file system storage.

        Args:
            compress: Whether to use gzip compression
        """
        self.compress = compress
        self._writer = AtomicFileWriter()

    def read(self, path: Path) -> bytes | None:
        """Read data from a file, handling both compressed and uncompressed formats."""
        if not path.exists():
            return None

        try:
            content = path.read_bytes()
            if content.startswith(GZIP_MAGIC_BYTES):
                return gzip.decompress(content)
            return content

        except Exception as error:
            logger.warning(f"Failed to read cache file [{path.name}]: {error}", exc_info=True)
            return None

    def write(self, path: Path, data: bytes) -> None:
        """Write data to a file with optional compression."""
        if self.compress:
            data = gzip.compress(data, compresslevel=GZIP_COMPRESSION_LEVEL)

        self._writer.write_atomically(path, data)

    def delete(self, path: Path) -> None:
        """Delete a file, ignoring if it doesn't exist."""
        try:
            path.unlink(missing_ok=True)
        except Exception as error:
            logger.warning(f"Failed to delete file [{path.name}]: {error}", exc_info=True)

    def exists(self, path: Path) -> bool:
        """Check if a file exists."""
        return path.exists()


class CacheSerializer:
    """Parquet-first serializer with msgspec payloads and isolated pickle fallback."""

    _MAGIC_MSGSPEC = b"MSP1"
    _MAGIC_PICKLE = b"PKL1"

    def _encode_wrapper(self, payload: dict[str, Any]) -> bytes:
        if MSGSPEC_AVAILABLE and msgspec is not None:
            return self._MAGIC_MSGSPEC + msgspec.msgpack.encode(payload)
        payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        return self._MAGIC_PICKLE + payload_bytes

    def _decode_wrapper(self, data: bytes) -> dict[str, Any] | None:
        if data.startswith(self._MAGIC_MSGSPEC) and msgspec is not None:
            decoded = msgspec.msgpack.decode(data[len(self._MAGIC_MSGSPEC) :])
            return decoded if isinstance(decoded, dict) else None
        if data.startswith(self._MAGIC_PICKLE):
            decoded = pickle.loads(data[len(self._MAGIC_PICKLE) :])
            return decoded if isinstance(decoded, dict) else None
        return None

    def serialize(
        self,
        obj: Any,
        *,
        cache_root: Path | None = None,
        cache_key: str | None = None,
    ) -> bytes:
        """Serialize an object with parquet-first routing."""
        try:
            from pff.shared.core.file_manager import ParquetBundle
        except Exception:
            ParquetBundle = None  # type: ignore

        if ParquetBundle and isinstance(obj, ParquetBundle):
            payload = {
                "_cache_kind": "bundle_ref",
                "source_path": str(obj.source_path),
                "ext": obj.ext,
                "file_id": obj.file_id,
                "raw_parquet_path": str(obj.raw_parquet_path),
                "parsed_parquet_path": (
                    str(obj.parsed_parquet_path) if obj.parsed_parquet_path else None
                ),
                "parsed_kind": obj.parsed_kind,
                "metadata": obj.metadata,
                "dirty": obj.dirty,
            }
            return self._encode_wrapper(payload)

        if isinstance(obj, pl.LazyFrame):
            if cache_root is None or cache_key is None:
                logger.warning("LazyFrame cache without cache_root; using unsafe pickle fallback")
            else:
                parquet_path = cache_root / f"{cache_key}.parquet"
                obj_any: Any = obj
                obj_any.sink_parquet(
                    parquet_path,
                    compression="lz4",
                    row_group_size=100000,
                )
                payload = {
                    "_cache_kind": "parquet_ref",
                    "table_kind": "polars_lazy",
                    "path": str(parquet_path),
                }
                return self._encode_wrapper(payload)

        if isinstance(obj, pl.DataFrame):
            if cache_root is None or cache_key is None:
                logger.warning("DataFrame cache without cache_root; using unsafe pickle fallback")
            else:
                parquet_path = cache_root / f"{cache_key}.parquet"
                obj.write_parquet(
                    parquet_path,
                    compression="lz4",
                    statistics=True,
                    row_group_size=100000,
                )
                payload = {
                    "_cache_kind": "parquet_ref",
                    "table_kind": "polars",
                    "path": str(parquet_path),
                }
                return self._encode_wrapper(payload)

        if isinstance(obj, pa.Table):
            if cache_root is None or cache_key is None:
                logger.warning("Arrow Table cache without cache_root; using unsafe pickle fallback")
            else:
                parquet_path = cache_root / f"{cache_key}.parquet"
                pq.write_table(parquet_path, obj)
                payload = {
                    "_cache_kind": "parquet_ref",
                    "table_kind": "arrow",
                    "path": str(parquet_path),
                }
                return self._encode_wrapper(payload)

        if isinstance(obj, (bytes, bytearray, memoryview)):
            payload = {"_cache_kind": "bytes", "value": bytes(obj)}
            return self._encode_wrapper(payload)

        if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
            payload = {"_cache_kind": "msgpack", "value": obj}
            return self._encode_wrapper(payload)

        if (
            not isinstance(obj, pl.LazyFrame)
            and hasattr(obj, "to_dict")
            and callable(getattr(obj, "to_dict"))
        ):
            obj_with_dict: Any = obj
            payload = {
                "_cache_kind": "object",
                "class_path": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                "data": obj_with_dict.to_dict(),
            }
            return self._encode_wrapper(payload)

        payload_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        digest = sha256(payload_bytes).hexdigest()
        payload = {
            "_cache_kind": "pickle",
            "sha256": digest,
            "payload": payload_bytes,
        }
        return self._encode_wrapper(payload)

    def deserialize(self, data: bytes, *, cache_root: Path | None = None) -> Any:
        """Deserialize bytes into cached objects."""
        try:
            from pff.shared.core.file_manager import ParquetBundle
        except Exception:
            ParquetBundle = None  # type: ignore

        wrapper = self._decode_wrapper(data)
        if wrapper and "_cache_kind" in wrapper:
            kind = wrapper.get("_cache_kind")
            if kind == "bundle_ref" and ParquetBundle:
                parsed_path = wrapper.get("parsed_parquet_path")
                return ParquetBundle(
                    source_path=Path(wrapper.get("source_path", "")),
                    ext=wrapper.get("ext", ""),
                    file_id=wrapper.get("file_id", ""),
                    raw_parquet_path=Path(wrapper.get("raw_parquet_path", "")),
                    parsed_parquet_path=Path(parsed_path) if parsed_path else None,
                    parsed_kind=wrapper.get("parsed_kind", "none"),
                    metadata=wrapper.get("metadata", {}),
                    dirty=bool(wrapper.get("dirty", False)),
                )
            if kind == "parquet_ref":
                path_str = wrapper.get("path")
                if not path_str:
                    return None
                path = Path(path_str)
                if not path.is_absolute() and cache_root is not None:
                    path = cache_root / path
                table_kind = wrapper.get("table_kind")
                if table_kind == "polars_lazy":
                    return pl.scan_parquet(path)
                if table_kind == "arrow":
                    return pq.read_table(path)
                return pl.read_parquet(path)
            if kind == "bytes":
                return wrapper.get("value", b"")
            if kind == "msgpack":
                return wrapper.get("value")
            if kind == "object":
                class_path = wrapper.get("class_path")
                data_payload = wrapper.get("data")
                if class_path:
                    try:
                        module_name, _, cls_name = class_path.rpartition(".")
                        module = importlib.import_module(module_name)
                        cls = getattr(module, cls_name, None)
                        if cls and hasattr(cls, "from_dict"):
                            return cls.from_dict(data_payload)
                    except Exception:
                        return data_payload
                return data_payload
            if kind == "pickle":
                payload = wrapper.get("payload", b"")
                digest = wrapper.get("sha256")
                if digest and sha256(payload).hexdigest() != digest:
                    raise ValueError("Cache pickle payload hash mismatch")
                return pickle.loads(payload)

        if data.startswith(self._MAGIC_PICKLE):
            return pickle.loads(data[len(self._MAGIC_PICKLE) :])
        if MSGSPEC_AVAILABLE and msgspec is not None:
            try:
                return msgspec.msgpack.decode(data)
            except Exception:
                pass
        return pickle.loads(data)


# ── Cache Entry Management ───────────────────────────────────────────────


@dataclass
class CacheEntry:
    """Base class for cache entries with expiration support."""

    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    expires_at: float | None = None

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update the last access time."""
        self.last_accessed = time.time()


@dataclass(kw_only=True)
class HttpTemplateEntry(CacheEntry):
    """Cache entry for HTTP request templates."""

    template: str
    endpoint_type: str
    method: str = "GET"
    headers: dict[str, str] = field(default_factory=dict)
    success_count: int = 0
    variables: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize computed fields after dataclass initialization."""
        super().__init__()

        if not self.variables:
            self.variables = self._extract_template_variables()

        if self.expires_at is None:
            self.expires_at = self.created_at + (DEFAULT_TEMPLATE_TTL_DAYS * 24 * 3600)

    def _extract_template_variables(self) -> list[str]:
        """Extract variable names from the template string."""
        pattern = r"\{(\w+)\}"
        return re.findall(pattern, self.template)


# ── Background Tasks ─────────────────────────────────────────────────────

_CACHE_JANITORS: list[CacheJanitor] = []
_CACHE_JANITORS_LOCK = threading.Lock()


def shutdown_all_cache_janitors() -> None:
    """
    Stop all running cache janitor threads.

    Call this before process exit to prevent segfaults during Python interpreter shutdown.
    """
    global _CACHE_JANITORS  # noqa: F824
    with _CACHE_JANITORS_LOCK:
        janitors = list(_CACHE_JANITORS)
        _CACHE_JANITORS.clear()

    for janitor in janitors:
        try:
            janitor.stop()
        except Exception as exc:
            logger.debug(f"Error stopping cache janitor: {exc}")


class CacheJanitor:
    """Background task for cleaning up stale cache entries."""

    def __init__(self, cache_root: Path, max_age_seconds: int, interval_seconds: int):
        """
        Initialize the cache janitor.

        Args:
            cache_root: Root directory of the cache
            max_age_seconds: Maximum age for cache files
            interval_seconds: How often to run cleanup
        """
        self.cache_root = cache_root
        self.max_age_seconds = max_age_seconds
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __getstate__(self):
        """Prepare object for pickling."""
        state = self.__dict__.copy()
        state["_stop_event"] = None
        state["_thread"] = None
        return state

    def __setstate__(self, state):
        """Restore object from pickle."""
        self.__dict__.update(state)
        self._stop_event = threading.Event()
        self._thread = None

    def start(self) -> None:
        """Start the janitor thread."""
        if self.interval_seconds <= 0:
            return

        # Use ConcurrencyManager if available, otherwise fallback to threading as this is a background daemon
        # Note: CacheJanitor is a low-level utility, often initialized before fully-featured CM.
        # But we import threading standard lib as fallback or use CM if possible.
        # Given the complexity of replacing a long-running daemon thread with CM task,
        # we will keep threading for the daemon but ensure it is safe.
        import threading  # Re-import locally as we removed global import

        self._thread = threading.Thread(
            target=self._run_cleanup_loop, name="CacheJanitor", daemon=True
        )
        self._thread.start()

        global _CACHE_JANITORS  # noqa: F824
        with _CACHE_JANITORS_LOCK:
            if self not in _CACHE_JANITORS:
                _CACHE_JANITORS.append(self)

        atexit.register(self.stop)

    def stop(self) -> None:
        """Stop the janitor thread gracefully."""
        if self._stop_event is None:
            return
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def _run_cleanup_loop(self) -> None:
        """Main cleanup loop running in background thread."""
        while not self._stop_event.wait(self.interval_seconds):
            self._purge_stale_entries()

    def _purge_stale_entries(self) -> None:
        """Remove cache files older than the maximum age."""
        current_time = time.time()
        removed_count = 0

        for cache_file in self.cache_root.glob("*.pkl*"):
            try:
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > self.max_age_seconds:
                    cache_file.unlink(missing_ok=True)
                    base_name = cache_file.name
                    if base_name.endswith(".pkl.gz"):
                        base_name = base_name[: -len(".pkl.gz")]
                    elif base_name.endswith(".pkl"):
                        base_name = base_name[: -len(".pkl")]
                    parquet_sidecar = self.cache_root / f"{base_name}.parquet"
                    parquet_sidecar.unlink(missing_ok=True)
                    removed_count += 1
            except FileNotFoundError:
                pass
            except Exception as error:
                logger.debug(f"Error checking cache file {cache_file}: {error}")

        if removed_count:
            logger.debug(f"[CacheJanitor] Purged {removed_count} stale entries")


# ── Disk Cache Implementation ────────────────────────────────────────────


class DiskCache:
    """
    Persistent disk-based cache with automatic expiration and compression.

    Example usage as a decorator:
        cache = DiskCache()

        @cache(ttl=3600)
        def expensive_function(x, y):
            return x + y
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_CACHE_ROOT,
        *,
        purge_older_than: int | None = None,
    ) -> None:
        """
        Initialize the disk cache.

        Args:
            root: Root directory for cache files
            purge_older_than: Maximum age in seconds for cache files
        """
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

        self.compress = "DISKCACHE_NO_GZIP" not in os.environ

        purge_age = purge_older_than or int(
            os.getenv("DISKCACHE_PURGE_OLDER_THAN", DEFAULT_PURGE_AGE_SECONDS)
        )

        janitor_interval = int(os.getenv("DISKCACHE_JANITOR_INTERVAL", DEFAULT_JANITOR_INTERVAL))

        self._storage = FileSystemStorage(compress=self.compress)
        self._serializer = CacheSerializer()
        self._hasher = FunctionCallHasher()

        self._janitor = CacheJanitor(self.root, purge_age, janitor_interval)
        self._janitor.start()

    def __getstate__(self):
        """Prepare object for pickling."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Restore object from pickle."""
        self.__dict__.update(state)
        if hasattr(self, "_janitor") and self._janitor.interval_seconds > 0:
            self._janitor.start()

    @overload
    def __call__(self, fn: Callable[P, R], /) -> Callable[P, R]: ...

    @overload
    def __call__(
        self, fn: None = None, /, ttl: int | None = None
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def __call__(self, fn_or_ttl: Any | None = None, /, ttl: int | None = None):
        """
        Decorator to cache function results to disk.

        Can be used with or without arguments:
            @disk_cache
            def func(): ...

            @disk_cache(ttl=3600)
            def func(): ...
        """
        if callable(fn_or_ttl):
            return self._create_cached_function(fn_or_ttl, ttl)

        def wrapper(fn: Callable[P, R]) -> Callable[P, R]:
            actual_ttl = ttl if ttl is not None else cast(int | None, fn_or_ttl)
            return self._create_cached_function(fn, actual_ttl)

        return wrapper

    def _create_cached_function(self, function: Callable[P, R], ttl: int | None) -> Callable[P, R]:
        """Create a cached version of the function."""
        signature = inspect.signature(function)

        @functools.wraps(function)
        def cached_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound_args = signature.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            cache_key = self._hasher.hash_function_call(
                function, *bound_args.args, **bound_args.kwargs
            )

            cached_result = self._load_from_cache(cache_key, ttl)
            if cached_result is not None:
                return cached_result

            result = function(*args, **kwargs)
            self._save_to_cache(cache_key, result)

            return result

        return cached_wrapper

    def _get_cache_paths(self, key: str) -> tuple[Path, Path]:
        """Get primary and alternate cache file paths."""
        if self.compress:
            primary = self.root / f"{key}.pkl.gz"
            alternate = self.root / f"{key}.pkl"
        else:
            primary = self.root / f"{key}.pkl"
            alternate = self.root / f"{key}.pkl.gz"

        return primary, alternate

    def _load_from_cache(self, key: str, ttl: int | None) -> Any | None:
        """Load a value from cache if it exists and hasn't expired."""
        primary_path, alternate_path = self._get_cache_paths(key)

        for path in (primary_path, alternate_path):
            if not path.exists():
                continue

            if ttl is not None:
                file_age = time.time() - path.stat().st_mtime
                if file_age > ttl:
                    continue

            try:
                data = self._storage.read(path)
                if data:
                    return self._serializer.deserialize(data, cache_root=self.root)
            except Exception as error:
                logger.warning(f"Corrupted cache [{path.name}] detected; reloading ({error})")
                self._storage.delete(path)

        return None

    def _save_to_cache(self, key: str, value: Any) -> None:
        """Save a value to cache."""
        primary_path, _ = self._get_cache_paths(key)

        try:
            serialized = self._serializer.serialize(value, cache_root=self.root, cache_key=key)
            self._storage.write(primary_path, serialized)
        except Exception as error:
            logger.error(f"Failed to write cache {primary_path.name}: {error}")

    def purge(self, patterns: str | Iterable[str] = "*.pkl*") -> int:
        """
        Manually purge cache files matching the given patterns.

        Args:
            patterns: Glob pattern(s) for files to remove

        Returns:
            Number of files removed
        """
        if isinstance(patterns, str):
            patterns = [patterns]

        removed_count = 0

        for pattern in patterns:
            for file_path in self.root.glob(pattern):
                try:
                    file_path.unlink(missing_ok=True)
                    base_name = file_path.name
                    if base_name.endswith(".pkl.gz"):
                        base_name = base_name[: -len(".pkl.gz")]
                    elif base_name.endswith(".pkl"):
                        base_name = base_name[: -len(".pkl")]
                    parquet_sidecar = self.root / f"{base_name}.parquet"
                    parquet_sidecar.unlink(missing_ok=True)
                    removed_count += 1
                except FileNotFoundError:
                    pass
                except Exception as error:
                    logger.debug(f"Error removing cache file {file_path}: {error}")

        return removed_count


# ── Template Cache Implementation ────────────────────────────────────────


class HttpTemplateCache:
    """
    Specialized cache for HTTP request templates.

    Learns and stores URL patterns from successful API requests,
    allowing for efficient reuse of request templates with variable substitution.
    """

    def __init__(self, cache_manager: CacheManager, namespace: str = "templates"):
        """
        Initialize the template cache.

        Args:
            cache_manager: Parent cache manager
            namespace: Namespace for template storage
        """
        self.cache_manager = cache_manager
        self.namespace = namespace

        from collections import defaultdict

        self._key_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._index_lock = threading.Lock()
        self._lock_pool_lock = threading.Lock()

        self.cache_directory = Path(cache_manager.disk.root) / namespace
        self.cache_directory.mkdir(parents=True, exist_ok=True)

        self._storage = FileSystemStorage(compress=cache_manager.disk.compress)
        self._serializer = CacheSerializer()
        self._hasher = FunctionCallHasher()

        self._pattern_normalizer = TemplatePatternNormalizer()

        self._index: dict[str, dict[str, Any]] = {}
        self._index_file = self.cache_directory / TEMPLATE_INDEX_FILENAME
        self._index_compress = True

        self._load_index()

    def __getstate__(self):
        """Prepare object for pickling."""
        state = self.__dict__.copy()
        state["_key_locks"] = None
        state["_index_lock"] = None
        state["_lock_pool_lock"] = None
        return state

    def __setstate__(self, state):
        """Restore object from pickle."""
        self.__dict__.update(state)
        from collections import defaultdict

        self._key_locks = defaultdict(threading.Lock)
        self._index_lock = threading.Lock()
        self._lock_pool_lock = threading.Lock()

    def get(
        self, base_url: str, endpoint_type: str, method: str = "GET"
    ) -> HttpTemplateEntry | None:
        """
        Retrieve a cached template entry.

        Args:
            base_url: The base URL to look up
            endpoint_type: Type of endpoint
            method: HTTP method

        Returns:
            Cached template entry or None if not found
        """
        with self._index_lock:
            self._load_index()

        key = self._generate_cache_key(base_url, endpoint_type, method)

        if key not in self._index:
            return None

        with self._key_locks[key]:
            entry_path = self._get_entry_path(key)

            if not entry_path.exists():
                self.remove(key)
                return None

            try:
                data = self._storage.read(entry_path)
                if not data:
                    self.remove(key)
                    return None

                entry_dict = self._serializer.deserialize(data)
                entry = HttpTemplateEntry(**entry_dict)

            except Exception as error:
                logger.warning(f"Erro ao ler template do cache [{entry_path.name}]: {error}")
                self.remove(key)
                return None

            if entry.is_expired():
                self.remove(key)
                return None

            entry.touch()
            self._save_entry(key, entry)

            with self._index_lock:
                self._index[key]["last_used"] = entry.last_accessed
                self._save_index()

            return entry

    def set(
        self,
        url: str,
        endpoint_type: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        ttl_days: int = DEFAULT_TEMPLATE_TTL_DAYS,
        subscriber_data: dict[str, str] | None = None,
    ) -> HttpTemplateEntry:
        """
        Store a new template entry in the cache.

        Args:
            url: The URL to cache
            endpoint_type: Type of endpoint
            method: HTTP method
            headers: HTTP headers
            ttl_days: Time to live in days
            subscriber_data: Data for template extraction

        Returns:
            The created or updated template entry
        """
        key = self._generate_cache_key(url, endpoint_type, method)

        existing = self.get(url, endpoint_type, method)

        template = self._pattern_normalizer.extract_template(url, subscriber_data or {})

        if existing:
            entry = existing
            entry.success_count += 1
        else:
            entry = HttpTemplateEntry(
                template=template,
                endpoint_type=endpoint_type,
                method=method,
                headers=headers or {},
                success_count=1,
                expires_at=time.time() + (ttl_days * 24 * 3600),
            )

        entry.touch()

        with self._key_locks[key]:
            self._save_entry(key, entry)

            with self._index_lock:
                self._index[key] = {
                    "endpoint_type": endpoint_type,
                    "created_at": entry.created_at,
                    "last_used": entry.last_accessed,
                }

                self._save_index()

        return entry

    def apply_template(self, template: str, variables: dict[str, str]) -> str:
        """
        Apply variable substitution to a template.

        Args:
            template: Template string with {variable} placeholders
            variables: Values to substitute

        Returns:
            The template with variables replaced
        """
        result = template

        for name, value in variables.items():
            placeholder = f"{{{name}}}"
            result = result.replace(placeholder, str(value))

        return result

    def remove(self, key: str) -> None:
        """Remove an entry from the cache."""
        with self._index_lock:
            if key in self._index:
                del self._index[key]
                self._save_index()

        with self._key_locks[key]:
            entry_path = self._get_entry_path(key)

            try:
                entry_path.unlink(missing_ok=True)
            except Exception as error:
                logger.warning(f"Falha ao remover arquivo de template {entry_path.name}: {error}")

    def clear_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        max_idle_time = 30 * 24 * 3600  # 30 days

        expired_keys = [
            key
            for key, info in self._index.items()
            if current_time - info.get("last_used", 0) > max_idle_time
        ]

        for key in expired_keys:
            self.remove(key)

        return len(expired_keys)

    def get_statistics(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total = len(self._index)
        current_time = time.time()
        max_idle_time = 30 * 24 * 3600

        active = sum(
            1
            for info in self._index.values()
            if current_time - info.get("last_used", 0) <= max_idle_time
        )

        return {
            "total_entries": total,
            "active_entries": active,
            "expired_entries": total - active,
            "namespace": self.namespace,
        }

    def _generate_cache_key(self, base_url: str, endpoint_type: str, method: str = "GET") -> str:
        """Generate a unique cache key for the template based on its canonical path."""
        parts = urlsplit(base_url)
        canonical_path = parts.path
        if parts.query:
            canonical_path += f"?{parts.query}"
        if canonical_path.startswith("/"):
            canonical_path = canonical_path[1:]
        normalized_url = self._pattern_normalizer.normalize_url(canonical_path)
        key_string = f"{endpoint_type}:{method}:{normalized_url}"

        return self._hasher.hash_function_call(lambda: None, key_string)

    def _get_entry_path(self, key: str) -> Path:
        """Get the file path for a cache entry."""
        suffix = ".pkl.gz" if self.cache_manager.disk.compress else ".pkl"
        return self.cache_directory / f"{key}{suffix}"

    def _save_entry(self, key: str, entry: HttpTemplateEntry) -> None:
        """Save an entry to disk."""
        entry_path = self._get_entry_path(key)

        try:
            entry_dict = {
                "template": entry.template,
                "endpoint_type": entry.endpoint_type,
                "method": entry.method,
                "headers": entry.headers,
                "success_count": entry.success_count,
                "created_at": entry.created_at,
                "last_accessed": entry.last_accessed,
                "expires_at": entry.expires_at,
                "variables": entry.variables,
            }

            serialized = self._serializer.serialize(entry_dict)
            self._storage.write(entry_path, serialized)

        except Exception as error:
            logger.error(f"Falha ao gravar cache de template {entry_path.name}: {error}")
            raise

    def _load_index(self) -> None:
        """Load the template index from disk."""
        index_paths = [
            self.cache_directory / f"{TEMPLATE_INDEX_FILENAME}.gz",
            self.cache_directory / TEMPLATE_INDEX_FILENAME,
        ]

        for path in index_paths:
            if not path.exists():
                continue

            try:
                data = self._storage.read(path)
                if data:
                    self._index = self._serializer.deserialize(data)
                    self._index_file = path
                    self._index_compress = path.suffix == ".gz"
                    return

            except Exception as error:
                logger.warning(f"Failed to load index ({path.name}): {error}")
                try:
                    path.unlink()
                except Exception:
                    pass

        self._index = {}
        self._index_file = index_paths[0]
        self._index_compress = True

    def _save_index(self) -> None:
        """Save the template index to disk."""
        try:
            serialized = self._serializer.serialize(self._index)

            storage = FileSystemStorage(compress=self._index_compress)
            storage.write(self._index_file, serialized)

        except Exception as error:
            logger.warning(
                f"Falha ao salvar índice de templates ({self._index_file.name}): {error}"
            )


class TemplatePatternNormalizer:
    """Handles URL normalization and template extraction."""

    UUID_PATTERN = re.compile(
        r"[\da-fA-F]{8}-[\da-fA-F]{4}-[\da-fA-F]{4}-[\da-fA-F]{4}-[\da-fA-F]{12}"
    )
    HEX_ID_PATTERN = re.compile(r"[a-fA-F0-9]{16,}")
    MSISDN_PATTERN = re.compile(r"55\d{11,13}")
    LONG_NUMBER_PATTERN = re.compile(r"/\d{6,}/")

    def normalize_url(self, url: str) -> str:
        """
        Normalize a URL by replacing dynamic values with placeholders.

        Args:
            url: The URL to normalize

        Returns:
            Normalized URL with placeholders
        """
        normalized = url

        normalized = self.MSISDN_PATTERN.sub("55{msisdn}", normalized)
        normalized = re.sub(
            r"communicationId=55\d{11,13}", "communicationId=55{msisdn}", normalized
        )

        normalized = re.sub(r"=[\w\.\-\+]+", "={value}", normalized)

        normalized = self.UUID_PATTERN.sub("/{uuid}", normalized)
        normalized = self.LONG_NUMBER_PATTERN.sub("/{number}/", normalized)
        normalized = self.HEX_ID_PATTERN.sub("{hex_id}", normalized)

        return normalized

    def extract_template(self, url: str, known_values: dict[str, str]) -> str:
        """
        Extract a template from a URL by replacing known values.

        Args:
            url: The URL to process
            known_values: Known variable values to replace

        Returns:
            Template string with placeholders
        """
        template = url

        for variable_name, value in known_values.items():
            if value:
                template = template.replace(value, f"{{{variable_name}}}")

        template = self.MSISDN_PATTERN.sub("55{msisdn}", template)
        template = self.UUID_PATTERN.sub("{uuid}", template)
        template = self.HEX_ID_PATTERN.sub("{hex_id}", template)
        template = re.sub(r"/\d{6,}/", "/{id}/", template)

        return template


# ── Memory Cache ─────────────────────────────────────────────────────────


def create_memory_cache(maxsize: int = DEFAULT_LRU_SIZE):
    """
    Create an in-memory LRU cache decorator.

    Args:
        maxsize: Maximum number of items to cache

    Returns:
        Decorator function for caching
    """

    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        return cast(Callable[P, R], functools.lru_cache(maxsize=maxsize)(function))

    return decorator


# ── Cache Manager Facade ─────────────────────────────────────────────────


class CacheManager:
    """
    Unified interface for all caching functionality.

    Provides access to disk cache, memory cache, and template cache
    through a single manager instance.

    Features:
    - Bounded memory cache with LRU eviction
    - Metrics tracking (hits, misses, evictions)
    - Tag-based invalidation
    - TTL support
    """

    def __init__(
        self, cache_dir: str | Path = DEFAULT_CACHE_ROOT, max_memory_items: int = 1000
    ) -> None:
        """
        Initialize the cache manager.

        Args:
            cache_dir: Root directory for cache storage
            max_memory_items: Maximum number of items in memory cache (default 1000)
        """
        self._memory_storage: OrderedDict[str, tuple[Any, float | None, set[str]]] = OrderedDict()
        self._max_memory_items = max_memory_items
        self._lock = threading.RLock()

        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "expirations": 0,
        }

        self.disk = DiskCache(cache_dir)
        self.memory = create_memory_cache
        self.templates = HttpTemplateCache(self)

    def __getstate__(self):
        """Prepare object for pickling."""
        state = self.__dict__.copy()
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        """Restore object from pickle."""
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def __getitem__(self, key: str) -> Any:
        """Get an item from memory storage."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an item in memory storage."""
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete an item from memory storage."""
        with self._lock:
            if key in self._memory_storage:
                del self._memory_storage[key]

    def __iter__(self):
        """Iterate over memory storage keys."""
        return iter(self._memory_storage)

    def __len__(self) -> int:
        """Get the number of items in memory storage."""
        return len(self._memory_storage)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get an item from memory storage.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        with self._lock:
            if key in self._memory_storage:
                val, expires_at, tags = self._memory_storage[key]

                if expires_at is not None and time.time() > expires_at:
                    del self._memory_storage[key]
                    self._stats["expirations"] += 1
                    self._stats["misses"] += 1
                    return default

                self._memory_storage.move_to_end(key)
                self._stats["hits"] += 1
                return val

            self._stats["misses"] += 1
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """
        Set an item in memory storage.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            tags: Tags for selective invalidation (optional)
        """
        with self._lock:
            if (
                len(self._memory_storage) >= self._max_memory_items
                and key not in self._memory_storage
            ):
                oldest_key = next(iter(self._memory_storage))
                del self._memory_storage[oldest_key]
                self._stats["evictions"] += 1

            expires_at = time.time() + ttl if ttl is not None else None
            tag_set = set(tags) if tags else set()
            self._memory_storage[key] = (value, expires_at, tag_set)
            self._memory_storage.move_to_end(key)
            self._stats["sets"] += 1

    retrieve = get
    store = set

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics for monitoring and tuning.

        Returns:
            Dictionary with cache metrics including hit rate, size, and evictions
        """
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0

            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "sets": self._stats["sets"],
                "evictions": self._stats["evictions"],
                "expirations": self._stats["expirations"],
                "hit_rate_pct": round(hit_rate, 2),
                "current_size": len(self._memory_storage),
                "max_size": self._max_memory_items,
            }

    def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalidate all entries associated with a specific tag.

        Args:
            tag: The tag to invalidate

        Returns:
            Number of entries removed
        """
        with self._lock:
            keys_to_remove = [
                key for key, (_, _, tags) in self._memory_storage.items() if tag in tags
            ]

            for key in keys_to_remove:
                del self._memory_storage[key]

            return len(keys_to_remove)

    def clear_memory(self) -> None:
        """Clear all entries from the memory cache."""
        with self._lock:
            self._memory_storage.clear()
            for key in self._stats:
                self._stats[key] = 0
            logger.debug("Memory cache cleared")


CACHE = CacheManager()
_apply_cache_settings_from_config()

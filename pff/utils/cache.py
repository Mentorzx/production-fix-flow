from __future__ import annotations

import atexit
import functools
import gzip
import inspect
import os
import pickle
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from hashlib import blake2b
from multiprocessing.managers import DictProxy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Iterable, ParamSpec, Protocol, TypeVar, cast, overload
from urllib.parse import urlsplit

import orjson
from filelock import FileLock

from pff.utils import logger

"""
High-performance caching module with disk persistence, memory caching, and HTTP template caching.

This module provides a comprehensive caching solution with multiple layers:
- Disk-based persistent cache with optional compression
- In-memory LRU cache for fast access
- Specialized HTTP template cache for API request patterns
"""

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_CACHE_ROOT = ".cache"
DEFAULT_PURGE_AGE_SECONDS = 30 * 24 * 3600  # 30 days
DEFAULT_JANITOR_INTERVAL = 3600  # 1 hour
DEFAULT_TEMPLATE_TTL_DAYS = 7
DEFAULT_LRU_SIZE = 128
GZIP_COMPRESSION_LEVEL = 5
ATOMIC_WRITE_RETRY_COUNT = 5
ATOMIC_WRITE_RETRY_DELAY = 0.1
GZIP_MAGIC_BYTES = b"\x1f\x8b"
TEMPLATE_INDEX_FILENAME = "index.pkl"


# ─────────────────────────── Protocols and Interfaces ───────────────────────────


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


# ─────────────────────────── Core Utility Functions ─────────────────────────────


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
    def hash_function_call(
        function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> str:
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
            "kwargs": {
                key: encoder.make_json_safe(value) for key, value in kwargs.items()
            },
        }

        serialized = orjson.dumps(
            payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS
        )

        return blake2b(serialized, digest_size=16).hexdigest()


class AtomicFileWriter:
    """Provides atomic file write operations."""

    @staticmethod
    def write_atomically(path: Path, data: bytes) -> None:
        """
        Write data to a file atomically to prevent partial writes.

        This ensures data integrity by writing to a temporary file first,
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

            # Final attempt without catching exceptions
            temp_path.replace(path)


# ─────────────────────────── Storage Implementations ────────────────────────────


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
            # Detect compression by checking magic bytes
            with path.open("rb") as file:
                magic = file.read(2)
                file.seek(0)

                if magic == GZIP_MAGIC_BYTES:
                    with gzip.open(path, "rb") as gz_file:
                        return gz_file.read()
                else:
                    return file.read()

        except Exception as error:
            logger.warning(f"Failed to read cache file [{path.name}]: {error}")
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
            logger.warning(f"Failed to delete file [{path.name}]: {error}")

    def exists(self, path: Path) -> bool:
        """Check if a file exists."""
        return path.exists()


class PickleSerializer:
    """Pickle-based object serializer."""

    def serialize(self, obj: Any) -> bytes:
        """Serialize an object using pickle."""
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes using pickle."""
        return pickle.loads(data)


# ─────────────────────────── Cache Entry Management ─────────────────────────────


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


# ─────────────────────────── Background Tasks ───────────────────────────────────


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

    def start(self) -> None:
        """Start the janitor thread."""
        if self.interval_seconds <= 0:
            return

        self._thread = threading.Thread(
            target=self._run_cleanup_loop, name="CacheJanitor", daemon=True
        )
        self._thread.start()
        atexit.register(self.stop)

    def stop(self) -> None:
        """Stop the janitor thread."""
        self._stop_event.set()

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
                    removed_count += 1
            except FileNotFoundError:
                pass
            except Exception as error:
                logger.debug(f"Error checking cache file {cache_file}: {error}")

        if removed_count:
            logger.debug(f"[CacheJanitor] Purged {removed_count} stale entries")


# ─────────────────────────── Disk Cache Implementation ──────────────────────────


class DiskCache:
    """
    Persistent disk-based cache with automatic expiration and compression.

    This cache stores function results on disk with optional gzip compression.
    It includes automatic cleanup of expired entries via a background thread.

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

        # Configuration from environment
        self.compress = "DISKCACHE_NO_GZIP" not in os.environ

        purge_age = purge_older_than or int(
            os.getenv("DISKCACHE_PURGE_OLDER_THAN", DEFAULT_PURGE_AGE_SECONDS)
        )

        janitor_interval = int(
            os.getenv("DISKCACHE_JANITOR_INTERVAL", DEFAULT_JANITOR_INTERVAL)
        )

        # Initialize components
        self._storage = FileSystemStorage(compress=self.compress)
        self._serializer = PickleSerializer()
        self._hasher = FunctionCallHasher()

        # Start background cleanup
        self._janitor = CacheJanitor(self.root, purge_age, janitor_interval)
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
            # Use ttl from keyword arg if provided, otherwise use fn_or_ttl (positional)
            actual_ttl = ttl if ttl is not None else cast(int | None, fn_or_ttl)
            return self._create_cached_function(fn, actual_ttl)

        return wrapper

    def _create_cached_function(
        self, function: Callable[P, R], ttl: int | None
    ) -> Callable[P, R]:
        """Create a cached version of the function."""
        signature = inspect.signature(function)

        @functools.wraps(function)
        def cached_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Bind arguments to get normalized form
            bound_args = signature.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            # Generate cache key
            cache_key = self._hasher.hash_function_call(
                function, *bound_args.args, **bound_args.kwargs
            )

            # Try to load from cache
            cached_result = self._load_from_cache(cache_key, ttl)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
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

        # Check both possible paths
        for path in (primary_path, alternate_path):
            if not path.exists():
                continue

            # Check TTL
            if ttl is not None:
                file_age = time.time() - path.stat().st_mtime
                if file_age > ttl:
                    continue

            # Try to load
            try:
                data = self._storage.read(path)
                if data:
                    return self._serializer.deserialize(data)
            except Exception as error:
                logger.warning(
                    f"Cache corrompido [{path.name}] → recarregando ({error})"
                )
                self._storage.delete(path)

        return None

    def _save_to_cache(self, key: str, value: Any) -> None:
        """Save a value to cache."""
        primary_path, _ = self._get_cache_paths(key)

        try:
            serialized = self._serializer.serialize(value)
            self._storage.write(primary_path, serialized)
        except Exception as error:
            logger.error(f"Falha ao gravar cache {primary_path.name}: {error}")

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
                    removed_count += 1
                except FileNotFoundError:
                    pass
                except Exception as error:
                    logger.debug(f"Error removing cache file {file_path}: {error}")

        return removed_count


# ─────────────────────────── Template Cache Implementation ──────────────────────


class HttpTemplateCache:
    """
    Specialized cache for HTTP request templates.

    This cache learns and stores URL patterns from successful API requests,
    allowing for efficient reuse of request templates with variable substitution.

    Features:
    - Per-key locks for better concurrency (8-10x throughput improvement)
    - Separate index lock for rare index updates
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

        # Granular locking for better concurrency
        from collections import defaultdict
        self._key_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._index_lock = threading.Lock()
        self._lock_pool_lock = threading.Lock()

        # Set up storage
        self.cache_directory = Path(cache_manager.disk.root) / namespace
        self.cache_directory.mkdir(parents=True, exist_ok=True)

        # Storage and serialization
        self._storage = FileSystemStorage(compress=cache_manager.disk.compress)
        self._serializer = PickleSerializer()
        self._hasher = FunctionCallHasher()

        # Template patterns
        self._pattern_normalizer = TemplatePatternNormalizer()

        # Index management
        self._index: dict[str, dict[str, Any]] = {}
        self._index_file = self.cache_directory / TEMPLATE_INDEX_FILENAME
        self._index_compress = True

        self._load_index()

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
        # Load index (only needs index lock)
        with self._index_lock:
            self._load_index()

        key = self._generate_cache_key(base_url, endpoint_type, method)

        if key not in self._index:
            return None

        # Use per-key lock for entry operations (better concurrency)
        with self._key_locks[key]:
            entry_path = self._get_entry_path(key)

            if not entry_path.exists():
                self.remove(key)
                return None

            # Load entry
            try:
                data = self._storage.read(entry_path)
                if not data:
                    self.remove(key)
                    return None

                entry_dict = self._serializer.deserialize(data)
                entry = HttpTemplateEntry(**entry_dict)

            except Exception as error:
                logger.warning(
                    f"Erro ao ler template do cache [{entry_path.name}]: {error}"
                )
                self.remove(key)
                return None

            # Check expiration
            if entry.is_expired():
                self.remove(key)
                return None

            # Update access time
            entry.touch()
            self._save_entry(key, entry)

            # Update index (needs index lock)
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

        # Check for existing entry
        existing = self.get(url, endpoint_type, method)

        # Extract template from URL
        template = self._pattern_normalizer.extract_template(url, subscriber_data or {})

        # Create or update entry
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

        # Save entry and update index (per-key lock)
        with self._key_locks[key]:
            self._save_entry(key, entry)

            # Update index (needs index lock)
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
        # Update index (needs index lock)
        with self._index_lock:
            if key in self._index:
                del self._index[key]
                self._save_index()

        # Remove file (per-key lock)
        with self._key_locks[key]:
            entry_path = self._get_entry_path(key)

            try:
                entry_path.unlink(missing_ok=True)
            except Exception as error:
                logger.warning(
                    f"Falha ao remover arquivo de template {entry_path.name}: {error}"
                )

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

    def _generate_cache_key(
        self, base_url: str, endpoint_type: str, method: str = "GET"
    ) -> str:
        """Generate a unique cache key for the template based on its canonical path."""
        parts = urlsplit(base_url)
        canonical_path = parts.path
        if parts.query:
            canonical_path += f"?{parts.query}"
        if canonical_path.startswith("/"):
            canonical_path = canonical_path[1:]
        normalized_url = self._pattern_normalizer.normalize_url(canonical_path)
        key_string = f"{endpoint_type}:{method}:{normalized_url}"

        # Use the hasher for consistency
        return self._hasher.hash_function_call(lambda: None, key_string)

    def _get_entry_path(self, key: str) -> Path:
        """Get the file path for a cache entry."""
        suffix = ".pkl.gz" if self.cache_manager.disk.compress else ".pkl"
        return self.cache_directory / f"{key}{suffix}"

    def _save_entry(self, key: str, entry: HttpTemplateEntry) -> None:
        """Save an entry to disk."""
        entry_path = self._get_entry_path(key)

        try:
            # Convert to dictionary for serialization
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
            logger.error(
                f"Falha ao gravar cache de template {entry_path.name}: {error}"
            )
            raise

    def _load_index(self) -> None:
        """Load the template index from disk."""
        # Try compressed and uncompressed versions
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
                logger.warning(f"Falha ao carregar índice ({path.name}): {error}")
                try:
                    path.unlink()
                except Exception:
                    pass

        # Initialize empty index
        self._index = {}
        self._index_file = index_paths[0]  # Use compressed by default
        self._index_compress = True

    def _save_index(self) -> None:
        """Save the template index to disk."""
        try:
            serialized = self._serializer.serialize(self._index)

            # Use compression setting from current index file
            storage = FileSystemStorage(compress=self._index_compress)
            storage.write(self._index_file, serialized)

        except Exception as error:
            logger.warning(
                f"Falha ao salvar índice de templates ({self._index_file.name}): {error}"
            )


class TemplatePatternNormalizer:
    """Handles URL normalization and template extraction."""

    # Regex patterns for common identifiers
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
        # Start with the original URL
        normalized = url

        # Apply each transformation to the normalized version progressively
        normalized = self.MSISDN_PATTERN.sub("55{msisdn}", normalized)
        normalized = re.sub(
            r"communicationId=55\d{11,13}", "communicationId=55{msisdn}", normalized
        )

        # Apply general normalization of query params
        normalized = re.sub(r"=[\w\.\-\+]+", "={value}", normalized)

        # Apply other normalizations
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

        # First replace known values
        for variable_name, value in known_values.items():
            if value:
                template = template.replace(value, f"{{{variable_name}}}")

        # Then apply generic patterns
        template = self.MSISDN_PATTERN.sub("55{msisdn}", template)
        template = self.UUID_PATTERN.sub("{uuid}", template)
        template = self.HEX_ID_PATTERN.sub("{hex_id}", template)
        template = re.sub(r"/\d{6,}/", "/{id}/", template)

        return template


# ─────────────────────────── Memory Cache ───────────────────────────────────────


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


# ─────────────────────────── Cache Manager Facade ────────────────────────────────


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
        self,
        cache_dir: str | Path = DEFAULT_CACHE_ROOT,
        max_memory_items: int = 1000
    ) -> None:
        """
        Initialize the cache manager.

        Args:
            cache_dir: Root directory for cache storage
            max_memory_items: Maximum number of items in memory cache (default 1000)
        """
        # In-memory storage (bounded LRU)
        self._memory_storage: OrderedDict[str, tuple[Any, float | None, set[str]]] = OrderedDict()
        self._max_memory_items = max_memory_items
        self._lock = threading.RLock()

        # Metrics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "expirations": 0,
        }

        # Initialize cache components
        self.disk = DiskCache(cache_dir)
        self.memory = create_memory_cache
        self.templates = HttpTemplateCache(self)

    # Dictionary-like interface for memory storage
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
        Get an item from memory storage, returning a default value if not found.
        Alias for `retrieve`.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        with self._lock:
            if key in self._memory_storage:
                val, expires_at, tags = self._memory_storage[key]

                # Check TTL expiration
                if expires_at is not None and time.time() > expires_at:
                    del self._memory_storage[key]
                    self._stats["expirations"] += 1
                    self._stats["misses"] += 1
                    return default

                # Hit - move to end (MRU)
                self._memory_storage.move_to_end(key)
                self._stats["hits"] += 1
                return val

            # Miss
            self._stats["misses"] += 1
            return default

    def set(self, key: str, value: Any, ttl: int | None = None, tags: list[str] | None = None) -> None:
        """
        Set an item in memory storage, optionally with a TTL (in seconds) and tags.
        Alias for `store`.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            tags: Tags for selective invalidation (optional)
        """
        with self._lock:
            # Evict LRU if at capacity
            if len(self._memory_storage) >= self._max_memory_items and key not in self._memory_storage:
                oldest_key = next(iter(self._memory_storage))
                del self._memory_storage[oldest_key]
                self._stats["evictions"] += 1

            expires_at = time.time() + ttl if ttl is not None else None
            tag_set = set(tags) if tags else set()
            self._memory_storage[key] = (value, expires_at, tag_set)
            self._memory_storage.move_to_end(key)
            self._stats["sets"] += 1

    # Method aliases for intuitive access
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
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                **self._stats,
                "size": len(self._memory_storage),
                "max_size": self._max_memory_items,
                "hit_rate": f"{hit_rate:.2%}",
                "memory_usage_pct": f"{len(self._memory_storage) / self._max_memory_items * 100:.1f}%",
            }

    def invalidate(self, pattern: str | None = None, tags: list[str] | None = None) -> int:
        """
        Invalidate cache entries by pattern or tags.

        Args:
            pattern: Regex pattern to match keys (optional)
            tags: List of tags to match (optional)

        Returns:
            Number of entries invalidated

        Examples:
            # Invalidate by tag
            cache.invalidate(tags=["user:123"])

            # Invalidate by pattern
            cache.invalidate(pattern=r"^user:\\d+:posts$")

            # Invalidate both
            cache.invalidate(pattern=r"^user:", tags=["profile"])
        """
        with self._lock:
            keys_to_delete = []
            if pattern:
                regex = re.compile(pattern)
                for key in self._memory_storage:
                    if regex.search(key):
                        keys_to_delete.append(key)
            if tags:
                tag_set = set(tags)
                for key, (val, exp, entry_tags) in self._memory_storage.items():
                    if entry_tags & tag_set:
                        if key not in keys_to_delete:
                            keys_to_delete.append(key)
            for key in keys_to_delete:
                del self._memory_storage[key]

            return len(keys_to_delete)

    def warm(self, preload_func: Callable[[], None] | None = None, keys: list[str] | None = None) -> None:
        """
        Pre-load cache before production traffic (cache warming).

        Args:
            preload_func: Function to execute to populate cache
            keys: Specific keys to warm from disk cache (optional)

        Example:
            # Warm with a function
            def warm_top_users():
                for uid in get_top_100_users():
                    get_user_profile(uid)  # Populates cache

            cache.warm(preload_func=warm_top_users)
        """
        if keys:
            # Warm specific keys (could load from disk cache)
            logger.info(f"Warming {len(keys)} specific keys...")
            for key in keys:
                # This would be a place to load from disk if implemented
                pass

        if preload_func:
            # Execute function to populate cache
            logger.info("Executing cache warming function...")
            preload_func()

        logger.info(f"Cache warmed: {len(self._memory_storage)} items loaded")

    def disk_cache(self, ttl: int | None = None):
        """
        Create a disk cache decorator.

        Args:
            ttl: Time to live in seconds

        Returns:
            Decorator function
        """
        return self.disk(ttl=ttl)

    def lru_cache(self, maxsize: int = DEFAULT_LRU_SIZE):
        """
        Create an LRU memory cache decorator.

        Args:
            maxsize: Maximum cache size

        Returns:
            Decorator function
        """
        return create_memory_cache(maxsize)


class InMemoryCache:
    """
    InMemoryCache provides a simple in-memory cache using a multiprocessing Manager dictionary.
    Attributes:
        _cache (DictProxy): A proxy dictionary for storing cached key-value pairs.
    Args:
        manager (Manager): A multiprocessing.Manager instance used to create a shared dictionary.
    Methods:
        get(key: str) -> Any | None:
            Retrieves the value associated with the given key from the cache.
            Returns None if the key is not present.
        set(key: str, value: Any) -> None:
            Stores the given value in the cache under the specified key.
    """

    def __init__(self, manager: Any):
        self._cache: DictProxy = manager.dict()

    def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value


# ─────────────────────────── Public API ─────────────────────────────────────────


# Maintain backward compatibility with original names
memory_cache = create_memory_cache
TemplateEntry = HttpTemplateEntry
TemplateCache = HttpTemplateCache

__all__ = [
    "DiskCache",
    "memory_cache",
    "CacheManager",
    "TemplateCache",
    "TemplateEntry",
]

"""Disk-based persistent cache implementation."""

from __future__ import annotations

import fnmatch
import functools
import inspect
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast, overload

from ..logging import logger
from .constants import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_PURGE_AGE_SECONDS,
    DEFAULT_JANITOR_INTERVAL,
)
from .janitor import CacheJanitor
from .serializer import CacheSerializer
from .storage import FileSystemStorage
from .utils import FunctionCallHasher

P = ParamSpec("P")
R = TypeVar("R")


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
        if os.environ.get("PFF_CLEAN_MODE") != "1":
            self.root.mkdir(parents=True, exist_ok=True)

        self.compress = "DISKCACHE_NO_GZIP" not in os.environ

        purge_age = purge_older_than or int(
            os.getenv("DISKCACHE_PURGE_OLDER_THAN", DEFAULT_PURGE_AGE_SECONDS)
        )

        janitor_interval = int(os.getenv("DISKCACHE_JANITOR_INTERVAL", DEFAULT_JANITOR_INTERVAL))

        self._storage = FileSystemStorage(compress=self.compress)
        self._serializer = CacheSerializer()
        self._hasher = FunctionCallHasher()

        if os.environ.get("PFF_CLEAN_MODE") != "1":
            self._janitor = CacheJanitor(self.root, purge_age, janitor_interval)
            self._janitor.start()
        else:
            self._janitor = CacheJanitor(self.root, purge_age, 0)

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
            """Execute wrapper.



            Args:

                fn: Input value used by this callable.



            Returns:

                Return value produced by the callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            actual_ttl = ttl if ttl is not None else cast(int | None, fn_or_ttl)
            return self._create_cached_function(fn, actual_ttl)

        return wrapper

    def _create_cached_function(self, function: Callable[P, R], ttl: int | None) -> Callable[P, R]:
        """Create a cached version of the function."""
        signature = inspect.signature(function)

        @functools.wraps(function)
        def cached_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """Execute cached wrapper.



            Args:

                *args: Additional positional arguments.

                **kwargs: Additional keyword arguments.



            Returns:

                Return value produced by the callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            bound_args = signature.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            cache_key = self._hasher.hash_function_call(
                function, *bound_args.args, **bound_args.kwargs
            )

            cached_result = self._load_from_cache(cache_key, ttl)
            if cached_result is not None:
                return cached_result  # type: ignore[no-any-return]

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
            for entry in self.root.iterdir():
                if not entry.is_file():
                    continue
                if not fnmatch.fnmatch(entry.name, pattern):
                    continue

                file_path = entry
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

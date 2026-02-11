"""Cache manager - unified interface for all caching functionality."""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .constants import DEFAULT_CACHE_ROOT
from .disk import DiskCache
from .http_template import HttpTemplateCache
from .utils import create_memory_cache


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

        self._last_accessed_key: str | None = None
        self._consecutive_hits = 0

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
                    self._last_accessed_key = None
                    self._consecutive_hits = 0
                    return default

                if self._last_accessed_key != key:
                    self._memory_storage.move_to_end(key)
                    self._last_accessed_key = key
                    self._consecutive_hits = 1
                else:
                    self._consecutive_hits += 1

                self._stats["hits"] += 1
                return val

            self._stats["misses"] += 1
            self._last_accessed_key = None
            self._consecutive_hits = 0
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
            hit_rate = (self._stats["hits"] / total_requests) if total_requests > 0 else 0.0
            usage_pct = (
                (len(self._memory_storage) / self._max_memory_items * 100)
                if self._max_memory_items > 0
                else 0.0
            )

            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "sets": self._stats["sets"],
                "evictions": self._stats["evictions"],
                "expirations": self._stats["expirations"],
                "hit_rate": f"{hit_rate * 100:.2f}%",
                "hit_rate_pct": round(hit_rate * 100, 2),
                "size": len(self._memory_storage),
                "current_size": len(self._memory_storage),
                "max_size": self._max_memory_items,
                "memory_usage_pct": f"{usage_pct:.1f}%",
            }

    def invalidate(self, tags: list[str] | None = None, pattern: str | None = None) -> int:
        """
        Invalidate cache entries by tags or key pattern.

        Args:
            tags: List of tags to invalidate
            pattern: Key pattern (regex) to invalidate

        Returns:
            Number of entries removed
        """
        removed = 0
        with self._lock:
            keys_to_remove = set()
            if tags:
                for tag in tags:
                    for key, (_, _, t_set) in self._memory_storage.items():
                        if tag in t_set:
                            keys_to_remove.add(key)

            if pattern:
                compiled = re.compile(pattern)
                for key in self._memory_storage:
                    if compiled.search(key):
                        keys_to_remove.add(key)

            for key in keys_to_remove:
                del self._memory_storage[key]
                removed += 1

        return removed

    def warm(self, preload_func=None, keys=None) -> None:
        """Warm the cache by executing a preload function or touching specific keys.

        Args:
            preload_func: Optional callable that populates the cache.
            keys: Optional list of keys to pre-fetch into memory.
        """
        if preload_func is not None:
            preload_func()

    def clear(self) -> None:
        """Clear all items from memory cache."""
        with self._lock:
            self._memory_storage.clear()
            self._stats["evictions"] += self._stats["size"]

"""HTTP template cache for API request patterns."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from ..logging import logger
from .constants import (
    DEFAULT_TEMPLATE_TTL_DAYS,
    DEFAULT_TEMPLATE_INDEX_FLUSH_INTERVAL,
    TEMPLATE_INDEX_FILENAME,
)
from .serializer import CacheSerializer
from .storage import FileSystemStorage
from .utils import FunctionCallHasher, HttpTemplateEntry, TemplatePatternNormalizer

if TYPE_CHECKING:
    from .manager import CacheManager


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
        self._index_dirty = False
        self._index_last_flush = 0.0
        self._index_flush_interval = DEFAULT_TEMPLATE_INDEX_FLUSH_INTERVAL

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
        self._key_locks = defaultdict(threading.Lock)
        self._index_lock = threading.Lock()
        self._lock_pool_lock = threading.Lock()
        self._index_dirty = False
        self._index_last_flush = 0.0
        self._index_flush_interval = DEFAULT_TEMPLATE_INDEX_FLUSH_INTERVAL

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
                logger.warning(
                    f"Failed to read template from cache [{entry_path.name}]: {error}"
                )
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
                logger.warning(
                    f"Failed to remove template file {entry_path.name}: {error}"
                )

    def clear_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        max_idle_time = 30 * 24 * 3600

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
            logger.error(f"Failed to write template cache {entry_path.name}: {error}")
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

    def _flush_index(self, force: bool = False) -> None:
        """Flush the index to disk if dirty and enough time has passed."""
        if not self._index_dirty and not force:
            return

        current_time = time.time()
        time_since_flush = current_time - self._index_last_flush

        if not force and time_since_flush < self._index_flush_interval:
            return

        try:
            serialized = self._serializer.serialize(self._index)
            storage = FileSystemStorage(compress=self._index_compress)
            storage.write(self._index_file, serialized)
            self._index_dirty = False
            self._index_last_flush = current_time
        except Exception as error:
            logger.warning(
                f"Failed to save template index ({self._index_file.name}): {error}"
            )

    def _save_index(self) -> None:
        """Mark index as dirty and attempt flush (legacy compatibility)."""
        self._index_dirty = True
        self._flush_index()

    def flush(self) -> None:
        """Force immediate flush of index to disk."""
        self._flush_index(force=True)

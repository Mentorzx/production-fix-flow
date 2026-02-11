"""Cache module - high-performance caching with disk persistence, memory caching, and HTTP templates.

This module provides a comprehensive caching solution with multiple layers:
- Disk-based persistent cache with optional compression
- In-memory LRU cache for fast access
- Specialized HTTP template cache for API request patterns
"""

from __future__ import annotations

# Protocols
from .protocols import CacheKeyGenerator, Serializer, StorageBackend

# Constants
from .constants import (
    ATOMIC_WRITE_RETRY_COUNT,
    ATOMIC_WRITE_RETRY_DELAY,
    DEFAULT_CACHE_ROOT,
    DEFAULT_JANITOR_INTERVAL,
    DEFAULT_LRU_SIZE,
    DEFAULT_PURGE_AGE_SECONDS,
    DEFAULT_TEMPLATE_INDEX_FLUSH_INTERVAL,
    DEFAULT_TEMPLATE_TTL_DAYS,
    GZIP_COMPRESSION_LEVEL,
    GZIP_MAGIC_BYTES,
    TEMPLATE_INDEX_FILENAME,
    apply_cache_settings_from_config,
    _apply_cache_settings_from_config,
    _load_cache_settings,
)

# Utilities
from .utils import (
    AtomicFileWriter,
    CacheEntry,
    FunctionCallHasher,
    HttpTemplateEntry,
    JsonSafeEncoder,
    TemplatePatternNormalizer,
    create_memory_cache,
)

# Storage
from .storage import FileSystemStorage

# Serialization
from .serializer import CacheSerializer

# Janitor
from .janitor import CacheJanitor, shutdown_all_cache_janitors

# Main cache implementations
from .disk import DiskCache
from .http_template import HttpTemplateCache
from .manager import CacheManager

__all__ = [
    # Protocols
    "CacheKeyGenerator",
    "Serializer",
    "StorageBackend",
    # Constants
    "ATOMIC_WRITE_RETRY_COUNT",
    "ATOMIC_WRITE_RETRY_DELAY",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_JANITOR_INTERVAL",
    "DEFAULT_LRU_SIZE",
    "DEFAULT_PURGE_AGE_SECONDS",
    "DEFAULT_TEMPLATE_INDEX_FLUSH_INTERVAL",
    "DEFAULT_TEMPLATE_TTL_DAYS",
    "GZIP_COMPRESSION_LEVEL",
    "GZIP_MAGIC_BYTES",
    "TEMPLATE_INDEX_FILENAME",
    "apply_cache_settings_from_config",
    "_apply_cache_settings_from_config",
    "_load_cache_settings",
    # Utilities
    "AtomicFileWriter",
    "CacheEntry",
    "FunctionCallHasher",
    "HttpTemplateEntry",
    "JsonSafeEncoder",
    "TemplatePatternNormalizer",
    "create_memory_cache",
    # Storage
    "FileSystemStorage",
    # Serialization
    "CacheSerializer",
    # Janitor
    "CacheJanitor",
    "shutdown_all_cache_janitors",
    # Main classes
    "DiskCache",
    "HttpTemplateCache",
    "CacheManager",
]

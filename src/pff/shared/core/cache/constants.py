"""Cache module constants and configuration."""

from __future__ import annotations

import os
from typing import Any

from pff.shared.core.config import CACHE_CONFIG_PATH, settings

from ..logging import logger


def _load_cache_settings() -> dict[str, Any]:
    """Load cache settings from config file."""
    if os.environ.get("PFF_CLEAN_MODE") == "1":
        return {}
    try:
        from pff.shared.core.file_manager import FileManager
    except Exception as exc:
        logger.warning(f"Failed to import FileManager for cache config: {exc}")
        return {}
    try:
        data = FileManager().read(CACHE_CONFIG_PATH, return_native=True)
        if data is None:
            return {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning(f"Failed to load cache config from {CACHE_CONFIG_PATH}: {exc}")
        return {}


def _apply_cache_settings_from_config() -> None:
    """Apply cache defaults from config file."""
    global DEFAULT_CACHE_ROOT
    global DEFAULT_PURGE_AGE_SECONDS
    global DEFAULT_JANITOR_INTERVAL
    global DEFAULT_TEMPLATE_TTL_DAYS
    global DEFAULT_TEMPLATE_INDEX_FLUSH_INTERVAL
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
    DEFAULT_TEMPLATE_INDEX_FLUSH_INTERVAL = float(
        settings.get(
            "template_index_flush_interval_seconds",
            DEFAULT_TEMPLATE_INDEX_FLUSH_INTERVAL,
        )
    )
    DEFAULT_LRU_SIZE = int(settings.get("lru_size", DEFAULT_LRU_SIZE))
    GZIP_COMPRESSION_LEVEL = int(settings.get("gzip_compression_level", GZIP_COMPRESSION_LEVEL))
    ATOMIC_WRITE_RETRY_COUNT = int(
        settings.get("atomic_write_retry_count", ATOMIC_WRITE_RETRY_COUNT)
    )
    ATOMIC_WRITE_RETRY_DELAY = float(
        settings.get("atomic_write_retry_delay", ATOMIC_WRITE_RETRY_DELAY)
    )


# Default configuration values
DEFAULT_CACHE_ROOT = str(settings.CACHE_DIR)
DEFAULT_PURGE_AGE_SECONDS = 30 * 24 * 3600
DEFAULT_JANITOR_INTERVAL = 3600
DEFAULT_TEMPLATE_TTL_DAYS = 7
DEFAULT_TEMPLATE_INDEX_FLUSH_INTERVAL = 30.0
DEFAULT_LRU_SIZE = 128
GZIP_COMPRESSION_LEVEL = 5
ATOMIC_WRITE_RETRY_COUNT = 5
ATOMIC_WRITE_RETRY_DELAY = 0.1
GZIP_MAGIC_BYTES = b"\x1f\x8b"
TEMPLATE_INDEX_FILENAME = "index.pkl"

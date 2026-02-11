"""Cache module constants and configuration."""

from __future__ import annotations

import os
import threading
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


def _apply_cache_settings_from_config(data: dict[str, Any] | None = None) -> bool:
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

    payload = data if data is not None else _load_cache_settings()
    if not payload:
        return False

    DEFAULT_CACHE_ROOT = payload.get("cache_root", DEFAULT_CACHE_ROOT)
    DEFAULT_PURGE_AGE_SECONDS = int(
        payload.get("purge_age_days", DEFAULT_PURGE_AGE_SECONDS / (24 * 3600)) * 24 * 3600
    )
    DEFAULT_JANITOR_INTERVAL = int(
        payload.get("janitor_interval_seconds", DEFAULT_JANITOR_INTERVAL)
    )
    DEFAULT_TEMPLATE_TTL_DAYS = int(payload.get("template_ttl_days", DEFAULT_TEMPLATE_TTL_DAYS))
    DEFAULT_TEMPLATE_INDEX_FLUSH_INTERVAL = float(
        payload.get(
            "template_index_flush_interval_seconds",
            DEFAULT_TEMPLATE_INDEX_FLUSH_INTERVAL,
        )
    )
    DEFAULT_LRU_SIZE = int(payload.get("lru_size", DEFAULT_LRU_SIZE))
    GZIP_COMPRESSION_LEVEL = int(payload.get("gzip_compression_level", GZIP_COMPRESSION_LEVEL))
    ATOMIC_WRITE_RETRY_COUNT = int(
        payload.get("atomic_write_retry_count", ATOMIC_WRITE_RETRY_COUNT)
    )
    ATOMIC_WRITE_RETRY_DELAY = float(
        payload.get("atomic_write_retry_delay", ATOMIC_WRITE_RETRY_DELAY)
    )
    return True


_CACHE_SETTINGS_LOCK = threading.Lock()
_CACHE_SETTINGS_APPLIED = False


def apply_cache_settings_from_config(*, force: bool = False) -> bool:
    """Apply cache settings once from config file.

    Returns:
        True when settings were loaded from config in this call.
        False when skipped (already applied or no config payload).
    """
    global _CACHE_SETTINGS_APPLIED

    with _CACHE_SETTINGS_LOCK:
        if _CACHE_SETTINGS_APPLIED and not force:
            return False

        payload = _load_cache_settings()
        if not payload:
            if force:
                _CACHE_SETTINGS_APPLIED = False
            return False

        _apply_cache_settings_from_config(payload)
        _CACHE_SETTINGS_APPLIED = True
        return True


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

"""Centralized YAML configuration loader with process-lifetime caching.

Eliminates the pattern of repeated ``FileManager().read(CONFIG_PATH, return_native=True)``
scattered across dozens of modules, each with its own ``@lru_cache`` or
none at all.  Every call with the same *config_path* returns the **same
cached dict** for the lifetime of the process.

Usage::

    from pff.shared.core.config import VALIDATOR_CONFIG_PATH
    from pff.shared.core.config_loader import load_config

    cfg = load_config(VALIDATOR_CONFIG_PATH)
    violation_cfg = cfg.get("violation_scoring", {})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pff.shared.core.cache import create_memory_cache
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger


@create_memory_cache(maxsize=32)
def load_config(config_path: Path) -> dict[str, Any]:
    """Load and cache a YAML configuration file.

    Uses process-lifetime LRU caching — each unique *config_path* is
    read from disk at most once.

    Args:
        config_path: Absolute ``Path`` to the YAML config file
            (typically one of the ``*_CONFIG_PATH`` constants from
            ``pff.shared.core.config``).

    Returns:
        Parsed dict.  Returns ``{}`` on any I/O or parse failure.
    """
    fm = FileManager()
    try:
        raw = fm.read(config_path, return_native=True) or {}
    except Exception as exc:
        logger.warning(f"Failed to load config from {config_path}: {exc}")
        return {}
    return raw if isinstance(raw, dict) else {}

from __future__ import annotations

from typing import Any

from pff.config import CLEANUP_CONFIG_PATH
from pff.utils.core.file_manager import FileManager
from pff.utils.core.logger import logger

CONFIG_PATH = CLEANUP_CONFIG_PATH


def _coerce_positive_int(value: Any, fallback: int) -> int:
    """Return a positive integer, falling back when the input is invalid.

    Args:
        value: Raw value to coerce.
        fallback: Value to return when coercion fails or is non-positive.

    Returns:
        int: Positive integer or the provided fallback.
    """
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return fallback
    return coerced if coerced > 0 else fallback


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries for config overrides.

    Args:
        base: Baseline dictionary.
        override: Override dictionary with user values.

    Returns:
        dict[str, Any]: Deep-merged result.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_cleanup_config(raw_config: Any) -> dict[str, Any]:
    """Extract the cleanup section regardless of top-level nesting.

    Args:
        raw_config: Raw config object read from YAML.

    Returns:
        dict[str, Any]: Normalized cleanup configuration.
    """
    if not isinstance(raw_config, dict):
        return {}
    if "cleanup" in raw_config and isinstance(raw_config["cleanup"], dict):
        return raw_config["cleanup"]
    return raw_config


def load_cleanup_config() -> dict[str, Any]:
    """Load cleanup configuration from YAML using FileManager with safe defaults.

    Returns:
        dict[str, Any]: Cleanup configuration merged with defaults.
    """
    defaults: dict[str, Any] = {
        "retention": {"execution_logs_days": 30, "kg_data_days": None},
        "backup": {
            "dir": "outputs/backups/postgres",
            "keep_last": 5,
        },
        "performance": {
            "max_concurrent_io": 10,
            "large_dir_threshold_bytes": 104857600,
        },
    }

    try:
        raw_config = FileManager.read(CONFIG_PATH)
    except FileNotFoundError:
        return defaults
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Falling back to cleanup defaults: {exc}")
        return defaults

    cleanup_section = _normalize_cleanup_config(raw_config)
    if not cleanup_section:
        return defaults

    merged = _merge_dicts(defaults, cleanup_section)

    retention_cfg = merged.get("retention") if isinstance(merged.get("retention"), dict) else {}
    merged["retention"] = retention_cfg or defaults["retention"].copy()
    merged["retention"]["execution_logs_days"] = _coerce_positive_int(
        merged["retention"].get("execution_logs_days"),
        defaults["retention"]["execution_logs_days"],
    )

    backup_cfg = merged.get("backup") if isinstance(merged.get("backup"), dict) else {}
    merged["backup"] = backup_cfg or defaults["backup"].copy()
    merged["backup"]["keep_last"] = _coerce_positive_int(
        merged["backup"].get("keep_last"),
        defaults["backup"]["keep_last"],
    )

    performance_cfg = merged.get("performance") if isinstance(merged.get("performance"), dict) else {}
    merged["performance"] = performance_cfg or defaults["performance"].copy()
    merged["performance"]["max_concurrent_io"] = _coerce_positive_int(
        merged["performance"].get("max_concurrent_io"),
        defaults["performance"]["max_concurrent_io"],
    )
    merged["performance"]["large_dir_threshold_bytes"] = _coerce_positive_int(
        merged["performance"].get("large_dir_threshold_bytes"),
        defaults["performance"]["large_dir_threshold_bytes"],
    )

    return merged


CLEANUP_CONFIG = load_cleanup_config()

__all__ = [
    "CLEANUP_CONFIG_PATH",
    "CLEANUP_CONFIG",
    "load_cleanup_config",
]

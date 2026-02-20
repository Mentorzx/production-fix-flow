"""FileManager configuration loading with caching.

Centralizes all configuration for the file_manager package using
the project's CacheManager utility for memoization.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

import ruamel.yaml

from pff.shared.core.config import PERFORMANCE_CONFIG_PATH, settings

from ..cache import CacheManager
from ..logging import logger

ParquetCompression = Literal["lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"]

if os.environ.get("FILEMANAGER_DISABLE_CONFIG_CACHE") == "1":

    class _NoopCache:
        def memory(self, *args, **kwargs):
            """Execute memory.



            Args:

                *args: Additional positional arguments.

                **kwargs: Additional keyword arguments.



            Returns:

                Return value produced by the callable.

            """

            def decorator(func):
                """Execute decorator.



                Args:

                    func: Input value used by this callable.



                Returns:

                    Return value produced by the callable.

                """

                return func

            return decorator

    _config_cache = _NoopCache()
else:
    _config_cache = CacheManager(cache_dir=settings.CACHE_DIR / "file_manager_config")  # type: ignore[assignment]


_STREAMING_THRESHOLD_BYTES: int | None = None
_ENCODER_BUFFER_SIZE = int(os.getenv("PFF_MSGSPEC_BUFFER_SIZE", "65536"))


def _load_file_io_config() -> dict[str, Any]:
    """Load file I/O configuration from performance config file."""
    try:
        if not PERFORMANCE_CONFIG_PATH.exists():
            return {}
        content = PERFORMANCE_CONFIG_PATH.read_text(encoding="utf-8")
        yaml_loader = ruamel.yaml.YAML(typ="safe")
        cfg = yaml_loader.load(content) or {}
        perf_cfg = cfg.get("performance", {})
        return perf_cfg.get("file_io", {}) if isinstance(perf_cfg, dict) else {}
    except Exception as exc:
        logger.debug(f"Failed to read file I/O config: {exc}")
        return {}


def _load_file_io_streaming_config() -> dict[str, Any]:
    """Load streaming threshold configuration."""
    file_io_cfg = _load_file_io_config()
    return file_io_cfg.get("streaming_thresholds", {}) if isinstance(file_io_cfg, dict) else {}  # type: ignore[no-any-return]


def _load_file_io_parquet_config() -> dict[str, Any]:
    """Load parquet-first configuration."""
    file_io_cfg = _load_file_io_config()
    return file_io_cfg.get("parquet_first", {}) if isinstance(file_io_cfg, dict) else {}  # type: ignore[no-any-return]


@_config_cache.memory(maxsize=1)
def get_parquet_first_config() -> dict[str, Any]:
    """Load parquet-first config from disk with caching.

    Returns:
        Configuration dictionary with defaults merged with file config.
    """
    cfg = _load_file_io_parquet_config()
    defaults = {
        "raw_chunk_mb": 8,
        "parsed_row_group_size": 200_000,
        "container_flush_rows": 2048,
        "compression": "lz4",
        "compression_level": 3,
        "cache_dir": str(settings.CACHE_DIR / "ingest"),
    }
    merged = defaults | cfg
    return merged


@_config_cache.memory(maxsize=1)
def get_arrow_config() -> dict[str, Any]:
    """Load Arrow IPC configuration from disk with caching."""
    file_io_cfg = _load_file_io_config()
    cfg = file_io_cfg.get("arrow", {})

    defaults = {
        "read_engine": "polars",
        "mmap_enabled": True,
        "rechunk": False,
        "use_threads": True,
        "unify_dictionaries": False,
    }
    return defaults | cfg  # type: ignore[no-any-return]


def get_parquet_cache_root() -> Path:
    """Get the root directory for parquet-first cache."""
    cfg = get_parquet_first_config()
    raw_root = Path(cfg.get("cache_dir", settings.CACHE_DIR / "ingest"))
    if raw_root.is_absolute():
        root = raw_root
    else:
        root = (settings.ROOT_DIR / raw_root).resolve()
    if os.environ.get("PFF_CLEAN_MODE") != "1":
        root.mkdir(parents=True, exist_ok=True)
    return root


def get_parquet_row_group_size() -> int:
    """Get configured row group size for parsed parquet files."""
    cfg = get_parquet_first_config()
    try:
        return int(cfg.get("parsed_row_group_size", 200_000))
    except (TypeError, ValueError):
        return 200_000


def get_container_flush_rows() -> int:
    """Get configured flush threshold for container parquet writes."""
    cfg = get_parquet_first_config()
    try:
        return int(cfg.get("container_flush_rows", 2048))
    except (TypeError, ValueError):
        return 2048


def get_raw_chunk_bytes() -> int:
    """Get configured chunk size for RAW parquet writes."""
    cfg = get_parquet_first_config()
    try:
        return int(cfg.get("raw_chunk_mb", 8)) * 1024 * 1024
    except (TypeError, ValueError):
        return 8 * 1024 * 1024


def get_parquet_compression() -> tuple[ParquetCompression, int | None]:
    """Get configured compression settings for parquet files."""
    cfg = get_parquet_first_config()
    _valid_compressions = {
        "lz4",
        "uncompressed",
        "snappy",
        "gzip",
        "lzo",
        "brotli",
        "zstd",
    }
    raw_compression = str(cfg.get("compression", "lz4"))
    compression: ParquetCompression = cast(
        ParquetCompression,
        raw_compression if raw_compression in _valid_compressions else "lz4",
    )
    level = cfg.get("compression_level", 3)
    try:
        level = int(level)
    except (TypeError, ValueError):
        level = None
    return compression, level


def get_streaming_threshold_bytes() -> int:
    """Get streaming threshold in bytes, computed adaptively based on RAM.

    Uses environment variable PFF_FILE_STREAM_THRESHOLD_MB if set,
    otherwise computes based on available system RAM.
    """
    global _STREAMING_THRESHOLD_BYTES
    if _STREAMING_THRESHOLD_BYTES is not None:
        return _STREAMING_THRESHOLD_BYTES

    env_value = os.getenv("PFF_FILE_STREAM_THRESHOLD_MB")
    if env_value:
        try:
            _STREAMING_THRESHOLD_BYTES = int(env_value) * 1024 * 1024
            return _STREAMING_THRESHOLD_BYTES
        except ValueError:
            logger.warning("Invalid PFF_FILE_STREAM_THRESHOLD_MB; using default fallback.")

    file_io_cfg = _load_file_io_streaming_config()

    try:
        from pff.shared.system.probe import get_system_ram_gb

        total_ram_gb, _ = get_system_ram_gb()
        low_ram_gb = float(file_io_cfg.get("low_ram_gb", 8))
        mid_ram_gb = float(file_io_cfg.get("mid_ram_gb", 24))
        low_ram_mb = int(file_io_cfg.get("low_ram_mb", 64))
        mid_ram_mb = int(file_io_cfg.get("mid_ram_mb", 512))
        high_ram_mb = int(file_io_cfg.get("high_ram_mb", 1024))

        if total_ram_gb < low_ram_gb:
            threshold_mb = low_ram_mb
        elif total_ram_gb < mid_ram_gb:
            threshold_mb = mid_ram_mb
        else:
            threshold_mb = high_ram_mb
        logger.debug(f"Adaptive streaming threshold: {threshold_mb}MB (RAM={total_ram_gb:.1f}GB)")
        _STREAMING_THRESHOLD_BYTES = threshold_mb * 1024 * 1024
        return _STREAMING_THRESHOLD_BYTES
    except Exception as exc:
        logger.debug(f"Failed to compute adaptive threshold: {exc}")

    fallback_mb = int(file_io_cfg.get("high_ram_mb", 128))
    _STREAMING_THRESHOLD_BYTES = fallback_mb * 1024 * 1024
    return _STREAMING_THRESHOLD_BYTES


def get_encoder_buffer_size() -> int:
    """Get msgspec encoder buffer size from environment."""
    return _ENCODER_BUFFER_SIZE


def get_zip_parquet_cache_path(
    zip_path: Path,
    stat_sig: tuple[int, int],
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Generate cache path for ZIP parquet using mtime_ns for collision avoidance."""
    base_dir = cache_dir or (settings.CACHE_DIR / "zip_parquet")
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = zip_path.stem.replace(" ", "_")
    return base_dir / f"{stem}_{stat_sig[0]}_{stat_sig[1]}.parquet"

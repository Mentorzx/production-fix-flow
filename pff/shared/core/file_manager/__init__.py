"""FileManager package - state-of-the-art file I/O with parquet-first caching.

This module provides backward-compatible imports for existing code while
organizing the implementation into a clean SRP-compliant structure.

Design Patterns Used:
- Strategy: FileHandler implementations for different file formats
- Template Method: IngestionPipeline for file/zip/zstd ingestion
- Factory: Handler registry with get_handler()
- Facade: FileManager class providing unified interface

Package Structure:
- handlers/: FileHandler Strategy implementations
- ingestion/: Template Method ingestion pipelines
- materializers/: Strategy for bundle-to-native conversion
- container/: ZIP and container utilities
- config.py: Centralized configuration with caching
- bundles.py: ParquetBundle dataclass
- utils.py: Shared utilities
- async_io.py: Async I/O primitives
- _legacy.py: Original monolithic implementation (for migration)
"""

from __future__ import annotations

# Import FileManager facade and core types
from .manager import FileManager
from .bundles import ParquetBundle

# Config (uses CacheManager for memoization)
from .config import (
    get_parquet_first_config,
    get_parquet_cache_root,
    get_streaming_threshold_bytes,
    get_raw_chunk_bytes,
    get_zip_parquet_cache_path,
)

from .utils import (
    compute_sha256,
    detect_encoding_sample,
    fast_suffix,
    ensure_dir,
    make_json_safe,
    read_manifest,
    write_manifest,
    read_raw_bytes,
    get_index_manifest_path,
    get_json_encoder,
    encode_json,
    decode_json,
    encode_msgpack,
)
from .async_io import (
    read_async_content,
    write_async_bytes,
    write_async_text,
    async_ensure_dir,
    run_coroutine_sync,
)

# Re-export handlers
from .handlers import (
    FileHandler,
    CSVHandler,
    ParquetHandler,
    JSONHandler,
    YAMLHandler,
    TextHandler,
    BinHandler,
    PickleHandler,
    NumPyHandler,
    ExcelHandler,
    NDJSONHandler,
    ZstdHandler,
    HANDLER_FACTORIES,
    SUPPORTED_EXTS,
    get_handler,
    clear_handler_cache,
)

# Re-export materializers
from .materializers import (
    Materializer,
    materialize_bundle,
    get_materializer,
    register_materializer,
)

# Re-export ingestion
from .ingestion import (
    IngestionPipeline,
    FileIngestionPipeline,
    ZipIngestionPipeline,
    ZstdIngestionPipeline,
    get_pipeline,
    ingest as ingest_file,
)

# Re-export container utilities
from .container import (
    get_cached_zip_members,
    process_zip_entry,
    load_zip_from_bytes,
    iter_zip_entries,
)


# Backward compatibility aliases
_HANDLER_FACTORIES = HANDLER_FACTORIES
_SUPPORTED_EXTS = SUPPORTED_EXTS
_get_handler = get_handler
_fast_suffix = fast_suffix
_ensure_dir = ensure_dir
_make_json_safe = make_json_safe
_read_manifest = read_manifest
_write_manifest = write_manifest
_read_raw_bytes = read_raw_bytes
_compute_sha256 = compute_sha256
_zip_parquet_cache_path = get_zip_parquet_cache_path

# Re-export os module and internal functions for test compatibility
import os  # noqa: E402, F401

SUPPORTED_EXTS = SUPPORTED_EXTS

# Backward-compatible internal helpers for tests/benchmarks
_get_json_encoder = get_json_encoder
_compute_sha256 = compute_sha256
_index_manifest_path = get_index_manifest_path
_parquet_first_cache_root = get_parquet_cache_root
_parquet_first_raw_chunk_bytes = get_raw_chunk_bytes


__all__ = [
    # Facade
    "FileManager",
    # Core types
    "ParquetBundle",
    # Config
    "get_parquet_first_config",
    "get_parquet_cache_root",
    "get_streaming_threshold_bytes",
    "get_raw_chunk_bytes",
    "get_zip_parquet_cache_path",
    # Utils
    "compute_sha256",
    "detect_encoding_sample",
    "fast_suffix",
    "ensure_dir",
    "make_json_safe",
    "read_manifest",
    "write_manifest",
    "read_raw_bytes",
    "encode_json",
    "decode_json",
    "encode_msgpack",
    # Async I/O
    "read_async_content",
    "write_async_bytes",
    "write_async_text",
    "async_ensure_dir",
    "run_coroutine_sync",
    # Handlers
    "FileHandler",
    "CSVHandler",
    "ParquetHandler",
    "JSONHandler",
    "YAMLHandler",
    "TextHandler",
    "BinHandler",
    "PickleHandler",
    "NumPyHandler",
    "ExcelHandler",
    "NDJSONHandler",
    "ZstdHandler",
    "HANDLER_FACTORIES",
    "SUPPORTED_EXTS",
    "get_handler",
    "clear_handler_cache",
    # Materializers
    "Materializer",
    "materialize_bundle",
    "get_materializer",
    "register_materializer",
    # Ingestion
    "IngestionPipeline",
    "FileIngestionPipeline",
    "ZipIngestionPipeline",
    "ZstdIngestionPipeline",
    "get_pipeline",
    "ingest_file",
    # Container
    "get_cached_zip_members",
    "process_zip_entry",
    "load_zip_from_bytes",
    "iter_zip_entries",
]

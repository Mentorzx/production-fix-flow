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

from .async_io import (
    async_ensure_dir,
    read_async_content,
    run_coroutine_sync,
    write_async_bytes,
    write_async_text,
)
from .bundles import ParquetBundle
from .config import (
    get_parquet_cache_root,
    get_parquet_first_config,
    get_raw_chunk_bytes,
    get_streaming_threshold_bytes,
    get_zip_parquet_cache_path,
)
from .container import (
    get_cached_zip_members,
    iter_zip_entries,
    load_zip_from_bytes,
    process_zip_entry,
)
from .handlers import (
    HANDLER_FACTORIES,
    SUPPORTED_EXTS,
    BinHandler,
    CSVHandler,
    ExcelHandler,
    FileHandler,
    JSONHandler,
    NDJSONHandler,
    NumPyHandler,
    ParquetHandler,
    PickleHandler,
    TextHandler,
    YAMLHandler,
    ZstdHandler,
    clear_handler_cache,
    get_handler,
)
from .ingestion import (
    FileIngestionPipeline,
    IngestionPipeline,
    ZipIngestionPipeline,
    ZstdIngestionPipeline,
    get_pipeline,
)
from .ingestion import (
    ingest as ingest_file,
)
from .manager import FileManager
from .materializers import (
    Materializer,
    get_materializer,
    materialize_bundle,
    register_materializer,
)
from .utils import (
    compute_sha256,
    decode_json,
    detect_encoding_sample,
    encode_json,
    encode_msgpack,
    ensure_dir,
    fast_suffix,
    get_index_manifest_path,
    get_json_encoder,
    make_json_safe,
    read_manifest,
    read_raw_bytes,
    write_manifest,
)

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


SUPPORTED_EXTS = SUPPORTED_EXTS


_get_json_encoder = get_json_encoder
_compute_sha256 = compute_sha256
_index_manifest_path = get_index_manifest_path
_parquet_first_cache_root = get_parquet_cache_root
_parquet_first_raw_chunk_bytes = get_raw_chunk_bytes


__all__ = [
    "FileManager",
    "ParquetBundle",
    "get_parquet_first_config",
    "get_parquet_cache_root",
    "get_streaming_threshold_bytes",
    "get_raw_chunk_bytes",
    "get_zip_parquet_cache_path",
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
    "read_async_content",
    "write_async_bytes",
    "write_async_text",
    "async_ensure_dir",
    "run_coroutine_sync",
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
    "Materializer",
    "materialize_bundle",
    "get_materializer",
    "register_materializer",
    "IngestionPipeline",
    "FileIngestionPipeline",
    "ZipIngestionPipeline",
    "ZstdIngestionPipeline",
    "get_pipeline",
    "ingest_file",
    "get_cached_zip_members",
    "process_zip_entry",
    "load_zip_from_bytes",
    "iter_zip_entries",
]

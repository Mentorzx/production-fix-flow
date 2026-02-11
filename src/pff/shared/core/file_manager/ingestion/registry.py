"""Pipeline registry for file ingestion.

Provides a single entry point for file ingestion that dispatches
to the appropriate pipeline based on file extension.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..handlers import SUPPORTED_EXTS
from .file import FileIngestionPipeline
from .zip import ZipIngestionPipeline
from .zstd import ZstdIngestionPipeline

if TYPE_CHECKING:
    from ..bundles import ParquetBundle
    from .base import IngestionPipeline


_FILE_PIPELINE = FileIngestionPipeline()
_ZIP_PIPELINE = ZipIngestionPipeline()
_ZSTD_PIPELINE = ZstdIngestionPipeline()


def get_pipeline(path: Path) -> IngestionPipeline:
    """Get the appropriate pipeline for a file path.

    Args:
        path: File path to ingest.

    Returns:
        Appropriate IngestionPipeline instance.

    Raises:
        ValueError: If extension is not supported.
    """
    ext = path.suffix.lower()

    if ext == ".zip":
        return _ZIP_PIPELINE

    if ext in {".zst", ".zstd"}:
        return _ZSTD_PIPELINE

    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported extension: {ext}")

    return _FILE_PIPELINE


def ingest(
    path: str | Path,
    *,
    build_parsed: bool = True,
    cache: bool = True,
    **kwargs: Any,
) -> ParquetBundle:
    """Ingest a file using the appropriate pipeline.

    This is the unified entry point for file ingestion that replaces
    the legacy FileManager._ingest_* methods.

    Args:
        path: File path to ingest.
        build_parsed: Whether to build the parsed parquet layer.
        cache: Whether to use caching.
        **kwargs: Additional options passed to the pipeline.

    Returns:
        ParquetBundle with RAW and optionally PARSED layers.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the extension is not supported.
    """
    p = Path(path)

    if p.is_dir():
        raise ValueError("Use ingest_directory for folders")

    if not p.exists():
        raise FileNotFoundError(f"Missing source file: {p}")

    pipeline = get_pipeline(p)
    return pipeline.ingest(p, build_parsed=build_parsed, cache=cache, **kwargs)

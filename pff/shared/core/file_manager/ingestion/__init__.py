"""Ingestion package - Template Method pipelines for file ingestion."""

from .base import IngestionPipeline
from .file import FileIngestionPipeline
from .zip import ZipIngestionPipeline
from .zstd import ZstdIngestionPipeline
from .registry import get_pipeline, ingest

__all__ = [
    "IngestionPipeline",
    "FileIngestionPipeline",
    "ZipIngestionPipeline",
    "ZstdIngestionPipeline",
    "get_pipeline",
    "ingest",
]

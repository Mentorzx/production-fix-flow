"""Concrete file ingestion pipeline using Template Method pattern."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from ..handlers import get_handler
from ..utils import make_json_safe
from .base import IngestionPipeline

if TYPE_CHECKING:
    from ..bundles import ParquetBundle


class FileIngestionPipeline(IngestionPipeline):
    """Ingestion pipeline for regular files (CSV, JSON, YAML, Parquet, etc.)."""

    def _probe(self, path: Path) -> tuple[int, int]:
        """Get file stat signature (mtime_ns, size_bytes)."""
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size

    def _get_extension(self, path: Path) -> str:
        """Get file extension."""
        return path.suffix.lower()

    def _build_raw(
        self,
        path: Path,
        *,
        file_id: str,
        sha256: str,
        stat_sig: tuple[int, int],
        cache_root: Path,
        chunk_size: int,
        **kwargs: Any,
    ) -> ParquetBundle:
        """Build RAW parquet layer for the file."""
        from ..bundles import ParquetBundle
        from ..parquet_io import write_raw_parquet, write_raw_parquet_from_bytes
        from ..utils import detect_encoding_sample

        ext = self._get_extension(path)
        bundle_dir = cache_root / file_id
        raw_parquet_path = bundle_dir / "raw.parquet"

        encoding = None
        if ext in {
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".jsonl",
            ".ndjson",
            ".csv",
            ".tsv",
        }:
            encoding = detect_encoding_sample(path)

        if not raw_parquet_path.exists():
            raw_bytes = kwargs.get("raw_bytes")
            if isinstance(raw_bytes, (bytes, bytearray, memoryview)):
                write_raw_parquet_from_bytes(
                    raw_bytes,
                    raw_parquet_path,
                    source_path=path,
                    file_id=file_id,
                    sha256=sha256,
                    stat_sig=stat_sig,
                    encoding=encoding,
                    extra_metadata=None,
                    chunk_size=chunk_size,
                )
            else:
                write_raw_parquet(
                    path,
                    raw_parquet_path,
                    file_id=file_id,
                    sha256=sha256,
                    stat_sig=stat_sig,
                    encoding=encoding,
                    extra_metadata=None,
                    chunk_size=chunk_size,
                )

        metadata: dict[str, Any] = {
            "sha256": sha256,
            "mtime_ns": stat_sig[0],
            "size_bytes": stat_sig[1],
            "encoding": encoding,
        }

        return ParquetBundle(
            source_path=path,
            ext=ext,
            file_id=file_id,
            raw_parquet_path=raw_parquet_path,
            parsed_parquet_path=None,
            parsed_kind="none",
            metadata=metadata,
        )

    def _build_parsed(self, bundle: ParquetBundle, **kwargs: Any) -> None:
        """Build PARSED parquet layer based on file type."""
        from ..parquet_io import (
            write_parsed_payload_parquet,
            write_tabular_parquet_from_path,
        )

        ext = bundle.ext
        bundle_dir = bundle.raw_parquet_path.parent
        parsed_parquet_path = bundle_dir / "parsed.parquet"

        if ext in {".parquet", ".pq", ".parq"}:
            bundle.parsed_parquet_path = bundle.source_path
            bundle.parsed_kind = "tabular"
            bundle.metadata["parsed_is_source"] = True
            return

        if ext in {".arrow", ".ipc", ".feather"}:
            if not parsed_parquet_path.exists():
                import polars as pl

                pl.scan_ipc(str(bundle.source_path)).sink_parquet(
                    parsed_parquet_path,
                    compression="lz4",
                    statistics=True,
                    row_group_size=200_000,
                )
            bundle.parsed_parquet_path = parsed_parquet_path
            bundle.parsed_kind = "tabular"
            return

        if ext in {".csv", ".tsv", ".ndjson", ".jsonl"}:
            bundle.parsed_kind = "tabular"
            if not parsed_parquet_path.exists():
                write_tabular_parquet_from_path(
                    bundle.source_path,
                    parsed_parquet_path,
                    ext=ext,
                    **kwargs,
                )
            bundle.parsed_parquet_path = parsed_parquet_path
            return

        if ext in {".yaml", ".yml"}:
            bundle.parsed_kind = "yaml"
            if not parsed_parquet_path.exists():
                raw = bundle.source_path.read_bytes()
                handler = get_handler(ext)
                obj = handler.load_bytes(raw) if handler is not None else raw

                payload_msgpack = msgspec.msgpack.encode(make_json_safe(obj))
                encoding = bundle.metadata.get("encoding")
                write_parsed_payload_parquet(
                    parsed_parquet_path,
                    file_id=bundle.file_id,
                    payload_text=None,
                    payload_msgpack=payload_msgpack,
                    payload_bytes=None,
                    parsed_kind="yaml",
                    parse_metadata={"encoding": encoding} if encoding else {},
                )
            bundle.parsed_parquet_path = parsed_parquet_path
            return

        if ext == ".json":
            bundle.parsed_kind = "json"
            if not parsed_parquet_path.exists():
                raw = bundle.source_path.read_bytes()
                handler = get_handler(ext)
                obj = handler.load_bytes(raw) if handler is not None else raw

                payload_msgpack = msgspec.msgpack.encode(make_json_safe(obj))
                encoding = bundle.metadata.get("encoding")
                write_parsed_payload_parquet(
                    parsed_parquet_path,
                    file_id=bundle.file_id,
                    payload_text=None,
                    payload_msgpack=payload_msgpack,
                    payload_bytes=None,
                    parsed_kind="json",
                    parse_metadata={"encoding": encoding} if encoding else {},
                )
            bundle.parsed_parquet_path = parsed_parquet_path
            return

        if ext == ".txt":
            bundle.parsed_kind = "text"
            if not parsed_parquet_path.exists():
                encoding = bundle.metadata.get("encoding") or "utf-8"
                text = bundle.source_path.read_text(encoding=encoding, errors="ignore")
                write_parsed_payload_parquet(
                    parsed_parquet_path,
                    file_id=bundle.file_id,
                    payload_text=text,
                    payload_msgpack=None,
                    payload_bytes=None,
                    parsed_kind=bundle.parsed_kind,
                    parse_metadata={"encoding": encoding},
                )
            bundle.parsed_parquet_path = parsed_parquet_path
            return

        bundle.parsed_kind = "none"

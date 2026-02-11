"""ZIP file ingestion pipeline using Template Method pattern."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import get_container_flush_rows
from ..handlers import SUPPORTED_EXTS
from ..utils import fast_suffix
from .base import IngestionPipeline

if TYPE_CHECKING:
    from ..bundles import ParquetBundle


class ZipIngestionPipeline(IngestionPipeline):
    """Ingestion pipeline for ZIP archives."""

    def _probe(self, path: Path) -> tuple[int, int]:
        """Get file stat signature (mtime_ns, size_bytes)."""
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size

    def _get_extension(self, path: Path) -> str:
        """Get file extension."""
        return ".zip"

    def _get_supported_members(
        self, path: Path, max_members: int | None = None
    ) -> list[str]:
        """Get list of supported ZIP members."""
        with zipfile.ZipFile(path, "r") as zf:
            members = [
                m
                for m in zf.namelist()
                if not m.endswith("/") and fast_suffix(m) in SUPPORTED_EXTS
            ]
        if max_members:
            members = members[:max_members]
        return members

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
        """Build RAW parquet layer for ZIP file."""
        from ..bundles import ParquetBundle
        from ..parquet_io import write_raw_parquet, write_raw_parquet_from_bytes

        bundle_dir = cache_root / file_id
        raw_parquet_path = bundle_dir / "raw.parquet"

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
                    encoding=None,
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
                    encoding=None,
                    extra_metadata=None,
                    chunk_size=chunk_size,
                )

        metadata: dict[str, Any] = {
            "sha256": sha256,
            "mtime_ns": stat_sig[0],
            "size_bytes": stat_sig[1],
        }

        return ParquetBundle(
            source_path=path,
            ext=".zip",
            file_id=file_id,
            raw_parquet_path=raw_parquet_path,
            parsed_parquet_path=None,
            parsed_kind="none",
            metadata=metadata,
        )

    def _build_parsed(self, bundle: ParquetBundle, **kwargs: Any) -> None:
        """Build PARSED container parquet from ZIP contents."""
        from ..container.parquet import (
            write_container_parquet_from_zip,
            write_container_parquet_index,
        )

        bundle_dir = bundle.raw_parquet_path.parent
        parsed_parquet_path = bundle_dir / "parsed.parquet"

        max_members = kwargs.get("max_members")
        use_mmap = kwargs.get("use_mmap", True)
        handler_kwargs = kwargs.get("handler_kwargs", {})
        read_parallel = bool(kwargs.pop("read_parallel", False))
        read_chunk_size = kwargs.pop("read_chunk_size", None)
        read_task_type = str(kwargs.pop("read_task_type", "thread"))
        lazy_payloads = bool(kwargs.pop("lazy_payloads", True))

        members = self._get_supported_members(bundle.source_path, max_members)
        text_like_exts = {".txt", ".json", ".yaml", ".yml"}
        all_text_like = all(fast_suffix(m) in text_like_exts for m in members)

        if read_chunk_size is None:
            read_chunk_size = max(16, min(256, get_container_flush_rows() // 2))

        if not parsed_parquet_path.exists():
            if lazy_payloads and all_text_like:
                container_meta = write_container_parquet_index(
                    members,
                    parsed_parquet_path,
                    file_id=bundle.file_id,
                )
            else:
                container_meta = write_container_parquet_from_zip(
                    bundle.source_path,
                    parsed_parquet_path,
                    file_id=bundle.file_id,
                    members=members,
                    handler_kwargs=handler_kwargs,
                    use_mmap=use_mmap,
                    read_parallel=read_parallel,
                    read_chunk_size=read_chunk_size,
                    read_task_type=read_task_type,
                )
            bundle.metadata.update(container_meta)

        bundle.parsed_parquet_path = parsed_parquet_path
        bundle.parsed_kind = "container"

"""Base ingestion pipeline using Template Method pattern.

The IngestionPipeline class defines the skeleton of the ingestion algorithm,
with concrete steps implemented by subclasses for file, ZIP, and zstd ingestion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..bundles import ParquetBundle

from ..config import (
    get_parquet_cache_root,
    get_raw_chunk_bytes,
    get_streaming_threshold_bytes,
)
from ..utils import (
    compute_sha256,
    compute_sha256_buffer,
    compute_sha256_bytes,
    get_index_manifest_path,
    memory_map_file,
    read_manifest,
    write_manifest,
)


class IngestionPipeline(ABC):
    """Template Method pattern for file ingestion.

    Subclasses implement:
    - _probe(): Get file stat signature
    - _get_extension(): Get file extension
    - _build_raw(): Build RAW parquet layer
    - _build_parsed(): Build PARSED parquet layer (optional)

    Base class handles:
    - Cache lookup and validation
    - Manifest persistence
    - SHA256 computation and verification
    """

    def ingest(
        self,
        path: Path,
        *,
        build_parsed: bool = True,
        cache: bool = True,
        **kwargs: Any,
    ) -> ParquetBundle:
        """Template method orchestrating the ingestion pipeline.

        Args:
            path: Path to file to ingest.
            build_parsed: Whether to build PARSED parquet layer.
            cache: Whether to use/update cache.
            **kwargs: Additional options passed to build methods.

        Returns:
            ParquetBundle with RAW and optionally PARSED layers.
        """
        stat_sig = self._probe(path)
        cache_root = get_parquet_cache_root()
        index_path = get_index_manifest_path(cache_root, path, stat_sig)
        chunk_size = get_raw_chunk_bytes()
        computed_sha: str | None = None

        if cache:
            bundle = self._cache_lookup(path, stat_sig, index_path, chunk_size)
            if bundle is not None:
                return bundle

        raw_bytes: bytes | bytearray | memoryview | None = None
        if computed_sha is None:
            size_bytes = stat_sig[1]
            try:
                with memory_map_file(path) as mm:
                    raw_bytes = mm  # type: ignore[assignment]
                    sha256 = compute_sha256_buffer(mm)
            except Exception:
                if size_bytes <= get_streaming_threshold_bytes():
                    raw_bytes = path.read_bytes()
                    sha256 = compute_sha256_bytes(raw_bytes)
                else:
                    sha256 = compute_sha256(path, chunk_size=chunk_size)
        else:
            sha256 = computed_sha
        file_id = sha256

        if raw_bytes is not None:
            bundle = self._build_raw(
                path,
                file_id=file_id,
                sha256=sha256,
                stat_sig=stat_sig,
                cache_root=cache_root,
                chunk_size=chunk_size,
                raw_bytes=raw_bytes,
                **kwargs,
            )
        else:
            bundle = self._build_raw(
                path,
                file_id=file_id,
                sha256=sha256,
                stat_sig=stat_sig,
                cache_root=cache_root,
                chunk_size=chunk_size,
                **kwargs,
            )

        if build_parsed:
            self._build_parsed(bundle, **kwargs)

        self._persist_manifest(bundle, index_path, cache=cache, **kwargs)

        return bundle

    def _cache_lookup(
        self,
        path: Path,
        stat_sig: tuple[int, int],
        index_path: Path,
        chunk_size: int,
    ) -> ParquetBundle | None:
        """Look up bundle in cache and validate.

        Returns:
            Cached bundle if valid, None if cache miss or invalid.
        """

        manifest = read_manifest(index_path)
        if not manifest:
            return None

        bundle = self._bundle_from_manifest(manifest)
        if bundle is None:
            return None

        if not bundle.raw_parquet_path.exists():
            return None

        if bundle.parsed_parquet_path is not None and not bundle.parsed_parquet_path.exists():
            return None

        expected_sha = manifest.get("sha256") or bundle.metadata.get("sha256")
        if expected_sha:
            computed_sha = compute_sha256(path, chunk_size=chunk_size)
            if computed_sha != expected_sha:
                return None

        return bundle

    def _bundle_from_manifest(self, data: dict[str, Any]) -> ParquetBundle | None:
        """Create ParquetBundle from manifest dict."""
        from ..bundles import ParquetBundle

        try:
            parsed_path = data.get("parsed_parquet_path")
            return ParquetBundle(
                source_path=Path(data["source_path"]),
                ext=str(data.get("ext", "")),
                file_id=str(data["file_id"]),
                raw_parquet_path=Path(data["raw_parquet_path"]),
                parsed_parquet_path=Path(parsed_path) if parsed_path else None,
                parsed_kind=data.get("parsed_kind", "none"),
                metadata=data.get("metadata", {}),
                dirty=bool(data.get("dirty", False)),
            )
        except Exception:
            return None

    def _persist_manifest(
        self,
        bundle: ParquetBundle,
        index_path: Path,
        *,
        cache: bool = True,
        **kwargs: Any,
    ) -> None:
        """Write manifest to bundle directory and index."""
        manifest = {
            "file_id": bundle.file_id,
            "source_path": str(bundle.source_path),
            "ext": bundle.ext,
            "mtime_ns": bundle.metadata.get("mtime_ns"),
            "size_bytes": bundle.metadata.get("size_bytes"),
            "sha256": bundle.metadata.get("sha256"),
            "raw_parquet_path": str(bundle.raw_parquet_path),
            "parsed_parquet_path": (
                str(bundle.parsed_parquet_path) if bundle.parsed_parquet_path else None
            ),
            "parsed_kind": bundle.parsed_kind,
            "metadata": bundle.metadata,
            "dirty": bundle.dirty,
        }

        bundle_dir = bundle.raw_parquet_path.parent
        bundle_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(bundle_dir / "manifest.json", manifest)

        if cache:
            write_manifest(index_path, manifest)

    @abstractmethod
    def _probe(self, path: Path) -> tuple[int, int]:
        """Get file stat signature (mtime_ns, size_bytes).

        Args:
            path: File path to probe.

        Returns:
            Tuple of (mtime_ns, size_bytes).
        """
        ...

    @abstractmethod
    def _get_extension(self, path: Path) -> str:
        """Get file extension for this path.

        Args:
            path: File path.

        Returns:
            Lowercase extension with dot (e.g., ".csv").
        """
        ...

    @abstractmethod
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
        """Build RAW parquet layer.

        Args:
            path: Source file path.
            file_id: Unique file identifier (SHA256).
            sha256: SHA256 hash string.
            stat_sig: Tuple of (mtime_ns, size_bytes).
            cache_root: Root directory for parquet cache.
            chunk_size: Chunk size for RAW parquet.
            **kwargs: Additional options.

        Returns:
            ParquetBundle with raw_parquet_path set.
        """
        ...

    @abstractmethod
    def _build_parsed(self, bundle: ParquetBundle, **kwargs: Any) -> None:
        """Build PARSED parquet layer (in-place modification).

        Args:
            bundle: Bundle to update with parsed layer.
            **kwargs: Additional options.
        """
        ...

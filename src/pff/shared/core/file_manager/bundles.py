"""ParquetBundle and related dataclasses for file_manager package."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

if TYPE_CHECKING:
    pass

from ..logging import logger


@dataclass
class ParquetBundle:
    """Parquet-first artifact with lossless RAW and optimized PARSED layers.

    Attributes:
        source_path: Original file path that was ingested.
        ext: File extension (lowercase with dot, e.g., ".csv").
        file_id: Unique identifier (typically SHA256 of content).
        raw_parquet_path: Path to RAW parquet containing chunked bytes.
        parsed_parquet_path: Path to PARSED parquet (tabular/json/text/container).
        parsed_kind: Type of parsed content.
        metadata: Additional metadata about the bundle.
        dirty: Whether the bundle has been modified since ingestion.
    """

    source_path: Path
    ext: str
    file_id: str
    raw_parquet_path: Path
    parsed_parquet_path: Path | None
    parsed_kind: Literal["tabular", "json", "yaml", "text", "bytes", "container", "none"]
    metadata: dict[str, Any] = field(default_factory=dict)
    dirty: bool = False

    def lazyframe(self) -> pl.LazyFrame:
        """Get a LazyFrame for tabular bundles."""
        if self.parsed_parquet_path is None:
            raise ValueError("No parsed parquet available")
        return pl.scan_parquet(self.parsed_parquet_path)

    def to_native(self, **kwargs: Any) -> Any:
        """Convert bundle to native Python object using materializers."""

        from .materializers import materialize_bundle

        return materialize_bundle(self, **kwargs)

    def mark_dirty(self) -> None:
        """Mark the bundle as modified."""
        self.dirty = True

    def iter_entries(self) -> Iterator[tuple[str, Any]]:
        """Iterate over container entries.

        Yields:
            Tuples of (entry_name, entry_value) for each file in the container.

        Raises:
            ValueError: If this is not a container bundle.
        """
        if self.parsed_kind != "container":
            raise ValueError("iter_entries is only available for container bundles")
        if not self.parsed_parquet_path:
            raise ValueError("Parsed parquet not available for container bundle")
        from .container.entries import iter_container_entries

        yield from iter_container_entries(
            parsed_parquet_path=self.parsed_parquet_path,
            raw_parquet_path=self.raw_parquet_path,
            source_ext=self.ext,
            metadata=self.metadata,
        )

    def iter_json_entries_as_dataframe(self, batch_size: int = 1000) -> Iterator[pl.DataFrame]:
        """Iterate over JSON container entries as DataFrames with parsed structs.

        This is an optimized alternative to iter_entries() for JSON-heavy containers.
        Instead of parsing JSON individually, it batches entries and returns DataFrames
        with a 'data' struct column containing the parsed JSON.

        Args:
            batch_size: Number of entries per DataFrame batch.

        Yields:
            DataFrames with columns: entry_name (str), data (struct)

        Raises:
            ValueError: If this is not a container bundle.
        """
        if self.parsed_kind != "container":
            raise ValueError(
                "iter_json_entries_as_dataframe is only available for container bundles"
            )
        if not self.parsed_parquet_path:
            raise ValueError("Parsed parquet not available for container bundle")
        from .container.entries import iter_container_entries

        batch_names: list[str] = []
        batch_data: list[dict[str, Any]] = []
        for name, payload in iter_container_entries(
            parsed_parquet_path=self.parsed_parquet_path,
            raw_parquet_path=self.raw_parquet_path,
            source_ext=self.ext,
            metadata=self.metadata,
        ):
            if not isinstance(payload, dict):
                continue
            batch_names.append(name)
            batch_data.append(payload)
            if len(batch_names) < batch_size:
                continue
            df = _build_json_entries_dataframe(batch_names, batch_data)
            if df is not None:
                yield df
            batch_names, batch_data = [], []

        df = _build_json_entries_dataframe(batch_names, batch_data)
        if df is not None:
            yield df


def _build_json_entries_dataframe(
    entry_names: list[str],
    entry_data: list[dict[str, Any]],
) -> pl.DataFrame | None:
    """Execute build json entries dataframe.



    Args:

        entry_names: Input value used by this callable.

        entry_data: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    if not entry_names:
        return None
    try:
        return pl.DataFrame({"entry_name": entry_names, "data": entry_data})
    except Exception as exc:
        logger.debug(f"Failed to create DataFrame from JSON batch: {exc}")
        return None

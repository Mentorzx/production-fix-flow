"""Persistence ports for KG checkpoints, splits, and mappings."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol


class PipelineCheckpointsPort(Protocol):
    """Port for accessing pipeline checkpoints persistence."""

    async def save_checkpoint(
        self,
        pipeline_name: str,
        step_name: str,
        status: str,
        progress: float = 0.0,
        metadata: dict[str, Any] | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> int:
        """Persist checkpoint state for a pipeline step."""
        ...

    async def get_checkpoint(self, pipeline_name: str, step_name: str) -> dict[str, Any] | None:
        """Load checkpoint payload for a single pipeline step."""
        ...

    async def get_pipeline_checkpoints(self, pipeline_name: str) -> list[dict[str, Any]]:
        """Load all checkpoints associated with a pipeline."""
        ...

    async def reset_pipeline(self, pipeline_name: str) -> int:
        """Reset checkpoint state for a pipeline."""
        ...

    async def delete_pipeline_checkpoints(self, pipeline_name: str) -> int:
        """Delete all persisted checkpoints for a pipeline."""
        ...

    async def checkpoint_exists(self, pipeline_name: str, step_name: str) -> bool:
        """Return whether a checkpoint exists for the given step."""
        ...

    async def get_pipeline_progress(self, pipeline_name: str) -> float:
        """Return aggregated progress for the pipeline."""
        ...


class KGSplitsPort(Protocol):
    """Port for accessing knowledge graph splits persistence."""

    async def load_split(
        self, split_name: str, split_type: str = "raw", map_to_ints: bool = False
    ) -> Any | None:
        """Load a single split by name and storage type."""
        ...

    async def split_exists(self, split_name: str, split_type: str = "raw") -> bool:
        """Return whether a split exists."""
        ...

    async def save_split(self, split_name: str, df: Any, split_type: str = "raw") -> None:
        """Persist a single split dataframe."""
        ...

    async def save_preprocessed_splits(
        self, train_df: Any, valid_df: Any, test_df: Any, source: str = "preprocess"
    ) -> None:
        """Persist preprocessed train/valid/test splits."""
        ...

    async def delete_preprocessed(self) -> None:
        """Delete preprocessed split artifacts."""
        ...

    async def preprocessed_exists(self) -> bool:
        """Return whether preprocessed splits exist."""
        ...

    async def load_preprocessed_splits(
        self, fallback_to_raw: bool = True, map_to_ints: bool = True
    ) -> tuple[Any | None, Any | None, Any | None, dict[str, Any]]:
        """Load preprocessed splits and associated metadata."""
        ...


class KGMappingsPort(Protocol):
    """Port for accessing knowledge graph mappings persistence."""

    async def load_mappings(
        self, mapping_type: str, use_cache: bool = True
    ) -> dict[str, int] | None:
        """Load mappings for entities or relations."""
        ...

    async def mapping_exists(self, mapping_type: str) -> bool:
        """Return whether mappings exist for the requested type."""
        ...

    async def save_mappings(self, mapping_type: str, mappings: dict[str, int]) -> None:
        """Persist in-memory mappings."""
        ...

    async def save_mappings_from_dataframe(
        self, mapping_type: str, df: Any, source: str = "preprocess"
    ) -> None:
        """Persist mappings generated from a dataframe."""
        ...

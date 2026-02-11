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
    ) -> int: ...

    async def get_checkpoint(self, pipeline_name: str, step_name: str) -> dict[str, Any] | None: ...

    async def get_pipeline_checkpoints(self, pipeline_name: str) -> list[dict[str, Any]]: ...

    async def reset_pipeline(self, pipeline_name: str) -> int: ...

    async def delete_pipeline_checkpoints(self, pipeline_name: str) -> int: ...

    async def checkpoint_exists(self, pipeline_name: str, step_name: str) -> bool: ...

    async def get_pipeline_progress(self, pipeline_name: str) -> float: ...


class KGSplitsPort(Protocol):
    """Port for accessing knowledge graph splits persistence."""

    async def load_split(
        self, split_name: str, split_type: str = "raw", map_to_ints: bool = False
    ) -> Any | None: ...

    async def split_exists(self, split_name: str, split_type: str = "raw") -> bool: ...

    async def save_split(self, split_name: str, df: Any, split_type: str = "raw") -> None: ...

    async def save_preprocessed_splits(
        self, train_df: Any, valid_df: Any, test_df: Any, source: str = "preprocess"
    ) -> None: ...

    async def delete_preprocessed(self) -> None: ...

    async def preprocessed_exists(self) -> bool: ...

    async def load_preprocessed_splits(
        self, fallback_to_raw: bool = True, map_to_ints: bool = True
    ) -> tuple[Any | None, Any | None, Any | None, dict[str, Any]]: ...


class KGMappingsPort(Protocol):
    """Port for accessing knowledge graph mappings persistence."""

    async def load_mappings(
        self, mapping_type: str, use_cache: bool = True
    ) -> dict[str, int] | None: ...

    async def mapping_exists(self, mapping_type: str) -> bool: ...

    async def save_mappings(self, mapping_type: str, mappings: dict[str, int]) -> None: ...

    async def save_mappings_from_dataframe(
        self, mapping_type: str, df: Any, source: str = "preprocess"
    ) -> None: ...

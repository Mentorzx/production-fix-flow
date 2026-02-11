"""Factory for KG pipeline components."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pff.domain.ports.persistence.kg_ports import (
        KGMappingsPort,
        KGSplitsPort,
        PipelineCheckpointsPort,
    )

from .builder import KGBuilder
from .config import KGConfig
from .pipeline import KGPipeline
from .preprocess import KGPreprocessor


class KGComponentFactory:
    """Create KG components with a single entry point."""

    def create_builder(self, config: KGConfig) -> KGBuilder:
        params = config.get_builder_config()
        return KGBuilder(
            source_path=params["source_path"],
            output_dir=config.graph_directory,
            max_members=params.get("max_members"),
            parallel=params.get("parallel", True),
            disk_cache=params.get("disk_cache", False),
            workers=params.get("workers"),
        )

    def create_preprocessor(
        self,
        config: KGConfig,
        *,
        splits_repo: KGSplitsPort | None = None,
        mappings_repo: KGMappingsPort | None = None,
    ) -> KGPreprocessor:
        return KGPreprocessor(config, splits_repo=splits_repo, mappings_repo=mappings_repo)

    def create_pipeline(
        self,
        config: KGConfig,
        checkpoints_repo: PipelineCheckpointsPort | None = None,
        splits_repo: KGSplitsPort | None = None,
    ) -> KGPipeline:
        """Create a pipeline with injected repositories."""
        return KGPipeline(
            config,
            factory=self,
            checkpoints_repo=checkpoints_repo,
            splits_repo=splits_repo,
        )

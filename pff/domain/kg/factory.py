"""Factory for KG pipeline components."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pff.domain.ports.persistence.kg_ports import PipelineCheckpointsPort, KGSplitsPort

from .builder import KGBuilder
from .preprocess import KGPreprocessor
from .pipeline import KGPipeline
from .config import KGConfig


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

    def create_preprocessor(self, config: KGConfig) -> KGPreprocessor:
        return KGPreprocessor(config)

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

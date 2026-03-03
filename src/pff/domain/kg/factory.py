"""Factory for KG pipeline components."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

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

    def create_builder(
        self,
        config: KGConfig,
        *,
        file_manager: Any | None = None,
        cache_manager: Any | None = None,
    ) -> KGBuilder:
        """Execute create builder.



        Args:

            config: Input value used by this callable.

            file_manager: Optional input value.

            cache_manager: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        params = config.get_builder_config()
        return KGBuilder(
            source_path=params["source_path"],
            output_dir=config.graph_directory,
            max_members=params.get("max_members"),
            parallel=params.get("parallel", True),
            disk_cache=params.get("disk_cache", False),
            workers=params.get("workers"),
            file_manager=file_manager,
            cache_manager=cache_manager,
        )

    def create_preprocessor(
        self,
        config: KGConfig,
        *,
        splits_repo: KGSplitsPort | None = None,
        mappings_repo: KGMappingsPort | None = None,
        save_splits_hook: Callable[[KGSplitsPort, dict[str, Any]], None] | None = None,
        save_mappings_hook: Callable[[KGMappingsPort, Any, Any], None] | None = None,
        file_manager: Any | None = None,
        cache_manager: Any | None = None,
    ) -> KGPreprocessor:
        """Execute create preprocessor.



        Args:

            config: Input value used by this callable.

            splits_repo: Optional input value.

            mappings_repo: Optional input value.

            file_manager: Optional input value.

            cache_manager: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return KGPreprocessor(
            config,
            splits_repo=splits_repo,
            mappings_repo=mappings_repo,
            save_splits_hook=save_splits_hook,
            save_mappings_hook=save_mappings_hook,
            file_manager=file_manager,
            cache_manager=cache_manager,
        )

    def create_pipeline(
        self,
        config: KGConfig,
        checkpoints_repo: PipelineCheckpointsPort | None = None,
        splits_repo: KGSplitsPort | None = None,
        save_splits_hook: Callable[[KGSplitsPort, dict[str, Any]], None] | None = None,
        save_mappings_hook: Callable[[KGMappingsPort, Any, Any], None] | None = None,
    ) -> KGPipeline:
        """Create a pipeline with injected repositories."""
        return KGPipeline(
            config,
            factory=self,
            checkpoints_repo=checkpoints_repo,
            splits_repo=splits_repo,
            save_splits_hook=save_splits_hook,
            save_mappings_hook=save_mappings_hook,
        )

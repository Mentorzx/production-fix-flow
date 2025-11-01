"""
Factory for creating and configuring feature processors.

This module implements the Factory pattern for creating appropriate
processors based on configuration and system capabilities.
"""

from __future__ import annotations

from typing import Any, Type

from .base import FeatureProcessor, ProcessingStrategy, ProcessorRegistry
from .config import ProcessorConfig
from .strategies import (
    ContextBasedStrategy,
    IndexedProcessingStrategy,
    NumbaProcessingStrategy,
    ParallelProcessingStrategy,
    SequentialProcessingStrategy,
)
from .symbolic_processor import SymbolicFeatureProcessorV2


class DefaultProcessorRegistry:
    """Default implementation of processor registry."""

    def __init__(self):
        self._strategies: dict[str, ProcessingStrategy] = {}
        self._register_default_strategies()

    def register_strategy(self, name: str, strategy: ProcessingStrategy) -> None:
        """Register a processing strategy."""
        self._strategies[name] = strategy

    def get_strategy(self, name: str) -> ProcessingStrategy | None:
        """Get a registered strategy."""
        return self._strategies.get(name)

    def list_strategies(self) -> list[str]:
        """List all registered strategies."""
        return list(self._strategies.keys())

    def _register_default_strategies(self) -> None:
        """Register default strategies."""
        self.register_strategy("context", ContextBasedStrategy())
        self.register_strategy("numba", NumbaProcessingStrategy())
        self.register_strategy("indexed", IndexedProcessingStrategy())
        self.register_strategy("parallel", ParallelProcessingStrategy())
        self.register_strategy("sequential", SequentialProcessingStrategy())


class ProcessorFactory:
    """Factory for creating feature processors with appropriate strategies."""

    def __init__(self, registry: ProcessorRegistry | None = None):
        self.registry = registry or DefaultProcessorRegistry()

    def create_processor(
        self,
        config: ProcessorConfig,
        processor_class: Type[FeatureProcessor] = SymbolicFeatureProcessorV2,
    ) -> FeatureProcessor:
        """Create a feature processor with the given configuration."""
        # Create the processor
        processor = processor_class(config)

        # Configure strategies based on capabilities and configuration
        strategies = self._select_strategies(config)

        # Configure the processor with selected strategies
        processor.set_strategies(strategies)

        return processor

    def create_optimized_processor(
        self,
        config: ProcessorConfig,
        data_size: int,
        processor_class: Type[FeatureProcessor] = SymbolicFeatureProcessorV2,
    ) -> FeatureProcessor:
        """Create an optimized processor based on data size and configuration."""
        # Adjust configuration based on data size
        optimized_config = self._optimize_config_for_data_size(config, data_size)

        return self.create_processor(optimized_config, processor_class)

    def register_strategy(self, name: str, strategy: ProcessingStrategy) -> None:
        """Register a new strategy."""
        self.registry.register_strategy(name, strategy)

    def _select_strategies(self, config: ProcessorConfig) -> list[ProcessingStrategy]:
        """Select appropriate strategies based on configuration."""
        strategies = []

        # Always try context first if available
        if config.use_context_violations:
            context_strategy = self.registry.get_strategy("context")
            if context_strategy:
                strategies.append(context_strategy)

        # Add Numba strategy if enabled
        if config.enable_numba:
            numba_strategy = self.registry.get_strategy("numba")
            if numba_strategy:
                strategies.append(numba_strategy)

        # Add indexed strategy if enabled
        if config.enable_rule_indexing:
            indexed_strategy = self.registry.get_strategy("indexed")
            if indexed_strategy:
                strategies.append(indexed_strategy)

        # Add parallel strategy for larger datasets
        parallel_strategy = self.registry.get_strategy("parallel")
        if parallel_strategy:
            strategies.append(parallel_strategy)

        # Always add sequential as fallback
        sequential_strategy = self.registry.get_strategy("sequential")
        if sequential_strategy:
            strategies.append(sequential_strategy)

        return strategies

    def _optimize_config_for_data_size(self, config: ProcessorConfig, data_size: int) -> ProcessorConfig:
        """Optimize configuration based on data size."""
        from .config import ProcessorConfigBuilder

        builder = ProcessorConfigBuilder(config.rules_path)

        # Copy existing configuration
        builder.with_confidence_threshold(config.min_confidence_threshold)
        builder.with_feature_grouping(
            config.enable_grouping,
            config.n_groups,
            config.boost_factor,
        )
        builder.with_rule_indexing(config.enable_rule_indexing)
        builder.with_numba(config.enable_numba)
        builder.with_context_violations(config.use_context_violations)

        # Optimize based on data size
        if data_size < 100:
            # Small dataset - prioritize accuracy over speed
            builder.with_performance(
                parallel_threshold=1000,  # Disable parallel for small datasets
                batch_size=100,
            )
        elif data_size < 1000:
            # Medium dataset - balance speed and accuracy
            builder.with_performance(
                parallel_threshold=200,
                batch_size=500,
            )
        else:
            # Large dataset - prioritize speed
            builder.with_performance(
                parallel_threshold=50,
                batch_size=2000,
            )

        return builder.build()


# Global factory instance
default_factory = ProcessorFactory()


def create_processor(
    rules_path: str,
    **kwargs,
) -> FeatureProcessor:
    """Convenience function to create a processor with default configuration."""
    from .config import ProcessorConfigBuilder

    config = ProcessorConfigBuilder(rules_path).build()
    return default_factory.create_processor(config, **kwargs)


def create_high_performance_processor(
    rules_path: str,
    data_size: int,
    **kwargs,
) -> FeatureProcessor:
    """Convenience function to create a high-performance processor."""
    from .config import create_high_performance_config

    config = create_high_performance_config(rules_path)
    return default_factory.create_optimized_processor(config, data_size, **kwargs)


def create_debug_processor(
    rules_path: str,
    debug_dir: str = "debug",
    **kwargs,
) -> FeatureProcessor:
    """Convenience function to create a debug processor."""
    from .config import create_debug_config

    config = create_debug_config(rules_path, debug_dir)
    return default_factory.create_processor(config, **kwargs)
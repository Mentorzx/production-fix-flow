"""Model Factory Pattern Implementation.

Provides a centralized factory for creating ML models used in the PFF pipeline.
DSLFM is the supported KGE path; legacy GBM entries have been removed.

Design Patterns Applied:
    - **Factory Pattern:** Centralized model creation with type-based dispatch.
    - **Strategy Pattern:** Uses KGEModelStrategy for KGE model variations.
    - **Dependency Injection:** Accepts external configurations and dependencies.

Note:
    DSLFM-KGC + PC2 is the primary stack.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

from torch import nn

from pff.shared import FileManager, logger

from .kge_strategy import DSLFMStrategy, KGEConfig, KGEModelStrategy


class ModelType(Enum):
    """Enumeration of supported model types."""

    DSLFM = auto()
    COMPLEX = auto()


class ModelFactory:
    """Factory for creating ML models."""

    def __init__(self, file_manager: FileManager | None = None) -> None:
        """Initialize factory with optional dependencies."""
        self.file_manager = file_manager or FileManager()
        self._strategies: dict[ModelType, type[KGEModelStrategy]] = {
            ModelType.DSLFM: DSLFMStrategy,
        }

    def create(self, model_type: ModelType, **kwargs: Any) -> Any:
        """Create a model of the specified type."""
        if model_type in self._strategies:
            return self._create_kge_model(model_type, **kwargs)
        raise ValueError(f"Unsupported model type: {model_type}")

    def _create_kge_model(
        self,
        model_type: ModelType,
        num_entities: int,
        num_relations: int,
        config: KGEConfig | None = None,
        device: Any = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Create a KGE model using the appropriate strategy."""
        strategy_class = self._strategies.get(model_type)
        if strategy_class is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        if config is None:
            config = KGEConfig(
                **{k: v for k, v in kwargs.items() if hasattr(KGEConfig, k)}
            )

        strategy = strategy_class(config)
        model = strategy.create_model(num_entities, num_relations, device)

        logger.info(f"Modelo {strategy.name} criado via Factory")
        return model

    def get_strategy(self, model_type: ModelType) -> KGEModelStrategy | None:
        """Get the strategy for a KGE model type."""
        if model_type in self._strategies:
            return self._strategies[model_type]()
        return None

    def register_strategy(
        self,
        model_type: ModelType,
        strategy_class: type[KGEModelStrategy],
    ) -> None:
        """Register a new KGE strategy."""
        self._strategies[model_type] = strategy_class
        logger.info(f"Estratégia registrada para {model_type.name}")

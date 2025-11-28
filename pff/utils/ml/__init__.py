"""ML Utilities and Design Pattern Abstractions.

This module provides design pattern implementations for ML components:
    - KGEModelStrategy: Strategy pattern for knowledge graph embedding models.
    - ModelFactory: Factory pattern for model creation.
    - BaseTrainer: Template Method pattern for training loops.

Author: PFF Team
Date: 2025-11-25
"""

from .kge_strategy import (
    KGEModelStrategy,
    TransEStrategy,
    KGEConfig,
)
from .model_factory import (
    ModelFactory,
    ModelType,
)
from .base_trainer import (
    BaseTrainer,
    TrainerConfig,
)

__all__ = [
    "KGEModelStrategy",
    "TransEStrategy",
    "KGEConfig",
    "ModelFactory",
    "ModelType",
    "BaseTrainer",
    "TrainerConfig",
]

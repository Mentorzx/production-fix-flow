"""ML Utilities and Design Pattern Abstractions.

This module provides design pattern implementations for ML components:
    - ModelFactory: Factory pattern for model creation.
    - BaseTrainer: Template Method pattern for training loops.
    - AdaptiveTrainingCalculator: Dynamic training hyperparameter computation.

Author: PFF Team
Date: 2025-11-25
"""

from .model_factory import (
    ModelFactory,
    ModelType,
)
from .base_trainer import (
    BaseTrainer,
    TrainerConfig,
)
from .kge_strategy import (
    KGEConfig,
    KGEModelStrategy,
    DSLFMStrategy,
)
from .adaptive_training import (
    AdaptiveTrainingCalculator,
    AdaptiveTrainingConfig,
    DatasetStats,
    DatasetScale,
    compute_adaptive_config,
)
from .ann_evaluator import (
    ANNEvaluator,
    ANNConfig,
    should_use_ann,
    create_ann_evaluator,
    FAISS_AVAILABLE,
)

__all__ = [
    "ModelFactory",
    "ModelType",
    "BaseTrainer",
    "TrainerConfig",
    "KGEConfig",
    "KGEModelStrategy",
    "DSLFMStrategy",
    "AdaptiveTrainingCalculator",
    "AdaptiveTrainingConfig",
    "DatasetStats",
    "DatasetScale",
    "compute_adaptive_config",
    "ANNEvaluator",
    "ANNConfig",
    "should_use_ann",
    "create_ann_evaluator",
    "FAISS_AVAILABLE",
]

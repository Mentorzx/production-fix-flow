"""Model Factory Pattern Implementation.

Provides a centralized factory for creating ML models used in the PFF pipeline.
Supports KGE models (RotatE), gradient boosted models, and ensemble meta-learners.

Design Patterns Applied:
    - **Factory Pattern:** Centralized model creation with type-based dispatch.
    - **Strategy Pattern:** Uses KGEModelStrategy for KGE model variations.
    - **Dependency Injection:** Accepts external configurations and dependencies.

Example:
    >>> from pff.utils.ml import ModelFactory, ModelType
    >>> factory = ModelFactory()
    >>> rotate = factory.create(ModelType.ROTATE, num_entities=1000, num_relations=50)
    >>> lgbm = factory.create(ModelType.LIGHTGBM, **lgbm_params)

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

import torch.nn as nn

from pff.utils import FileManager, logger
from .kge_strategy import (
    KGEConfig,
    KGEModelStrategy,
    TransEStrategy,
    RotatEStrategy,
    DSLFMStrategy,
)


class ModelType(Enum):
    """Enumeration of supported model types."""

    TRANSE = auto()  # DEPRECATED: Use ROTATE instead
    ROTATE = auto()  # Primary KGE model (SOTA)
    DSLFM = auto()  # RotatE + lógica diferenciável (lambda configurável)
    COMPLEX = auto()  # Reserved for future implementation
    LIGHTGBM = auto()
    XGBOOST = auto()
    CATBOOST = auto()


class ModelFactory:
    """Factory for creating ML models.

    Design Pattern: Factory
        - Centralizes object creation logic.
        - Decouples client code from concrete model classes.
        - Enables easy extension for new model types.

    Attributes:
        file_manager: FileManager for loading configurations.
        _strategies: Registry of KGE strategies by model type.
    """

    def __init__(self, file_manager: FileManager | None = None) -> None:
        """Initialize factory with optional dependencies.

        Args:
            file_manager: FileManager instance for config loading.
        """
        self.file_manager = file_manager or FileManager()
        self._strategies: dict[ModelType, type[KGEModelStrategy]] = {
            ModelType.TRANSE: TransEStrategy,
            ModelType.ROTATE: RotatEStrategy,
            ModelType.DSLFM: DSLFMStrategy,
        }

    def create(
        self,
        model_type: ModelType,
        **kwargs: Any,
    ) -> Any:
        """Create a model of the specified type.

        Args:
            model_type: Type of model to create.
            **kwargs: Model-specific parameters.

        Returns:
            Instantiated model.

        Raises:
            ValueError: If model type is not supported.
        """
        if model_type in self._strategies:
            return self._create_kge_model(model_type, **kwargs)
        elif model_type == ModelType.LIGHTGBM:
            return self._create_lightgbm(**kwargs)
        elif model_type == ModelType.XGBOOST:
            return self._create_xgboost(**kwargs)
        elif model_type == ModelType.CATBOOST:
            return self._create_catboost(**kwargs)
        else:
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
        """Create a KGE model using the appropriate strategy.

        Args:
            model_type: KGE model type.
            num_entities: Number of entities.
            num_relations: Number of relations.
            config: Optional KGE configuration.
            device: Target device.
            **kwargs: Additional parameters merged into config.

        Returns:
            KGE model instance.
        """
        strategy_class = self._strategies[model_type]

        if config is None:
            config = KGEConfig(**{k: v for k, v in kwargs.items() if hasattr(KGEConfig, k)})

        strategy = strategy_class(config)
        model = strategy.create_model(num_entities, num_relations, device)

        logger.info(f"Modelo {strategy.name} criado via Factory")
        return model

    def _create_lightgbm(self, **kwargs: Any) -> Any:
        """Create LightGBM classifier with GPU auto-detection (SOTA LightGBM 4.0+).

        Automatically enables GPU acceleration if CUDA is available and the
        LightGBM installation supports it. Falls back to CPU otherwise.

        SOTA Features (LightGBM 4.0+):
            - GPU acceleration via device="gpu"
            - Gradient quantization for 2-3x speedup with minimal accuracy loss

        Args:
            **kwargs: LightGBM parameters. Supports all LGBMClassifier params.

        Returns:
            LightGBM classifier with optimal device configuration.
        """
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm") from e

        # SOTA LightGBM 4.0+: GPU auto-detection
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False

        defaults = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
        }

        # SOTA: GPU acceleration if available
        if gpu_available:
            defaults["device"] = "gpu"
            defaults["gpu_platform_id"] = 0
            defaults["gpu_device_id"] = 0
            # gpu_use_dp=True for deterministic GPU results; max_bin=63 is GPU-optimal histogram binning
            defaults.setdefault("gpu_use_dp", True)
            defaults.setdefault("max_bin", 63)
        else:
            defaults["device"] = "cpu"

        # SOTA LightGBM 4.0+: Gradient quantization for faster training
        # 2-3x speedup with minimal accuracy loss
        defaults["use_quantized_grad"] = True
        defaults["num_grad_quant_bins"] = 8

        defaults.update(kwargs)

        model = lgb.LGBMClassifier(**defaults)
        device_used = defaults.get("device", "cpu")
        quant_status = "quantized" if defaults.get("use_quantized_grad") else "standard"
        logger.debug(f"LightGBM created: device={device_used}, gradient={quant_status}")
        return model

    def _create_xgboost(self, **kwargs: Any) -> Any:
        """Create XGBoost classifier with GPU auto-detection (SOTA XGBoost 3.0+).

        Args:
            **kwargs: XGBoost parameters.

        Returns:
            XGBoost classifier with device set to "cuda" if GPU available.
        """
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("XGBoost not installed. Run: pip install xgboost") from e

        # SOTA XGBoost 3.0+: GPU auto-detection via device="cuda"
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False

        defaults = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "use_label_encoder": False,
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbosity": 0,
            "tree_method": "hist",  # Required for GPU acceleration
            "device": "cuda" if gpu_available else "cpu",  # SOTA XGBoost 3.0+
        }
        defaults.update(kwargs)

        model = XGBClassifier(**defaults)
        device_used = defaults.get("device", "cpu")
        logger.debug(f"XGBoost created: device={device_used}")
        return model

    def _create_catboost(self, **kwargs: Any) -> Any:
        """Create CatBoost classifier.

        Args:
            **kwargs: CatBoost parameters.

        Returns:
            CatBoost classifier.
        """
        try:
            from catboost import CatBoostClassifier
        except ImportError as e:
            raise ImportError("CatBoost not installed. Run: pip install catboost") from e

        defaults = {
            "iterations": 100,
            "learning_rate": 0.1,
            "depth": 6,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": 42,
            "verbose": False,
        }
        defaults.update(kwargs)

        model = CatBoostClassifier(**defaults)
        logger.info("CatBoost criado via Factory")
        return model

    def get_strategy(self, model_type: ModelType) -> KGEModelStrategy | None:
        """Get the strategy for a KGE model type.

        Args:
            model_type: The model type.

        Returns:
            Strategy instance or None if not a KGE type.
        """
        if model_type in self._strategies:
            return self._strategies[model_type]()
        return None

    def register_strategy(
        self,
        model_type: ModelType,
        strategy_class: type[KGEModelStrategy],
    ) -> None:
        """Register a new KGE strategy.

        Args:
            model_type: Model type to register.
            strategy_class: Strategy class implementing KGEModelStrategy.
        """
        self._strategies[model_type] = strategy_class
        logger.info(f"Strategy registrada para {model_type.name}")

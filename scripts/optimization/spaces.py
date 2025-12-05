#!/usr/bin/env python3
"""
Search Spaces Module

Defines configuration classes and factory for creating search spaces
for different optimization targets.

Design Patterns:
- Factory Method: Creates different search space configurations
- Builder Pattern: Fluent configuration of TuningConfig
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pff.utils import logger
from pff.utils.core.file_manager import FileManager


def _load_hpo_defaults() -> dict[str, Any]:
    """Load HPO defaults from config/hpo/optimization.yaml."""
    try:
        fm = FileManager()
        config_path = Path("config/hpo/optimization.yaml")
        if config_path.exists():
            cfg = fm.read(config_path)
            return cfg.get("defaults", {})
    except Exception:
        pass
    return {}


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""

    # Optimization targets
    target_f1_score: float = 0.75
    target_violation_range: tuple[float, float] = (50.0, 150.0)
    target_symbolic_ratio: float = 0.70  # 70% symbolic, 30% hybrid

    # Search space bounds
    min_confidence_range: tuple[float, float] = (0.01, 0.20)
    max_violation_range: tuple[float, float] = (50.0, 300.0)

    # XGBoost hyperparameters
    xgb_max_depth_range: tuple[int, int] = (2, 6)
    xgb_learning_rate_range: tuple[float, float] = (0.01, 0.3)
    xgb_n_estimators_range: tuple[int, int] = (50, 300)
    xgb_subsample_range: tuple[float, float] = (0.6, 1.0)
    xgb_colsample_bytree_range: tuple[float, float] = (0.3, 0.8)

    # Optimization settings
    n_trials: int = 100
    cv_folds: int = 5
    random_state: int = 42
    timeout_seconds: int = 1800  # 30 minutes max
    n_jobs: int = -1  # Use all CPUs

    # Strategy selection
    optimization_strategy: str = "tpe"  # tpe, cmaes, hyperband, grid
    enable_pruning: bool = True
    enable_distributed: bool = False  # Use Ray Tune if available


class TuningConfigBuilder:
    """
    Builder Pattern for fluent TuningConfig construction.

    Allows step-by-step configuration with sensible defaults
    loaded from config/hpo/optimization.yaml.

    Example:
        config = (TuningConfigBuilder()
            .with_trials(200)
            .with_strategy("hyperband")
            .with_xgb_depth_range(3, 8)
            .build())
    """

    def __init__(self):
        """Initialize builder with defaults from config."""
        defaults = _load_hpo_defaults()
        self._config = {
            "n_trials": defaults.get("n_trials", 100),
            "cv_folds": defaults.get("cv_folds", 5),
            "random_state": 42,
            "timeout_seconds": defaults.get("timeout_seconds", 7200),
            "n_jobs": -1,
            "optimization_strategy": "tpe",
            "enable_pruning": defaults.get("enable_pruning", True),
            "enable_distributed": False,
            "target_f1_score": 0.75,
            "target_violation_range": (50.0, 150.0),
            "target_symbolic_ratio": 0.70,
            "min_confidence_range": (0.01, 0.20),
            "max_violation_range": (50.0, 300.0),
            "xgb_max_depth_range": (2, 6),
            "xgb_learning_rate_range": (0.01, 0.3),
            "xgb_n_estimators_range": (50, 300),
            "xgb_subsample_range": (0.6, 1.0),
            "xgb_colsample_bytree_range": (0.3, 0.8),
        }

    def with_trials(self, n_trials: int) -> "TuningConfigBuilder":
        """Set number of trials."""
        self._config["n_trials"] = n_trials
        return self

    def with_cv_folds(self, cv_folds: int) -> "TuningConfigBuilder":
        """Set number of cross-validation folds."""
        self._config["cv_folds"] = cv_folds
        return self

    def with_random_state(self, seed: int) -> "TuningConfigBuilder":
        """Set random state for reproducibility."""
        self._config["random_state"] = seed
        return self

    def with_timeout(self, seconds: int) -> "TuningConfigBuilder":
        """Set timeout in seconds."""
        self._config["timeout_seconds"] = seconds
        return self

    def with_strategy(self, strategy: str) -> "TuningConfigBuilder":
        """Set optimization strategy (tpe, cmaes, hyperband)."""
        self._config["optimization_strategy"] = strategy
        return self

    def with_pruning(self, enable: bool) -> "TuningConfigBuilder":
        """Enable or disable pruning."""
        self._config["enable_pruning"] = enable
        return self

    def with_distributed(self, enable: bool) -> "TuningConfigBuilder":
        """Enable or disable distributed optimization."""
        self._config["enable_distributed"] = enable
        return self

    def with_target_f1(self, target: float) -> "TuningConfigBuilder":
        """Set target F1 score."""
        self._config["target_f1_score"] = target
        return self

    def with_xgb_depth_range(self, min_depth: int, max_depth: int) -> "TuningConfigBuilder":
        """Set XGBoost max_depth range."""
        self._config["xgb_max_depth_range"] = (min_depth, max_depth)
        return self

    def with_xgb_lr_range(self, min_lr: float, max_lr: float) -> "TuningConfigBuilder":
        """Set XGBoost learning_rate range."""
        self._config["xgb_learning_rate_range"] = (min_lr, max_lr)
        return self

    def with_confidence_range(self, min_conf: float, max_conf: float) -> "TuningConfigBuilder":
        """Set min_confidence_threshold range."""
        self._config["min_confidence_range"] = (min_conf, max_conf)
        return self

    def build(self) -> TuningConfig:
        """Build TuningConfig from collected settings."""
        return TuningConfig(**self._config)


class SearchSpaceFactory:
    """
    Factory for creating search spaces for different optimization targets.

    This class provides a centralized way to define and configure
    search spaces for various machine learning models and components.
    """

    @staticmethod
    def create_ensemble_space(config: TuningConfig) -> dict[str, Any]:
        """
        Create search space for ensemble optimization.

        Covers ALL critical hyperparameters from:
        - config/models/ensemble.yaml (symbolic + XGBoost)
        - config/models/kg.yaml (AnyBURL)
        - config/models/transe.yaml (TransE + LightGBM)

        Args:
            config: Tuning configuration

        Returns:
            Dictionary defining the search space
        """
        return {
            # === Ensemble: Symbolic Features ===
            'min_confidence_threshold': (
                config.min_confidence_range[0],
                config.min_confidence_range[1],
            ),
            'max_violation_percentage': (
                config.max_violation_range[0],
                config.max_violation_range[1],
            ),

            # === Ensemble: XGBoost Meta-Learner ===
            'xgb_max_depth': config.xgb_max_depth_range,
            'xgb_learning_rate': config.xgb_learning_rate_range,
            'xgb_n_estimators': config.xgb_n_estimators_range,
            'xgb_subsample': config.xgb_subsample_range,
            'xgb_colsample_bytree': config.xgb_colsample_bytree_range,
            'xgb_reg_alpha': (0.01, 1.0),
            'xgb_reg_lambda': (0.1, 10.0),
            'xgb_min_child_weight': (1, 10),
            'xgb_gamma': (0.0, 0.5),

            # === KG: AnyBURL Rule Learning ===
            'anyburl_threshold_confidence': (0.01, 0.05),
            'anyburl_max_length_acyclic': (1, 3),
            'anyburl_max_length_cyclic': (2, 4),
            'anyburl_sample_size': (300, 1000),

            # === TransE: Embedding Model ===
            'transe_embedding_dim': (64, 256),
            'transe_learning_rate': (0.0001, 0.01),
            'transe_margin': (0.5, 2.0),
            'transe_batch_size': [64, 128, 256],
            'transe_weight_decay': (0.001, 0.1),

            # === TransE: LightGBM Hybrid ===
            'lgbm_num_leaves': (3, 15),
            'lgbm_max_depth': (2, 5),
            'lgbm_learning_rate': (0.0001, 0.01),
            'lgbm_feature_fraction': (0.2, 0.5),
            'lgbm_lambda_l1': (1.0, 20.0),
            'lgbm_lambda_l2': (1.0, 20.0),
        }

    @staticmethod
    def create_symbolic_space(config: TuningConfig) -> dict[str, Any]:
        """
        Create search space for symbolic feature extraction.

        Focuses on parameters related to rule-based features:
        - Confidence thresholds
        - Violation percentages
        - Rule filtering

        Args:
            config: Tuning configuration

        Returns:
            Dictionary defining the symbolic search space
        """
        return {
            'min_confidence_threshold': (
                config.min_confidence_range[0],
                config.min_confidence_range[1],
            ),
            'max_violation_percentage': (
                config.max_violation_range[0],
                config.max_violation_range[1],
            ),
            'symbolic_ratio_target': (0.5, 0.9),
            'enable_rule_grouping': [True, False],
            'n_rule_groups': (10, 100),
            'boost_factor': (0.5, 2.0),
        }

    @staticmethod
    def create_xgboost_space(config: TuningConfig) -> dict[str, Any]:
        """
        Create search space for XGBoost optimization.

        Focuses exclusively on XGBoost hyperparameters.

        Args:
            config: Tuning configuration

        Returns:
            Dictionary defining the XGBoost search space
        """
        return {
            'xgb_max_depth': config.xgb_max_depth_range,
            'xgb_learning_rate': config.xgb_learning_rate_range,
            'xgb_n_estimators': config.xgb_n_estimators_range,
            'xgb_subsample': config.xgb_subsample_range,
            'xgb_colsample_bytree': config.xgb_colsample_bytree_range,
            'xgb_reg_alpha': (0.01, 1.0),
            'xgb_reg_lambda': (0.1, 10.0),
            'xgb_min_child_weight': (1, 10),
            'xgb_gamma': (0.0, 0.5),
        }

    @staticmethod
    def create_anyburl_space(config: TuningConfig) -> dict[str, Any]:
        """
        Create search space for AnyBURL optimization.

        Focuses on AnyBURL rule learning parameters.

        Args:
            config: Tuning configuration

        Returns:
            Dictionary defining the AnyBURL search space
        """
        return {
            'anyburl_threshold_confidence': (0.01, 0.05),
            'anyburl_max_length_acyclic': (1, 3),
            'anyburl_max_length_cyclic': (2, 4),
            'anyburl_sample_size': (300, 1000),
            'anyburl_worker_threads': (1, 20),
            'anyburl_java_heap': ('2G', '32G'),
        }

    @staticmethod
    def create_transe_space(config: TuningConfig) -> dict[str, Any]:
        """
        Create search space for TransE optimization.

        Focuses on TransE embedding model parameters.

        Args:
            config: Tuning configuration

        Returns:
            Dictionary defining the TransE search space
        """
        return {
            'transe_embedding_dim': (64, 256),
            'transe_learning_rate': (0.0001, 0.01),
            'transe_margin': (0.5, 2.0),
            'transe_batch_size': [64, 128, 256],
            'transe_weight_decay': (0.001, 0.1),
            'transe_epochs': (50, 200),
            'transe_norm': [1, 2],
        }

    @staticmethod
    def create_rotate_space(config: TuningConfig) -> dict[str, Any]:
        """
        Create search space for RotatE optimization.

        RotatE uses complex embeddings with rotational transformations.
        Key differences from TransE:
        - gamma (margin) is typically higher (9-24 vs 0.5-2)
        - Uses self-adversarial negative sampling
        - embedding_dim must be even (complex representation)

        Reference: Sun et al. 2019 "RotatE: Knowledge Graph Embedding
        by Relational Rotation in Complex Space" (ICLR 2019)

        Args:
            config: Tuning configuration

        Returns:
            Dictionary defining the RotatE search space
        """
        return {
            # Embedding dimension (must be even for complex repr)
            'rotate_embedding_dim': [128, 256, 512],

            # Gamma (fixed margin) - RotatE uses higher values
            'rotate_gamma': (6.0, 24.0),

            # Epsilon for modular regularization
            'rotate_epsilon': (1.0, 3.0),

            # Learning rate - RotatE benefits from lower LR
            'rotate_learning_rate': (0.00001, 0.0005),

            # Batch size
            'rotate_batch_size': [512, 1024],

            # Negative samples - more samples improve self-adversarial
            'rotate_negative_samples': (256, 1024),

            # Self-adversarial temperature
            'rotate_adversarial_temperature': (0.5, 2.0),

            # Training epochs
            'rotate_epochs': (120, 200),

            # Regularization
            'rotate_entity_reg_weight': (0.0, 0.0001),
            'rotate_relation_reg_weight': (0.0, 0.0001),
        }

    @staticmethod
    def create_lightgbm_space(config: TuningConfig) -> dict[str, Any]:
        """
        Create search space for LightGBM optimization.

        Focuses on LightGBM parameters for hybrid models.

        Args:
            config: Tuning configuration

        Returns:
            Dictionary defining the LightGBM search space
        """
        return {
            'lgbm_num_leaves': (3, 15),
            'lgbm_max_depth': (2, 5),
            'lgbm_learning_rate': (0.0001, 0.01),
            'lgbm_feature_fraction': (0.2, 0.5),
            'lgbm_lambda_l1': (1.0, 20.0),
            'lgbm_lambda_l2': (1.0, 20.0),
            'lgbm_min_data_in_leaf': (5, 50),
            'lgbm_bagging_fraction': (0.6, 1.0),
            'lgbm_bagging_freq': (0, 10),
        }

    @staticmethod
    def get_space_bounds(space: dict[str, Any]) -> dict[str, tuple]:
        """
        Extract parameter bounds from search space.

        Args:
            space: Search space dictionary

        Returns:
            Dictionary with parameter bounds
        """
        bounds = {}
        for param, value in space.items():
            if isinstance(value, (list, tuple)):
                if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
                    bounds[param] = tuple(value)
                elif len(value) > 2:
                    bounds[param] = value  # Categorical
                else:
                    bounds[param] = value
            else:
                bounds[param] = value
        return bounds

    @staticmethod
    def validate_space(space: dict[str, Any]) -> bool:
        """
        Validate search space configuration.

        Args:
            space: Search space to validate

        Returns:
            True if valid, False otherwise
        """
        required_types = (list, tuple)

        for param, value in space.items():
            # Check if value is a valid range or categorical
            if not isinstance(value, required_types):
                logger.warning(f"Parameter {param} has invalid type: {type(value)}")
                return False

            # Check if range is valid (min < max for numeric)
            if len(value) == 2:
                try:
                    min_val, max_val = float(value[0]), float(value[1])
                    if min_val >= max_val:
                        logger.warning(f"Parameter {param} has invalid range: {value}")
                        return False
                except (ValueError, TypeError):
                    # Categorical parameter, that's OK
                    pass

        return True

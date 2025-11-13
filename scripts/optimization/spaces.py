#!/usr/bin/env python3
"""
Search Spaces Module

Defines configuration classes and factory for creating search spaces
for different optimization targets.

Design Patterns:
- Factory Method: Creates different search space configurations
- Builder Pattern: Configures complex parameter spaces
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""

    # Optimization targets
    target_f1_score: float = 0.75
    target_violation_range: Tuple[float, float] = (50.0, 150.0)
    target_symbolic_ratio: float = 0.70  # 70% symbolic, 30% hybrid

    # Search space bounds
    min_confidence_range: Tuple[float, float] = (0.01, 0.20)
    max_violation_range: Tuple[float, float] = (50.0, 300.0)

    # XGBoost hyperparameters
    xgb_max_depth_range: Tuple[int, int] = (2, 6)
    xgb_learning_rate_range: Tuple[float, float] = (0.01, 0.3)
    xgb_n_estimators_range: Tuple[int, int] = (50, 300)
    xgb_subsample_range: Tuple[float, float] = (0.6, 1.0)
    xgb_colsample_bytree_range: Tuple[float, float] = (0.3, 0.8)

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


class SearchSpaceFactory:
    """
    Factory for creating search spaces for different optimization targets.

    This class provides a centralized way to define and configure
    search spaces for various machine learning models and components.
    """

    @staticmethod
    def create_ensemble_space(config: TuningConfig) -> Dict[str, Any]:
        """
        Create search space for ensemble optimization.

        Covers ALL critical hyperparameters from:
        - config/ensemble.yaml (symbolic + XGBoost)
        - config/kg.yaml (AnyBURL)
        - config/transe.yaml (TransE + LightGBM)

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
    def create_symbolic_space(config: TuningConfig) -> Dict[str, Any]:
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
    def create_xgboost_space(config: TuningConfig) -> Dict[str, Any]:
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
    def create_anyburl_space(config: TuningConfig) -> Dict[str, Any]:
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
    def create_transe_space(config: TuningConfig) -> Dict[str, Any]:
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
    def create_lightgbm_space(config: TuningConfig) -> Dict[str, Any]:
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
    def get_space_bounds(space: Dict[str, Any]) -> Dict[str, Tuple]:
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
    def validate_space(space: Dict[str, Any]) -> bool:
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

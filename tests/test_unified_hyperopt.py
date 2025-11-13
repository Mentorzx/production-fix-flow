"""
Tests for Unified Hyperparameter Optimization System.

Tests unified hyperparameter optimization with:
- Multiple optimization strategies
- Multi-objective optimization
- Ensemble and TransE optimization
- Parameter importance analysis
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from scripts.unified_hyperopt import (
    TPEStrategy,
    CMAESStrategy,
    BOHBStrategy,
    TransEStrategy,
    UnifiedHyperoptimizer,
    HyperparameterTuner,
    OptimizerFactory,
    MultiObjectiveConfig,
)


class TestOptimizationStrategies:
    """Test optimization strategies."""

    def test_tpe_strategy(self):
        """Test TPE strategy creation."""
        strategy = TPEStrategy()
        sampler = strategy.create_sampler()
        pruner = strategy.create_pruner()
        params = strategy.suggest_params(MagicMock())

        assert sampler is not None
        assert pruner is not None
        assert "min_confidence_threshold" in params
        assert "xgb_n_estimators" in params

    def test_cmaes_strategy(self):
        """Test CMA-ES strategy creation."""
        strategy = CMAESStrategy()
        sampler = strategy.create_sampler()
        pruner = strategy.create_pruner()
        params = strategy.suggest_params(MagicMock())

        assert sampler is not None
        assert pruner is not None
        assert "min_confidence_threshold" in params

    def test_bohb_strategy(self):
        """Test BOHB strategy creation."""
        strategy = BOHBStrategy()
        sampler = strategy.create_sampler()
        pruner = strategy.create_pruner()
        params = strategy.suggest_params(MagicMock())

        assert sampler is not None
        assert pruner is not None
        assert "min_confidence_threshold" in params

    def test_transe_strategy(self):
        """Test TransE strategy creation."""
        strategy = TransEStrategy()
        sampler = strategy.create_sampler()
        pruner = strategy.create_pruner()
        params = strategy.suggest_params(MagicMock())

        assert sampler is not None
        assert pruner is not None
        assert "embedding_dim" in params
        assert "margin" in params


class TestUnifiedHyperoptimizer:
    """Test unified hyperparameter optimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = UnifiedHyperoptimizer()
        assert optimizer is not None
        assert optimizer.strategy is not None
        assert optimizer.output_dir is not None

    def test_initialization_with_strategy(self):
        """Test optimizer initialization with custom strategy."""
        strategy = CMAESStrategy()
        optimizer = UnifiedHyperoptimizer(strategy=strategy)
        assert optimizer.strategy == strategy

    def test_initialization_multi_objective(self):
        """Test optimizer initialization with multi-objective."""
        config = MultiObjectiveConfig()
        optimizer = UnifiedHyperoptimizer(
            multi_objective=True,
            multi_obj_config=config
        )
        assert optimizer.multi_objective is True
        assert optimizer.multi_obj_config == config

    def test_optimizer_factory(self):
        """Test optimizer factory."""
        optimizer = OptimizerFactory.create_optimizer("tpe")
        assert isinstance(optimizer, UnifiedHyperoptimizer)
        assert isinstance(optimizer.strategy, TPEStrategy)

    def test_optimizer_factory_bohb(self):
        """Test optimizer factory with BOHB."""
        optimizer = OptimizerFactory.create_optimizer("bohb")
        assert isinstance(optimizer, UnifiedHyperoptimizer)
        assert isinstance(optimizer.strategy, BOHBStrategy)

    def test_optimizer_factory_transe(self):
        """Test optimizer factory with TransE strategy."""
        optimizer = OptimizerFactory.create_optimizer("transe")
        assert isinstance(optimizer, UnifiedHyperoptimizer)
        assert isinstance(optimizer.strategy, TransEStrategy)

    def test_optimizer_factory_multi_objective(self):
        """Test optimizer factory with multi-objective."""
        optimizer = OptimizerFactory.create_optimizer(
            "tpe",
            multi_objective=True
        )
        assert optimizer.multi_objective is True

    def test_invalid_strategy_fallback(self):
        """Test factory fallback for invalid strategy."""
        optimizer = OptimizerFactory.create_optimizer("invalid_strategy")
        assert isinstance(optimizer.strategy, TPEStrategy)


class TestHyperparameterTuner:
    """Test legacy hyperparameter tuner (backward compatibility)."""

    def test_initialization(self):
        """Test tuner initialization."""
        tuner = HyperparameterTuner(
            neural_model_path="path/to/neural",
            rules_path="path/to/rules",
            lightgbm_model_path="path/to/lightgbm",
        )
        assert tuner is not None
        assert tuner.neural_model_path == "path/to/neural"
        assert tuner.rules_path == "path/to/rules"
        assert tuner.lightgbm_model_path == "path/to/lightgbm"

    def test_ensemble_pipeline_creation(self):
        """Test ensemble pipeline creation."""
        tuner = HyperparameterTuner(
            neural_model_path="path/to/neural",
            rules_path="path/to/rules",
            lightgbm_model_path="path/to/lightgbm",
        )
        pipeline = tuner.create_ensemble_pipeline()
        assert pipeline is None

    def test_grid_search(self):
        """Test grid search optimization."""
        tuner = HyperparameterTuner(
            neural_model_path="path/to/neural",
            rules_path="path/to/rules",
            lightgbm_model_path="path/to/lightgbm",
        )
        param_grid = {"param1": [1, 2, 3]}
        results = tuner.grid_search(param_grid)
        assert "best_params" in results
        assert "best_score" in results

    def test_random_search(self):
        """Test random search optimization."""
        tuner = HyperparameterTuner(
            neural_model_path="path/to/neural",
            rules_path="path/to/rules",
            lightgbm_model_path="path/to/lightgbm",
        )
        param_distributions = {"param1": [1, 2, 3]}
        results = tuner.random_search(param_distributions, n_iter=10)
        assert "best_params" in results
        assert "best_score" in results

    def test_optuna_optimize(self):
        """Test Optuna optimization."""
        tuner = HyperparameterTuner(
            neural_model_path="path/to/neural",
            rules_path="path/to/rules",
            lightgbm_model_path="path/to/lightgbm",
        )
        results = tuner.optuna_optimize(n_trials=10)
        assert "best_params" in results
        assert "best_value" in results


class TestMultiObjectiveConfig:
    """Test multi-objective configuration."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = MultiObjectiveConfig()
        assert config.objectives == ['f1', 'roc_auc', 'precision']
        assert config.weights == [0.5, 0.3, 0.2]
        assert config.direction == ['maximize', 'maximize', 'maximize']
        assert config.enable_pareto_front is True
        assert config.save_pareto_solutions is True

    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        config = MultiObjectiveConfig(
            objectives=['f1', 'precision'],
            weights=[0.7, 0.3],
            direction=['maximize', 'minimize']
        )
        assert config.objectives == ['f1', 'precision']
        assert config.weights == [0.7, 0.3]
        assert config.direction == ['maximize', 'minimize']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

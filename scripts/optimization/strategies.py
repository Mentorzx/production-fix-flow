#!/usr/bin/env python3
"""
Optimization Strategies Module

Implements Strategy Pattern for different optimization algorithms.

Design Patterns:
- Strategy Pattern: Encapsulates different optimization algorithms
- Factory Method: Creates strategy instances
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pff.utils import logger

# Optuna imports
try:
    import warnings
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    # Suppress ExperimentalWarning for WilcoxonPruner - we know it's experimental
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
        from optuna.pruners import WilcoxonPruner
    OPTUNA_AVAILABLE = True
    WILCOXON_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    WILCOXON_AVAILABLE = False
    logger.warning("Optuna not available")
except AttributeError:
    # WilcoxonPruner requires Optuna >= 3.6.0
    WILCOXON_AVAILABLE = False
    logger.warning("WilcoxonPruner requires Optuna >= 3.6.0")


class OptimizationStrategy(ABC):
    """
    Abstract base class for optimization strategies.

    Strategy Pattern: Defines the interface for all optimization strategies.
    """

    def __init__(self, config):
        """
        Initialize strategy with configuration.

        Args:
            config: Tuning configuration
        """
        self.config = config

    @abstractmethod
    def create_sampler(self) -> Any:
        """Create optimizer sampler."""
        pass

    @abstractmethod
    def create_pruner(self) -> Any]:
        """Create pruner for early stopping."""
        pass

    @abstractmethod
    def suggest_params(self, trial: Any) -> dict[str, Any]:
        """Suggest hyperparameters for trial."""
        pass

    def create_study(self, study_name: str = None) -> optuna.Study:
        """
        Create Optuna study with this strategy's configuration.

        Args:
            study_name: Optional name for the study

        Returns:
            Configured Optuna study
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for optimization")

        sampler = self.create_sampler()
        pruner = self.create_pruner()

        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
        )

        logger.info(f"Estudo criado com estratégia {self.__class__.__name__}")
        return study


class TPEStrategy(OptimizationStrategy):
    """
    TPE (Tree-structured Parzen Estimator) strategy.

    Best for general use. Uses Bayesian optimization with
    tree-structured Parzen estimators.

    Strengths:
    - Good for mixed discrete/continuous parameters
    - Handles categorical parameters well
    - Considers parameter interactions
    """

    def create_sampler(self) -> TPESampler:
        """Create TPE sampler."""
        return TPESampler(
            seed=self.config.random_state,
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True,  # Consider parameter interactions
        )

    def create_pruner(self) -> MedianPruner]:
        """Create median pruner."""
        if not self.config.enable_pruning:
            return None
        return MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1,
        )

    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """
        Suggest hyperparameters using TPE.

        Covers ALL critical hyperparameters from:
        - config/models/ensemble.yaml (symbolic + XGBoost)
        - config/models/kg.yaml (AnyBURL)
        - config/models/transe.yaml (TransE + LightGBM)
        """
        return {
            # === Ensemble: Symbolic Features ===
            'min_confidence_threshold': trial.suggest_float(
                'min_confidence_threshold',
                self.config.min_confidence_range[0],
                self.config.min_confidence_range[1],
                log=True,
            ),
            'max_violation_percentage': trial.suggest_float(
                'max_violation_percentage',
                self.config.max_violation_range[0],
                self.config.max_violation_range[1],
            ),

            # === Ensemble: XGBoost Meta-Learner ===
            'xgb_max_depth': trial.suggest_int(
                'xgb_max_depth',
                self.config.xgb_max_depth_range[0],
                self.config.xgb_max_depth_range[1],
            ),
            'xgb_learning_rate': trial.suggest_float(
                'xgb_learning_rate',
                self.config.xgb_learning_rate_range[0],
                self.config.xgb_learning_rate_range[1],
                log=True,
            ),
            'xgb_n_estimators': trial.suggest_int(
                'xgb_n_estimators',
                self.config.xgb_n_estimators_range[0],
                self.config.xgb_n_estimators_range[1],
                step=50,
            ),
            'xgb_subsample': trial.suggest_float(
                'xgb_subsample',
                self.config.xgb_subsample_range[0],
                self.config.xgb_subsample_range[1],
            ),
            'xgb_colsample_bytree': trial.suggest_float(
                'xgb_colsample_bytree',
                self.config.xgb_colsample_bytree_range[0],
                self.config.xgb_colsample_bytree_range[1],
            ),
            'xgb_reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.01, 1.0, log=True),
            'xgb_reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.1, 10.0, log=True),
            'xgb_min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
            'xgb_gamma': trial.suggest_float('xgb_gamma', 0.0, 0.5),

            # === KG: AnyBURL Rule Learning ===
            'anyburl_threshold_confidence': trial.suggest_float(
                'anyburl_threshold_confidence', 0.01, 0.05, log=True
            ),
            'anyburl_max_length_acyclic': trial.suggest_int(
                'anyburl_max_length_acyclic', 1, 3
            ),
            'anyburl_max_length_cyclic': trial.suggest_int(
                'anyburl_max_length_cyclic', 2, 4
            ),
            'anyburl_sample_size': trial.suggest_int(
                'anyburl_sample_size', 300, 1000, step=100
            ),

            # === TransE: Embedding Model ===
            'transe_embedding_dim': trial.suggest_int(
                'transe_embedding_dim', 64, 256, step=16
            ),
            'transe_learning_rate': trial.suggest_float(
                'transe_learning_rate', 0.0001, 0.01, log=True
            ),
            'transe_margin': trial.suggest_float('transe_margin', 0.5, 2.0),
            'transe_batch_size': trial.suggest_categorical(
                'transe_batch_size', [64, 128, 256]
            ),
            'transe_weight_decay': trial.suggest_float(
                'transe_weight_decay', 0.001, 0.1, log=True
            ),

            # === TransE: LightGBM Hybrid ===
            'lgbm_num_leaves': trial.suggest_int('lgbm_num_leaves', 3, 15),
            'lgbm_max_depth': trial.suggest_int('lgbm_max_depth', 2, 5),
            'lgbm_learning_rate': trial.suggest_float(
                'lgbm_learning_rate', 0.0001, 0.01, log=True
            ),
            'lgbm_feature_fraction': trial.suggest_float(
                'lgbm_feature_fraction', 0.2, 0.5
            ),
            'lgbm_lambda_l1': trial.suggest_float('lgbm_lambda_l1', 1.0, 20.0),
            'lgbm_lambda_l2': trial.suggest_float('lgbm_lambda_l2', 1.0, 20.0),
        }


class CMAESStrategy(OptimizationStrategy):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) strategy.

    Best for continuous optimization. Evolutionary algorithm that
    adapts covariance matrix of the search distribution.

    Strengths:
    - Excellent for continuous parameters
    - Robust to local minima
    - Good convergence properties
    """

    def create_sampler(self) -> CmaEsSampler:
        """Create CMA-ES sampler."""
        return CmaEsSampler(
            seed=self.config.random_state,
            n_startup_trials=10,
        )

    def create_pruner(self) -> MedianPruner]:
        """CMA-ES works better without aggressive pruning."""
        if not self.config.enable_pruning:
            return None
        return MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest hyperparameters using CMA-ES."""
        # CMA-ES requires continuous parameters
        return {
            'min_confidence_threshold': trial.suggest_float(
                'min_confidence_threshold',
                self.config.min_confidence_range[0],
                self.config.min_confidence_range[1],
            ),
            'max_violation_percentage': trial.suggest_float(
                'max_violation_percentage',
                self.config.max_violation_range[0],
                self.config.max_violation_range[1],
            ),
            'xgb_max_depth': float(trial.suggest_int(
                'xgb_max_depth',
                self.config.xgb_max_depth_range[0],
                self.config.xgb_max_depth_range[1],
            )),
            'xgb_learning_rate': trial.suggest_float(
                'xgb_learning_rate',
                self.config.xgb_learning_rate_range[0],
                self.config.xgb_learning_rate_range[1],
            ),
            'xgb_n_estimators': float(trial.suggest_int(
                'xgb_n_estimators',
                self.config.xgb_n_estimators_range[0],
                self.config.xgb_n_estimators_range[1],
            )),
            'xgb_subsample': trial.suggest_float(
                'xgb_subsample',
                self.config.xgb_subsample_range[0],
                self.config.xgb_subsample_range[1],
            ),
            'xgb_colsample_bytree': trial.suggest_float(
                'xgb_colsample_bytree',
                self.config.xgb_colsample_bytree_range[0],
                self.config.xgb_colsample_bytree_range[1],
            ),
        }


class HyperbandStrategy(OptimizationStrategy):
    """
    Hyperband strategy.

    Best for large search spaces with early stopping.
    Uses successive halving to allocate resources efficiently.

    Strengths:
    - Very fast for large spaces
    - Automatic early stopping
    - Resource-efficient
    """

    def create_sampler(self) -> TPESampler:
        """Hyperband uses TPE for sampling."""
        return TPESampler(seed=self.config.random_state)

    def create_pruner(self) -> HyperbandPruner:
        """Create Hyperband pruner (aggressive early stopping)."""
        return HyperbandPruner(
            min_resource=1,
            max_resource=self.config.cv_folds,
            reduction_factor=3,
        )

    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest hyperparameters for Hyperband."""
        return TPEStrategy(self.config).suggest_params(trial)


class WilcoxonCVStrategy(OptimizationStrategy):
    """
    Wilcoxon Pruner Strategy for k-fold Cross-Validation (SOTA).

    Best for cross-validation scenarios where mean performance over
    multiple folds is the optimization target. Uses Wilcoxon signed-rank
    test to statistically compare trials.

    SOTA Features:
    - Uses Wilcoxon signed-rank test for statistical comparison
    - Recommended for k-fold CV optimization (Optuna v3.6.0+)
    - Ideal when optimizing mean accuracy over multiple problem instances
    - More statistically rigorous than MedianPruner for CV scenarios

    Reference:
    - Optuna documentation: pruners.WilcoxonPruner
    - Use case: mean performance optimization over problem instances

    Strengths:
    - Statistically robust pruning decisions
    - Handles variance across CV folds properly
    - Better for neural-symbolic ensemble training with k-fold validation
    """

    def __init__(self, config, p_threshold: float = 0.1, n_startup_steps: int = 2):
        """
        Initialize Wilcoxon CV strategy.

        Args:
            config: Tuning configuration
            p_threshold: P-value threshold for pruning decision (default 0.1)
            n_startup_steps: Minimum steps before pruning (default 2)
        """
        super().__init__(config)
        self.p_threshold = p_threshold
        self.n_startup_steps = n_startup_steps

    def create_sampler(self) -> TPESampler:
        """Create TPE sampler for Wilcoxon CV strategy."""
        return TPESampler(
            seed=self.config.random_state,
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True,
        )

    def create_pruner(self) -> Any:
        """
        Create Wilcoxon pruner for k-fold CV (SOTA).

        Falls back to MedianPruner if WilcoxonPruner not available.
        """
        if not self.config.enable_pruning:
            return None

        # SOTA: WilcoxonPruner for statistical CV pruning
        if WILCOXON_AVAILABLE:
            logger.info(
                f"Usando WilcoxonPruner SOTA (p_threshold={self.p_threshold})"
            )
            return WilcoxonPruner(
                p_threshold=self.p_threshold,
                n_startup_steps=self.n_startup_steps,
            )
        else:
            logger.warning(
                "WilcoxonPruner not available (requires Optuna >= 3.6.0), "
                "falling back to MedianPruner"
            )
            return MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3,
                interval_steps=1,
            )

    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest hyperparameters using TPE with Wilcoxon pruning."""
        return TPEStrategy(self.config).suggest_params(trial)


class StrategyFactory:
    """
    Factory for creating optimization strategies.

    Factory Pattern: Creates strategy instances based on name.
    """

    _strategies = {
        'tpe': TPEStrategy,
        'cmaes': CMAESStrategy,
        'hyperband': HyperbandStrategy,
        'wilcoxon_cv': WilcoxonCVStrategy,
    }

    @classmethod
    def create_strategy(cls, strategy_name: str, config) -> OptimizationStrategy:
        """
        Create optimization strategy based on name.

        Args:
            strategy_name: Name of the strategy ('tpe', 'cmaes', 'hyperband')
            config: Tuning configuration

        Returns:
            OptimizationStrategy instance

        Raises:
            ValueError: If strategy_name is not supported
        """
        if strategy_name not in cls._strategies:
            logger.warning(f"Unknown strategy '{strategy_name}', using TPE")
            strategy_name = 'tpe'

        return cls._strategies[strategy_name](config)

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """Get list of available strategy names."""
        return list(cls._strategies.keys())

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """
        Register a new strategy.

        Args:
            name: Name of the strategy
            strategy_class: Class implementing OptimizationStrategy
        """
        if not issubclass(strategy_class, OptimizationStrategy):
            raise ValueError("Strategy must inherit from OptimizationStrategy")

        cls._strategies[name] = strategy_class
        logger.info(f"Nova estratégia registrada: {name}")

"""
Legacy-friendly façade for unified hyperparameter optimization.

The real production code now lives under `scripts.optimization`, but the test
suite (and some historical notebooks) still import `scripts.unified_hyperopt`.
This module provides lightweight stand-ins that preserve the old API surface
without pulling heavyweight dependencies during test collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Legacy configuration objects
# ---------------------------------------------------------------------------


@dataclass
class TuningConfig:
    """Minimal tunable configuration (subset of the old options)."""

    min_confidence_threshold: float = 0.05
    max_violation_percentage: float = 150.0
    random_state: int = 42
    enable_pruning: bool = True


@dataclass
class MultiObjectiveConfig:
    """Simple data container used by tests."""

    objectives: list[str] = None  # type: ignore[assignment]
    weights: list[float] = None  # type: ignore[assignment]
    reference_point: tuple[float, ...] = (0.0, 0.0, 0.0)
    direction: list[str] | str = None  # type: ignore[assignment]
    enable_pareto_front: bool = True
    save_pareto_solutions: bool = True

    def __post_init__(self) -> None:
        if self.objectives is None:
            self.objectives = ["f1", "roc_auc", "precision"]
        if self.weights is None:
            self.weights = [0.5, 0.3, 0.2]
        if self.direction is None:
            self.direction = ["maximize"] * len(self.objectives)
        elif isinstance(self.direction, str):
            self.direction = [self.direction] * len(self.objectives)


# ---------------------------------------------------------------------------
# Strategy implementations (stubs that keep the old interface alive)
# ---------------------------------------------------------------------------


class _BaseStrategy:
    """Shared helpers for strategy stubs."""

    def __init__(self, config: Optional[TuningConfig] = None):
        self.config = config or TuningConfig()

    def create_sampler(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__.lower(), "seed": self.config.random_state}

    def create_pruner(self) -> Optional[Dict[str, Any]]:
        if not self.config.enable_pruning:
            return None
        return {"type": "median", "startup_trials": 5}

    def suggest_params(self, _trial: Any) -> Dict[str, Any]:
        """Return deterministic hyperparameters so tests can assert on keys."""
        return {
            "min_confidence_threshold": self.config.min_confidence_threshold,
            "max_violation_percentage": self.config.max_violation_percentage,
            "xgb_n_estimators": 200,
        }


class TPEStrategy(_BaseStrategy):
    """Tree-structured Parzen Estimator."""


class CMAESStrategy(_BaseStrategy):
    """Covariance Matrix Adaptation Evolution Strategy."""

    def create_pruner(self) -> Optional[Dict[str, Any]]:
        if not self.config.enable_pruning:
            return None
        return {"type": "median", "startup_trials": 10}


class BOHBStrategy(_BaseStrategy):
    """Bayesian Optimization with HyperBand."""

    def create_pruner(self) -> Optional[Dict[str, Any]]:
        if not self.config.enable_pruning:
            return None
        return {"type": "hyperband", "reduction_factor": 3}


class TransEStrategy(_BaseStrategy):
    """Specialized strategy for TransE-specific hyperparameters."""

    def suggest_params(self, _trial: Any) -> Dict[str, Any]:
        return {
            "embedding_dim": 256,
            "margin": 2.0,
            "learning_rate": 0.005,
            "batch_size": 256,
        }


# ---------------------------------------------------------------------------
# Optimizer façade
# ---------------------------------------------------------------------------


class UnifiedHyperoptimizer:
    """
    Compatibility wrapper that mimics the old orchestrator.
    The heavy lifting now lives elsewhere, so this class only stores metadata.
    """

    def __init__(
        self,
        *,
        strategy: Optional[_BaseStrategy] = None,
        config: Optional[TuningConfig] = None,
        output_dir: Optional[str | Path] = None,
        multi_objective: bool = False,
        multi_obj_config: Optional[MultiObjectiveConfig] = None,
    ):
        self.config = config or TuningConfig()
        self.strategy = strategy or TPEStrategy(self.config)
        self.output_dir = Path(output_dir or "outputs/optimization/unified")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.multi_objective = multi_objective
        self.multi_obj_config = multi_obj_config
        logger.debug(
            "UnifiedHyperoptimizer initialized (strategy=%s, multi_objective=%s)",
            self.strategy.__class__.__name__,
            self.multi_objective,
        )


class HyperparameterTuner:
    """
    Legacy helper used by docs/tests. Methods return structured stubs so callers
    receive predictable dictionaries without kicking off expensive jobs.
    """

    def __init__(
        self,
        *,
        neural_model_path: str,
        rules_path: str,
        lightgbm_model_path: str,
        output_dir: Optional[str | Path] = None,
    ):
        self.neural_model_path = neural_model_path
        self.rules_path = rules_path
        self.lightgbm_model_path = lightgbm_model_path
        self.output_dir = Path(output_dir or "outputs/optimization/legacy")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_ensemble_pipeline(self):
        return None

    def grid_search(self, param_grid: Dict[str, Any], *, scoring: str = "f1") -> Dict[str, Any]:
        return {
            "strategy": "grid",
            "scoring": scoring,
            "best_params": {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in param_grid.items()},
            "best_score": 0.0,
        }

    def random_search(self, param_distributions: Dict[str, Any], n_iter: int = 10) -> Dict[str, Any]:
        return {
            "strategy": "random",
            "iterations": n_iter,
            "best_params": {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in param_distributions.items()},
            "best_score": 0.0,
        }

    def optuna_optimize(self, n_trials: int = 10) -> Dict[str, Any]:
        return {
            "best_params": {"min_confidence_threshold": 0.05},
            "best_value": 0.0,
            "trials": n_trials,
        }


class OptimizerFactory:
    """Factory returning UnifiedHyperoptimizer instances for the requested strategy."""

    _MAP = {
        "tpe": TPEStrategy,
        "cmaes": CMAESStrategy,
        "bohb": BOHBStrategy,
        "transe": TransEStrategy,
    }

    @classmethod
    def create_optimizer(
        cls,
        strategy_name: str,
        *,
        multi_objective: bool = False,
        config: Optional[TuningConfig] = None,
    ) -> UnifiedHyperoptimizer:
        strategy_cls = cls._MAP.get(strategy_name.lower(), TPEStrategy)
        config = config or TuningConfig()
        strategy = strategy_cls(config)
        return UnifiedHyperoptimizer(
            strategy=strategy,
            config=config,
            multi_objective=multi_objective,
        )


__all__ = [
    "TPEStrategy",
    "CMAESStrategy",
    "BOHBStrategy",
    "TransEStrategy",
    "UnifiedHyperoptimizer",
    "HyperparameterTuner",
    "OptimizerFactory",
    "MultiObjectiveConfig",
]

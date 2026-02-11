#!/usr/bin/env python3
"""
Base Strategy Abstract Class

Defines the interface for all optimization strategies.
Strategy Pattern: Allows interchangeable optimization algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class OptimizationConfig:
    """Configuration for optimization strategies.

    Attributes:
        n_trials: Number of trials to run.
        timeout_seconds: Optional wall-clock timeout for the optimization run.
        n_jobs: Parallel jobs for the underlying optimizer (framework dependent).
        random_state: Random seed for samplers.
        enable_pruning: Whether pruning is enabled for the study.
        show_progress_bar: Whether to show framework progress bars.
        storage_url: Optional Optuna storage URL (RDB).
        study_name: Optional study name.
        direction: Optimization direction ("maximize" or "minimize").
        pruner_type: Pruner type ("hyperband", "median", or "wilcoxon").
        wilcoxon_p_threshold: WilcoxonPruner p-value threshold (Optuna >= 3.6.0).
        wilcoxon_n_startup_steps: WilcoxonPruner warmup steps before pruning starts.
    """

    n_trials: int = 50
    timeout_seconds: int | None = None
    n_jobs: int = -1
    random_state: int = 42
    enable_pruning: bool = True
    show_progress_bar: bool = True
    storage_url: str | None = None
    study_name: str | None = None
    direction: str = "maximize"
    pruner_type: str = "hyperband"
    wilcoxon_p_threshold: float = 0.1
    wilcoxon_n_startup_steps: int = 2


@dataclass
class TrialResult:
    """Result from a single trial."""

    params: dict[str, Any]
    value: float | None
    trial_number: int
    state: str
    intermediate_values: dict[int, float] | None = None
    user_attrs: dict[str, Any] | None = None


@dataclass
class OptimizationResult:
    """Complete result from optimization."""

    best_params: dict[str, Any]
    best_value: float
    best_trial_number: int
    n_trials: int
    trials: list[TrialResult]
    study_name: str
    optimization_time: float
    framework: str


class BaseOptimizerStrategy(ABC):
    """
    Abstract base class for optimization strategies.

    Strategy Pattern: Defines interface for all optimization frameworks.
    Concrete strategies implement these methods for their specific framework.
    """

    def __init__(self, config: OptimizationConfig):
        """
        Initialize strategy with configuration.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.study = None
        self._framework_name = "Unknown"

    @property
    def framework_name(self) -> str:
        """Get the framework name (e.g., 'optuna', 'hyperopt')."""
        return self._framework_name

    @abstractmethod
    def create_study(self) -> Any:
        """
        Create optimization study.

        Returns:
            Framework-specific study object
        """
        raise NotImplementedError

    @abstractmethod
    def suggest_params(self, trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
        """
        Suggest hyperparameters for a trial.

        Args:
            trial: Framework-specific trial object
            search_space: Dictionary defining search space

        Returns:
            Dictionary of suggested parameters
        """
        raise NotImplementedError

    @abstractmethod
    def run_optimization(
        self,
        objective_fn: Callable[[Any], float | list[float]],
        search_space: dict[str, Any],
    ) -> OptimizationResult:
        """
        Run the complete optimization process.

        Args:
            objective_fn: Objective function to optimize
            search_space: Search space definition

        Returns:
            OptimizationResult with best parameters and metrics
        """
        raise NotImplementedError

    @abstractmethod
    def get_best_trial(self) -> TrialResult:
        """
        Get the best trial from optimization.

        Returns:
            TrialResult for the best trial
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_trials(self) -> list[TrialResult]:
        """
        Get all trials from optimization.

        Returns:
            List of TrialResult objects
        """
        raise NotImplementedError

    @abstractmethod
    def get_optimization_history(self) -> list[tuple[int, float]]:
        """
        Get optimization history (trial_number, value).

        Returns:
            List of (trial_number, value) tuples
        """
        raise NotImplementedError

    @abstractmethod
    def get_param_importances(self) -> dict[str, float]:
        """
        Get parameter importance scores.

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        raise NotImplementedError

    def should_prune(self, trial: Any) -> bool:
        """
        Check if trial should be pruned (if pruning is enabled).

        Args:
            trial: Framework-specific trial object

        Returns:
            True if trial should be pruned
        """
        if not self.config.enable_pruning:
            return False

        try:
            return self._check_pruning_condition(trial)
        except Exception:
            return False

    @abstractmethod
    def _check_pruning_condition(self, trial: Any) -> bool:
        """
        Framework-specific pruning check.

        Args:
            trial: Framework-specific trial object

        Returns:
            True if trial should be pruned
        """
        raise NotImplementedError

    def save_study(self, output_path: Path) -> None:
        """
        Save study to disk.

        Args:
            output_path: Path to save study
        """
        try:
            self._save_study_impl(output_path)
        except Exception as e:
            from pff.shared.core.logging import logger

            logger.warning(f"Failed to save study: {e}")

    @abstractmethod
    def _save_study_impl(self, output_path: Path) -> None:
        """
        Framework-specific study saving implementation.

        Args:
            output_path: Path to save study
        """
        raise NotImplementedError

    def load_study(self, input_path: Path) -> None:
        """
        Load study from disk.

        Args:
            input_path: Path to load study from
        """
        try:
            self._load_study_impl(input_path)
        except Exception as e:
            from pff.shared.core.logging import logger

            logger.warning(f"Failed to load study: {e}")

    @abstractmethod
    def _load_study_impl(self, input_path: Path) -> None:
        """
        Framework-specific study loading implementation.

        Args:
            input_path: Path to load study from
        """
        raise NotImplementedError

    def get_stats(self) -> dict[str, Any]:
        """
        Get optimization statistics.

        Returns:
            Dictionary with optimization statistics
        """
        if not self.study:
            return {}

        trials = self.get_all_trials()
        return {
            "n_trials": len(trials),
            "n_completed": len([t for t in trials if t.state == "COMPLETE"]),
            "n_pruned": len([t for t in trials if t.state == "PRUNED"]),
            "n_failed": len([t for t in trials if t.state == "FAIL"]),
            "best_value": max([t.value for t in trials]) if trials else None,
            "framework": self.framework_name,
        }

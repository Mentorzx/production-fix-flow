"""Observer implementations for optimization callbacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import optuna

from pff.shared import logger

from .configs import _get_callback_config


class OptimizationObserver(ABC):
    """
    Abstract base class for optimization observers.

    Supports full optimization lifecycle: start, trial completion, end.
    """

    def on_optimization_start(self, study_name: str, n_trials: int) -> None:
        """
        Args:
            study_name: Name of the study
            n_trials: Total number of trials planned
        """
        return None

    @abstractmethod
    def on_trial_complete(self, trial: Any, value: float) -> None:
        """
        Args:
            trial: Optuna trial object
            value: Trial's objective value
        """
        raise NotImplementedError

    def on_optimization_end(self, best_value: float, best_params: dict[str, Any]) -> None:
        """
        Args:
            best_value: Best objective value found
            best_params: Best parameters found
        """
        return None


class CompositeObserver(OptimizationObserver):
    """
    Composite Observer for dispatching to multiple observers.

    Allows treating a group of observers as a single observer,
    simplifying callback management. Propagates all lifecycle events.
    """

    def __init__(self, observers: list[OptimizationObserver] | None = None):
        """
        Args:
            observers: Optional list of observers to wrap
        """
        self._observers: list[OptimizationObserver] = observers or []

    def add(self, observer: OptimizationObserver) -> CompositeObserver:
        """
        Args:
            observer: Observer to add

        Returns:
            Self for chaining
        """
        self._observers.append(observer)
        return self

    def remove(self, observer: OptimizationObserver) -> CompositeObserver:
        """
        Args:
            observer: Observer to remove

        Returns:
            Self for chaining
        """
        if observer in self._observers:
            self._observers.remove(observer)
        return self

    def on_optimization_start(self, study_name: str, n_trials: int) -> None:
        for observer in self._observers:
            try:
                observer.on_optimization_start(study_name, n_trials)
            except Exception as e:
                logger.error(f"Observer {observer.__class__.__name__} failed on start: {e}")

    def on_trial_complete(self, trial: Any, value: float) -> None:
        """
        Args:
            trial: Optuna trial object
            value: Trial's objective value
        """
        for observer in self._observers:
            try:
                observer.on_trial_complete(trial, value)
            except Exception as e:
                logger.error(f"Observer {observer.__class__.__name__} failed: {e}")

    def on_optimization_end(self, best_value: float, best_params: dict[str, Any]) -> None:
        for observer in self._observers:
            try:
                observer.on_optimization_end(best_value, best_params)
            except Exception as e:
                logger.error(f"Observer {observer.__class__.__name__} failed on end: {e}")

    def __len__(self) -> int:
        return len(self._observers)

    def __iter__(self):
        return iter(self._observers)


class LoggingObserver(OptimizationObserver):
    """
    Observer that logs trial progress.

    Logs trial completion at specified intervals to avoid
    overwhelming the logs.
    """

    def __init__(self, log_interval: int | None = None):
        """
        Args:
            log_interval: Log every N trials (default from config or 10)
        """
        callback_config = _get_callback_config()
        self.log_interval = log_interval or callback_config.get("log_interval", 10)
        self.trial_count = 0

    def on_trial_complete(self, trial, value: float) -> None:
        """
        Args:
            trial: Optuna trial object
            value: Trial's objective value
        """
        self.trial_count += 1
        if self.trial_count % self.log_interval == 0:
            logger.info(f"Ensaio {self.trial_count}: score={value:.4f}, parametros={trial.params}")


class BestScoreObserver(OptimizationObserver):
    """
    Observer that tracks and reports best score.

    Updates and reports when a new best score is found.
    """

    def __init__(self):
        self.best_score = -np.inf
        self.best_trial_number = 0
        self.improvement_count = 0

    def on_trial_complete(self, trial, value: float) -> None:
        """
        Args:
            trial: Optuna trial object
            value: Trial's objective value
        """
        if value > self.best_score:
            improvement = value - self.best_score
            self.best_score = value
            self.best_trial_number = trial.number
            self.improvement_count += 1

            logger.success(
                f"Novo melhor score: {value:.4f} "
                f"(+{improvement:.4f}, trial {trial.number}, "
                f"melhoria #{self.improvement_count})"
            )

    def get_best_score(self) -> float:
        return self.best_score

    def get_improvement_count(self) -> int:
        return self.improvement_count


class CallbackManager:
    """
    Manager for optimization callbacks.

    Coordinates multiple observers and provides centralized
    callback handling. Uses CompositeObserver internally.
    """

    def __init__(self):
        self._composite = CompositeObserver()

    @property
    def observers(self) -> list[OptimizationObserver]:
        return list(self._composite)

    def add_observer(self, observer: OptimizationObserver):
        """
        Args:
            observer: Observer to add
        """
        self._composite.add(observer)
        logger.debug(f"Added observer: {observer.__class__.__name__}")

    def remove_observer(self, observer: OptimizationObserver):
        """
        Args:
            observer: Observer to remove
        """
        self._composite.remove(observer)
        logger.debug(f"Removed observer: {observer.__class__.__name__}")

    def notify_all(self, trial, value: float):
        """
        Args:
            trial: Optuna trial object
            value: Trial's objective value
        """
        self._composite.on_trial_complete(trial, value)

    def notify_start(self, study_name: str, n_trials: int) -> None:
        self._composite.on_optimization_start(study_name, n_trials)

    def notify_end(self, best_value: float, best_params: dict[str, Any]) -> None:
        self._composite.on_optimization_end(best_value, best_params)

    def get_observer_names(self) -> list[str]:
        return [obs.__class__.__name__ for obs in self._composite]

    def clear(self):
        self._composite = CompositeObserver()
        logger.debug("Cleared all observers")


class MLflowTrialObserver(OptimizationObserver):
    """
    Observer that integrates MLflow tracking with the optimization workflow.

    Design Patterns:
    - Observer: Observes optimization events
    - Adapter: Adapts MLflowTracker to OptimizationObserver interface
    """

    def __init__(self, tracker: Any) -> None:
        """
        Args:
            tracker: MLflowTracker instance
        """
        self.tracker = tracker
        self.trial_count = 0
        self.study_name = "optuna_study"
        self.best_value = float("-inf")
        self.best_trial_number = -1

    def on_optimization_start(self, study_name: str, n_trials: int) -> None:
        """
        Args:
            study_name: Name of the study
            n_trials: Total number of trials planned
        """
        self.trial_count = 0
        self.study_name = study_name
        try:
            from pff.infrastructure.hpo.tracker import MLflowTracker

            if isinstance(self.tracker, MLflowTracker):
                self.tracker.log_optimization_start(
                    n_trials=n_trials,
                    strategy_name=study_name,
                    search_space={},
                )
        except Exception as e:
            logger.warning(f"Failed to log optimization start to MLflow: {e}")

    def on_trial_complete(self, trial: Any, value: float) -> None:
        """
        Args:
            trial: Optuna trial object
            value: Trial's objective value
        """
        try:
            from pff.infrastructure.hpo.strategies.base import TrialResult

            self.trial_count += 1
            if value > self.best_value:
                self.best_value = float(value)
                self.best_trial_number = int(getattr(trial, "number", -1))

            trial_result = TrialResult(
                trial_number=trial.number,
                value=value,
                params=dict(trial.params),
                state=str(getattr(trial, "state", "COMPLETE")),
                intermediate_values=dict(getattr(trial, "intermediate_values", {}) or {}),
                user_attrs=dict(getattr(trial, "user_attrs", {}) or {}),
            )

            self.tracker.log_trial(trial_result, self.trial_count)

        except Exception as e:
            logger.debug(f"Failed to log trial to MLflow: {e}")

    def on_optimization_end(self, best_value: float, best_params: dict[str, Any]) -> None:
        """
        Args:
            best_value: Best objective value found
            best_params: Best parameters found
        """
        try:
            from pff.infrastructure.hpo.strategies.base import OptimizationResult

            result = OptimizationResult(
                best_params=best_params,
                best_value=best_value,
                best_trial_number=self.best_trial_number,
                n_trials=self.trial_count,
                optimization_time=0.0,
                framework="optuna",
                trials=[],
                study_name=self.study_name,
            )

            self.tracker.log_optimization_end(result)

        except Exception as e:
            logger.warning(f"Failed to log optimization end to MLflow: {e}")


class MaxTrialsCallback:
    """
    Optuna callback that stops the study when a global maximum number of trials is reached.
    """

    def __init__(self, max_trials: int):
        self.max_trials = max_trials

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.Trial) -> None:
        if len(study.trials) >= self.max_trials:
            study.stop()

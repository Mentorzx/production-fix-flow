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


class StagnationDetector(OptimizationObserver):
    """
    Detects when HPO optimization is stuck in a local optimum.

    Monitors score improvements and triggers warnings/recommendations
    when stagnation is detected, suggesting sampler changes.
    """

    def __init__(
        self, window_size: int = 7, min_trials: int = 10, improvement_threshold: float = 0.01
    ):
        """
        Args:
            window_size: Number of recent trials to check for stagnation
            min_trials: Minimum trials before checking for stagnation
            improvement_threshold: Minimum relative improvement to not be considered stagnant
        """
        self.window_size = window_size
        self.min_trials = min_trials
        self.improvement_threshold = improvement_threshold
        self.scores: list[float] = []
        self.best_score = -np.inf
        self.best_trial_number = 0
        self.stagnation_detected = False
        self.trials_since_improvement = 0

    def on_trial_complete(self, trial, value: float) -> None:
        """
        Args:
            trial: Optuna trial object
            value: Trial's objective value
        """
        self.scores.append(value)

        if value > self.best_score:
            self.best_score = value
            self.best_trial_number = trial.number
            self.trials_since_improvement = 0
        else:
            self.trials_since_improvement += 1

        if trial.number >= self.min_trials and not self.stagnation_detected:
            self._check_stagnation(trial.number)

    def _check_stagnation(self, current_trial_number: int) -> None:
        """Check if optimization has stagnated based on recent trials."""
        if len(self.scores) < self.window_size:
            return

        recent_scores = self.scores[-self.window_size :]
        recent_best = max(recent_scores)
        recent_worst = min(recent_scores)

        if recent_worst == 0:
            relative_range = 0
        else:
            relative_range = (recent_best - recent_worst) / abs(recent_worst)

        has_significant_variance = relative_range > self.improvement_threshold
        has_recent_improvement = self.trials_since_improvement < self.window_size

        if not has_significant_variance and not has_recent_improvement:
            self.stagnation_detected = True
            stagnation_msg = (
                f"HPO stagnation detected after {current_trial_number} trials. "
                f"Best score: {self.best_score:.4f} (trial {self.best_trial_number}), "
                f"Recent range: {relative_range:.2%}. "
                f"Recommend: restart with sampler.type='cmaes' in config/hpo/optimization.yaml"
            )
            logger.warning(f"component_name=stagnation_detector message='{stagnation_msg}'")

    def is_stagnant(self) -> bool:
        return self.stagnation_detected

    def get_trials_since_improvement(self) -> int:
        return self.trials_since_improvement


class AdaptiveSamplerController(OptimizationObserver):
    """
    Automatically switches between samplers when stagnation is detected.

    Implements an alternating strategy: TPE (primary) ↔ GPSampler (alternative).
    When stagnation is detected in the current sampler, it switches to the other
    and uses warm-start with the best parameters found so far.

    This pattern continues until optimization completes or max switches reached.
    """

    def __init__(
        self,
        study: Any,
        sampler_settings: dict[str, Any],
        window_size: int = 7,
        min_trials: int = 10,
        improvement_threshold: float = 0.01,
        max_switches: int = 3,
    ):
        """
        Args:
            study: Optuna study object to modify sampler
            sampler_settings: Configuration dict for samplers
            window_size: Trials window for stagnation detection
            min_trials: Minimum trials before checking stagnation
            improvement_threshold: Relative improvement threshold
            max_switches: Maximum number of sampler switches allowed
        """
        self.study = study
        self.sampler_settings = sampler_settings
        self.window_size = window_size
        self.min_trials = min_trials
        self.improvement_threshold = improvement_threshold
        self.max_switches = max_switches

        self.scores: list[float] = []
        self.best_score = -np.inf
        self.best_params: dict[str, Any] = {}
        self.best_trial_number = 0
        self.trials_since_improvement = 0
        self.switch_count = 0

        self.current_sampler_type = sampler_settings.get("type", "tpe")
        self.sampler_history: list[str] = [self.current_sampler_type]

        self.stagnation_active = False
        self.trials_since_switch = 0

    def on_trial_complete(self, trial: Any, value: float) -> None:
        """
        Monitor trials and trigger sampler switch on stagnation.

        Args:
            trial: Optuna trial object
            value: Trial objective value
        """
        self.scores.append(value)
        self.trials_since_switch += 1

        if value > self.best_score:
            self.best_score = value
            self.best_params = trial.params
            self.best_trial_number = trial.number
            self.trials_since_improvement = 0
            self.stagnation_active = False
        else:
            self.trials_since_improvement += 1

        if (
            trial.number >= self.min_trials
            and self.switch_count < self.max_switches
            and self.trials_since_switch >= self.window_size
        ):
            self._check_and_switch(trial.number)

    def _check_and_switch(self, current_trial_number: int) -> None:
        """Check for stagnation and switch sampler if needed."""
        recent_scores = self.scores[-self.window_size :]
        recent_best = max(recent_scores)
        recent_worst = min(recent_scores)

        if recent_worst == 0:
            relative_range = 0
        else:
            relative_range = (recent_best - recent_worst) / abs(recent_worst)

        has_significant_variance = relative_range > self.improvement_threshold
        has_recent_improvement = self.trials_since_improvement < self.window_size

        if not has_significant_variance and not has_recent_improvement:
            if not self.stagnation_active:
                self.stagnation_active = True
                self._switch_sampler(current_trial_number)

    def _switch_sampler(self, trial_number: int) -> None:
        """
        Switch between TPE and GPSampler with warm-start.

        Args:
            trial_number: Current trial number for logging
        """
        new_sampler_type = "gp" if self.current_sampler_type == "tpe" else "tpe"
        self.switch_count += 1

        logger.info(
            f"component_name=adaptive_sampler_controller "
            f"key_parameters={{'trial': {trial_number}, 'from_sampler': '{self.current_sampler_type}', "
            f"'to_sampler': '{new_sampler_type}', 'switch_num': {self.switch_count}, "
            f"'best_score': {self.best_score:.4f}}} "
            f"message='Stagnation detected: switching sampler with warm-start'"
        )

        new_sampler = self._create_sampler(new_sampler_type)

        if self.best_params and new_sampler_type == "tpe":
            try:
                if hasattr(self.study, "_study"):
                    target_study = self.study._study
                else:
                    target_study = self.study

                for param_name, param_value in self.best_params.items():
                    try:
                        target_study.enqueue_trial({param_name: param_value})
                    except Exception:
                        pass

                logger.info(
                    f"component_name=adaptive_sampler_controller "
                    f"key_parameters={{'best_trial': {self.best_trial_number}, 'params_transferred': {len(self.best_params)}}} "
                    f"message='Warm-start: enqueued best parameters from previous sampler'"
                )
            except Exception as exc:
                logger.warning(
                    f"component_name=adaptive_sampler_controller "
                    f"key_parameters={{'error': {exc!r}}} "
                    "message='Warm-start failed, continuing without it'"
                )

        if hasattr(self.study, "set_sampler"):
            self.study.set_sampler(new_sampler)
        elif hasattr(self.study, "_study") and hasattr(self.study._study, "sampler"):
            self.study._study.sampler = new_sampler

        self.current_sampler_type = new_sampler_type
        self.sampler_history.append(new_sampler_type)
        self.trials_since_switch = 0
        self.trials_since_improvement = 0
        self.stagnation_active = False

        logger.success(
            f"component_name=adaptive_sampler_controller "
            f"key_parameters={{'new_sampler': '{new_sampler_type}', 'total_switches': {self.switch_count}}} "
            f"message='Sampler switch completed successfully'"
        )

    def _create_sampler(self, sampler_type: str) -> Any:
        """
        Create sampler instance based on type.

        Args:
            sampler_type: 'tpe' or 'gp'

        Returns:
            Optuna sampler instance
        """
        import optuna
        import warnings

        sampler_seed = int(self.sampler_settings.get("seed", 42))

        if sampler_type == "tpe":
            sampler_kwargs: dict[str, Any] = {
                "seed": sampler_seed,
                "n_startup_trials": 3,
                "n_ei_candidates": 24,
                "constant_liar": True,
                "consider_prior": True,
                "consider_magic_clip": True,
                "multivariate": True,
                "group": True,
            }
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
                return optuna.samplers.TPESampler(**sampler_kwargs)

        elif sampler_type == "gp":
            gp_kwargs: dict[str, Any] = {
                "seed": sampler_seed,
                "n_startup_trials": 3,
            }
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
                try:
                    return optuna.samplers.GPSampler(**gp_kwargs)
                except AttributeError:
                    logger.warning(
                        "component_name=adaptive_sampler_controller "
                        "message='GPSampler not available (Optuna < 3.6), falling back to TPESampler'"
                    )
                    return optuna.samplers.TPESampler(
                        seed=sampler_seed,
                        n_startup_trials=3,
                        n_ei_candidates=24,
                        multivariate=True,
                        group=True,
                    )

        else:
            logger.warning(
                f"component_name=adaptive_sampler_controller "
                f"key_parameters={{'unknown_sampler': '{sampler_type}'}} "
                f"message='Unknown sampler type, using TPE'"
            )
            return optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=3)

    def is_stagnant(self) -> bool:
        return self.stagnation_active

    def get_sampler_history(self) -> list[str]:
        return self.sampler_history

    def get_switch_count(self) -> int:
        return self.switch_count

    def get_current_sampler(self) -> str:
        return self.current_sampler_type


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

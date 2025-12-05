#!/usr/bin/env python3
"""
Callbacks Module

Implements Observer Pattern for monitoring and visualization
during optimization.

Design Patterns:
- Observer Pattern: Notifies observers of trial completion
- Template Method: Defines structure for observer updates
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from pff.utils import logger


class OptimizationObserver(ABC):
    """
    Abstract base class for optimization observers.

    Observer Pattern: Defines interface for observers that monitor optimization.
    Supports full optimization lifecycle: start, trial completion, end.
    """

    def on_optimization_start(self, study_name: str, n_trials: int) -> None:
        """
        Called when optimization starts.

        Args:
            study_name: Name of the study
            n_trials: Total number of trials planned
        """
        pass  # Optional hook - subclasses can override

    @abstractmethod
    def on_trial_complete(self, trial: Any, value: float) -> None:
        """
        Called when a trial completes.

        Args:
            trial: Optuna trial object
            value: Trial's objective value
        """
        pass

    def on_optimization_end(self, best_value: float, best_params: dict[str, Any]) -> None:
        """
        Called when optimization ends.

        Args:
            best_value: Best objective value found
            best_params: Best parameters found
        """
        pass  # Optional hook - subclasses can override


class CompositeObserver(OptimizationObserver):
    """
    Composite Observer Pattern for dispatching to multiple observers.

    Allows treating a group of observers as a single observer,
    simplifying callback management. Propagates all lifecycle events.
    """

    def __init__(self, observers: list[OptimizationObserver] | None = None):
        """
        Initialize composite observer.

        Args:
            observers: Optional list of observers to wrap
        """
        self._observers: list[OptimizationObserver] = observers or []

    def add(self, observer: OptimizationObserver) -> "CompositeObserver":
        """
        Add observer to composite.

        Args:
            observer: Observer to add

        Returns:
            Self for chaining
        """
        self._observers.append(observer)
        return self

    def remove(self, observer: OptimizationObserver) -> "CompositeObserver":
        """
        Remove observer from composite.

        Args:
            observer: Observer to remove

        Returns:
            Self for chaining
        """
        if observer in self._observers:
            self._observers.remove(observer)
        return self

    def on_optimization_start(self, study_name: str, n_trials: int) -> None:
        """Dispatch optimization start to all observers."""
        for observer in self._observers:
            try:
                observer.on_optimization_start(study_name, n_trials)
            except Exception as e:
                logger.error(f"Observer {observer.__class__.__name__} failed on start: {e}")

    def on_trial_complete(self, trial: Any, value: float) -> None:
        """
        Dispatch trial completion to all observers.

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
        """Dispatch optimization end to all observers."""
        for observer in self._observers:
            try:
                observer.on_optimization_end(best_value, best_params)
            except Exception as e:
                logger.error(f"Observer {observer.__class__.__name__} failed on end: {e}")

    def __len__(self) -> int:
        """Get number of observers."""
        return len(self._observers)

    def __iter__(self):
        """Iterate over observers."""
        return iter(self._observers)


def _get_callback_config() -> dict[str, Any]:
    """Load callback config from optimization.yaml."""
    try:
        from pff.utils.core.file_manager import FileManager
        from pathlib import Path
        fm = FileManager()
        config_path = Path("config/hpo/optimization.yaml")
        if config_path.exists():
            cfg = fm.read(config_path)
            return cfg.get("callbacks", {})
    except Exception:
        pass
    return {}


class LoggingObserver(OptimizationObserver):
    """
    Observer that logs trial progress.

    Logs trial completion at specified intervals to avoid
    overwhelming the logs.
    """

    def __init__(self, log_interval: int | None = None):
        """
        Initialize logging observer.

        Args:
            log_interval: Log every N trials (default from config or 10)
        """
        callback_config = _get_callback_config()
        self.log_interval = log_interval or callback_config.get("log_interval", 10)
        self.trial_count = 0

    def on_trial_complete(self, trial, value: float) -> None:
        """
        Log trial completion.

        Args:
            trial: Optuna trial object
            value: Trial's objective value
        """
        self.trial_count += 1
        if self.trial_count % self.log_interval == 0:
            logger.info(
                f"Ensaio {self.trial_count}: score={value:.4f}, "
                f"parametros={trial.params}"
            )


class BestScoreObserver(OptimizationObserver):
    """
    Observer that tracks and reports best score.

    Updates and reports when a new best score is found.
    """

    def __init__(self):
        """Initialize best score tracker."""
        self.best_score = -np.inf
        self.best_trial_number = 0
        self.improvement_count = 0

    def on_trial_complete(self, trial, value: float) -> None:
        """
        Update best score if improved.

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
        """Get current best score."""
        return self.best_score

    def get_improvement_count(self) -> int:
        """Get number of improvements found."""
        return self.improvement_count


class RealTimeVisualizer(OptimizationObserver):
    """
    Real-time visualization of optimization progress.

    Creates and updates matplotlib plots showing:
    - Optimization progress (current vs best scores)
    - Score distribution histogram
    - Parameter importance (if available)

    Note: Requires matplotlib and suitable display environment.
    """

    def __init__(self):
        """Initialize visualizer."""
        self.trial_numbers = []
        self.scores = []
        self.best_scores = []

        # Setup matplotlib
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.initialized = False

        try:
            plt.ion()  # Interactive mode
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
            self.fig.suptitle(
                'Hyperparameter Optimization - Real-Time',
                fontsize=14,
                fontweight='bold'
            )
            plt.show(block=False)  # Non-blocking
            self.initialized = True
            logger.success("Janela de visualizacao em tempo real aberta")
        except Exception as e:
            logger.warning(f"Could not create visualization window: {e}")
            self.initialized = False

    def on_trial_complete(self, trial, value: float) -> None:
        """
        Update visualization when a trial completes.

        Args:
            trial: Optuna trial object
            value: Trial's objective value
        """
        if not self.initialized:
            return

        try:
            self.trial_numbers.append(trial.number)
            self.scores.append(value)

            # Track best score
            if not self.best_scores or value > self.best_scores[-1]:
                self.best_scores.append(value)
            else:
                self.best_scores.append(self.best_scores[-1])

            # Update plots
            self._update_progress_plot()
            self._update_distribution_plot()

            plt.tight_layout()
            plt.pause(0.01)
        except Exception as e:
            logger.debug(f"Visualization update failed: {e}")

    def _update_progress_plot(self):
        """Update optimization progress plot."""
        if not self.ax1 or not self.trial_numbers:
            return

        self.ax1.clear()
        self.ax1.plot(
            self.trial_numbers,
            self.scores,
            'b-',
            label='Score',
            alpha=0.6,
            linewidth=1.5
        )
        self.ax1.plot(
            self.trial_numbers,
            self.best_scores,
            'r-',
            label='Best Score',
            linewidth=2
        )

        # Add ideal threshold line if available
        current_best = self.best_scores[-1] if self.best_scores else 0
        self.ax1.axhline(
            y=current_best,
            color='green',
            linestyle='--',
            alpha=0.5,
            label=f'Current Best: {current_best:.4f}'
        )

        self.ax1.set_xlabel('Trial')
        self.ax1.set_ylabel('Score')
        self.ax1.set_title('Optimization Progress')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)

        # Set y-axis limits for better visualization
        if self.scores:
            y_min = min(self.scores) - 0.05
            y_max = max(self.scores) + 0.05
            self.ax1.set_ylim(y_min, y_max)

    def _update_distribution_plot(self):
        """Update score distribution histogram."""
        if not self.ax2 or not self.scores:
            return

        self.ax2.clear()

        # Create histogram
        n_bins = min(20, max(5, len(self.scores) // 5))
        n, bins, patches = self.ax2.hist(
            self.scores,
            bins=n_bins,
            alpha=0.7,
            color='skyblue',
            edgecolor='black'
        )

        # Color bars based on score value
        for i, (patch, bin_center) in enumerate(zip(patches, (bins[:-1] + bins[1:]) / 2)):
            if self.best_scores and bin_center >= max(self.best_scores) * 0.95:
                patch.set_facecolor('gold')
            else:
                patch.set_facecolor('lightblue')

        self.ax2.set_xlabel('Score')
        self.ax2.set_ylabel('Frequency')
        self.ax2.set_title('Score Distribution')
        self.ax2.grid(True, alpha=0.3)

        # Add statistics text
        if self.scores:
            mean_score = np.mean(self.scores)
            std_score = np.std(self.scores)
            self.ax2.text(
                0.02,
                0.98,
                f'Mean: {mean_score:.4f}\nStd: {std_score:.4f}',
                transform=self.ax2.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

    def close(self):
        """Close visualization window."""
        if self.fig:
            plt.close(self.fig)
            logger.info("Janela de visualizacao encerrada")

    def save_plots(self, output_dir: str, prefix: str = "optimization") -> dict[str, str]:
        """
        Save current plots to files.

        Args:
            output_dir: Directory to save plots
            prefix: Filename prefix

        Returns:
            Dictionary mapping plot names to file paths
        """
        if not self.initialized or not self.fig:
            return {}

        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        try:
            # Save progress plot
            progress_file = output_path / f"{prefix}_progress.png"
            if self.ax1:
                self.ax1.figure.savefig(progress_file, dpi=300, bbox_inches='tight')
                saved_files['progress'] = str(progress_file)

            # Save distribution plot
            dist_file = output_path / f"{prefix}_distribution.png"
            if self.ax2:
                self.ax2.figure.savefig(dist_file, dpi=300, bbox_inches='tight')
                saved_files['distribution'] = str(dist_file)

            # Save combined plot
            combined_file = output_path / f"{prefix}_combined.png"
            self.fig.savefig(combined_file, dpi=300, bbox_inches='tight')
            saved_files['combined'] = str(combined_file)

            logger.success(f"Plots salvos em {output_path}")
            return saved_files

        except Exception as e:
            logger.error(f"Failed to save plots: {e}")
            return {}


class LivePlotCallback:
    """
    Optuna callback that writes convergence plots to disk after each trial.

    Design Pattern: Observer via Optuna callback; keeps a fixed x-axis range
    while updating values in real time.
    """

    def __init__(self, output_dir: Path, max_trials_axis: float = 50.0, expected_trials: int | None = None):
        """
        Initialize the live plot callback.

        Args:
            output_dir: Directory where plots will be written.
            max_trials_axis: Fixed x-axis limit (trials) to avoid rescaling.
            expected_trials: Optional expected trial count to size the axis.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_trials_axis = max(1.0, float(max_trials_axis))
        self.x_limit = max(self.max_trials_axis, float(expected_trials or 0))
        if self.x_limit <= 0:
            self.x_limit = self.max_trials_axis

        self.progress_fig, self.progress_ax = plt.subplots(figsize=(10, 4))
        self.hist_fig, self.hist_ax = plt.subplots(figsize=(10, 3.5))
        self.param_fig, self.param_ax = plt.subplots(figsize=(10, 4))

        self.progress_path = self.output_dir / "convergence.png"
        self.hist_path = self.output_dir / "score_distribution.png"
        self.param_path = self.output_dir / "param_convergence.png"

        logger.info(f"Plotagem em tempo real salva em: {self.output_dir}")

    def __call__(self, study: Any, trial: Any) -> None:
        """Optuna callback hook; refresh plots after each completed trial."""
        try:
            from optuna.trial import TrialState
        except Exception:
            return

        completed = [
            t for t in getattr(study, "trials", [])
            if getattr(t, "state", None) == TrialState.COMPLETE and getattr(t, "value", None) is not None
        ]
        if not completed:
            return

        ordered = sorted(completed, key=lambda t: t.number)
        trial_numbers = [t.number + 1 for t in ordered]
        scores = [float(t.value) for t in ordered]
        best_so_far = list(np.maximum.accumulate(scores))

        self._update_progress_plot(trial_numbers, scores, best_so_far)
        self._update_histogram(scores)
        self._update_param_plot(trial_numbers, ordered)
        self._save_plots()

    def _update_progress_plot(self, trial_numbers: list[int], scores: list[float], best_scores: list[float]) -> None:
        """Render convergence chart with fixed x-axis."""
        self.progress_ax.clear()
        self.progress_ax.plot(trial_numbers, scores, color="steelblue", linewidth=1.6, label="Score")
        self.progress_ax.plot(trial_numbers, best_scores, color="crimson", linewidth=1.8, label="Best Score")
        current_best = best_scores[-1] if best_scores else 0.0
        self.progress_ax.axhline(current_best, color="green", linestyle="--", alpha=0.4, label="Best Atual")

        self.progress_ax.set_xlim(0, self.x_limit)
        y_min = min(scores) if scores else 0.0
        y_max = max(scores) if scores else 1.0
        margin = max(0.05, (y_max - y_min) * 0.1)
        self.progress_ax.set_ylim(y_min - margin, y_max + margin)
        self.progress_ax.set_xlabel("Trial")
        self.progress_ax.set_ylabel("Score")
        self.progress_ax.set_title("Convergência do HPO")
        self.progress_ax.grid(True, alpha=0.3)
        self.progress_ax.legend()

    def _update_histogram(self, scores: list[float]) -> None:
        """Render histogram of observed scores."""
        self.hist_ax.clear()
        bins = min(20, max(5, len(scores)))
        self.hist_ax.hist(scores, bins=bins, color="skyblue", edgecolor="black", alpha=0.8)
        self.hist_ax.set_xlabel("Score")
        self.hist_ax.set_ylabel("Frequência")
        self.hist_ax.set_title("Distribuição dos Scores")
        self.hist_ax.grid(True, alpha=0.25)
        if scores:
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            self.hist_ax.text(
                0.02,
                0.98,
                f"Média: {mean_score:.4f}\nDesvio: {std_score:.4f}",
                transform=self.hist_ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    def _update_param_plot(self, trial_numbers: list[int], trials: list[Any]) -> None:
        """Render parameter trajectories for up to six numeric parameters."""
        self.param_ax.clear()
        if not trials:
            return

        param_series: dict[str, list[float]] = {}
        for trial in trials:
            params = getattr(trial, "params", {})
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    param_series.setdefault(key, []).append(float(value))

        if not param_series:
            self.param_ax.text(
                0.5,
                0.5,
                "Sem parâmetros numéricos para plotar",
                ha="center",
                va="center",
                transform=self.param_ax.transAxes,
            )
            return

        max_params = 6
        for name, series in list(param_series.items())[:max_params]:
            if len(series) != len(trial_numbers):
                pad_len = len(trial_numbers) - len(series)
                padded = [series[0]] * pad_len + series
                series = padded
            self.param_ax.plot(trial_numbers, series, linewidth=1.4, label=name)

        self.param_ax.set_xlim(0, self.x_limit)
        self.param_ax.set_xlabel("Trial")
        self.param_ax.set_ylabel("Valor do parâmetro")
        self.param_ax.set_title("Convergência dos parâmetros HPO")
        self.param_ax.grid(True, alpha=0.25)
        self.param_ax.legend(loc="upper right", fontsize=8, ncol=2)

    def _save_plots(self) -> None:
        """Persist plots to disk with fixed filenames."""
        try:
            self.progress_fig.tight_layout()
            self.progress_fig.savefig(self.progress_path, dpi=200, bbox_inches="tight")
            self.hist_fig.tight_layout()
            self.hist_fig.savefig(self.hist_path, dpi=200, bbox_inches="tight")
            self.param_fig.tight_layout()
            self.param_fig.savefig(self.param_path, dpi=200, bbox_inches="tight")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to save live plots: {exc}")


class CallbackManager:
    """
    Manager for optimization callbacks.

    Coordinates multiple observers and provides centralized
    callback handling. Uses CompositeObserver internally.
    """

    def __init__(self):
        """Initialize callback manager."""
        self._composite = CompositeObserver()

    @property
    def observers(self) -> list[OptimizationObserver]:
        """Get list of observers."""
        return list(self._composite)

    def add_observer(self, observer: OptimizationObserver):
        """
        Add an observer to the manager.

        Args:
            observer: Observer to add
        """
        self._composite.add(observer)
        logger.debug(f"Added observer: {observer.__class__.__name__}")

    def remove_observer(self, observer: OptimizationObserver):
        """
        Remove an observer from the manager.

        Args:
            observer: Observer to remove
        """
        self._composite.remove(observer)
        logger.debug(f"Removed observer: {observer.__class__.__name__}")

    def notify_all(self, trial, value: float):
        """
        Notify all observers of trial completion.

        Args:
            trial: Optuna trial object
            value: Trial's objective value
        """
        self._composite.on_trial_complete(trial, value)

    def get_observer_names(self) -> list[str]:
        """Get names of all registered observers."""
        return [obs.__class__.__name__ for obs in self._composite]

    def clear(self):
        """Remove all observers."""
        self._composite = CompositeObserver()
        logger.debug("Cleared all observers")


# ============================================================================
# MLflow Integration Observer
# ============================================================================

class MLflowTrialObserver(OptimizationObserver):
    """
    Observer that integrates MLflow tracking with the optimization workflow.
    
    Design Patterns:
    - Observer Pattern: Observes optimization events
    - Adapter Pattern: Adapts MLflowTracker to OptimizationObserver interface
    
    Example:
        tracker = MLflowTracker("my_experiment")
        observer = MLflowTrialObserver(tracker)
        callback_manager.add_observer(observer)
    """
    
    def __init__(self, tracker: Any) -> None:
        """
        Initialize MLflow trial observer.
        
        Args:
            tracker: MLflowTracker instance
        """
        self.tracker = tracker
        self.trial_count = 0
    
    def on_optimization_start(self, study_name: str, n_trials: int) -> None:
        """
        Log optimization start to MLflow.
        
        Args:
            study_name: Name of the study
            n_trials: Total number of trials planned
        """
        try:
            from .tracker import MLflowTracker
            if isinstance(self.tracker, MLflowTracker):
                self.tracker.log_optimization_start(
                    n_trials=n_trials,
                    strategy_name=study_name,
                    search_space={},  # Will be populated later
                )
        except Exception as e:
            logger.warning(f"Failed to log optimization start to MLflow: {e}")
    
    def on_trial_complete(self, trial: Any, value: float) -> None:
        """
        Log trial completion to MLflow.
        
        Args:
            trial: Optuna trial object
            value: Trial's objective value
        """
        try:
            from .strategies.base import TrialResult
            
            self.trial_count += 1
            
            # Create TrialResult for tracker
            trial_result = TrialResult(
                trial_number=trial.number,
                value=value,
                params=dict(trial.params),
                state=str(getattr(trial, "state", "COMPLETE")),
                intermediate_values={},
            )
            
            self.tracker.log_trial(trial_result, self.trial_count)
            
        except Exception as e:
            logger.debug(f"Failed to log trial to MLflow: {e}")
    
    def on_optimization_end(self, best_value: float, best_params: dict[str, Any]) -> None:
        """
        Log optimization end to MLflow.
        
        Args:
            best_value: Best objective value found
            best_params: Best parameters found
        """
        try:
            from .strategies.base import OptimizationResult, TrialResult
            
            # Create minimal OptimizationResult for tracker
            result = OptimizationResult(
                best_params=best_params,
                best_value=best_value,
                n_trials=self.trial_count,
                optimization_time=0.0,  # Not tracked here
                framework="optuna",
                trials=[],  # Not needed for end log
            )
            
            self.tracker.log_optimization_end(result)
            
        except Exception as e:
            logger.warning(f"Failed to log optimization end to MLflow: {e}")

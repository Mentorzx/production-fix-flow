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
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from pff.utils import logger


class OptimizationObserver(ABC):
    """
    Abstract base class for optimization observers.

    Observer Pattern: Defines interface for observers that monitor optimization.
    """

    @abstractmethod
    def on_trial_complete(self, trial: Any, value: float) -> None:
        """
        Called when a trial completes.

        Args:
            trial: Optuna trial object
            value: Trial's objective value
        """
        pass


class LoggingObserver(OptimizationObserver):
    """
    Observer that logs trial progress.

    Logs trial completion at specified intervals to avoid
    overwhelming the logs.
    """

    def __init__(self, log_interval: int = 10):
        """
        Initialize logging observer.

        Args:
            log_interval: Log every N trials
        """
        self.log_interval = log_interval
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
                f"Trial {self.trial_count}: score={value:.4f}, "
                f"params={trial.params}"
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
                f"🎯 New best score: {value:.4f} "
                f"(+{improvement:.4f}, trial {trial.number}, "
                f"improvement #{self.improvement_count})"
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
            logger.success("✅ Real-time visualization window opened")
        except Exception as e:
            logger.warning(f"⚠️ Could not create visualization window: {e}")
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
            logger.info("Visualization window closed")

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

            logger.success(f"Plots saved to {output_path}")
            return saved_files

        except Exception as e:
            logger.error(f"Failed to save plots: {e}")
            return {}


class CallbackManager:
    """
    Manager for optimization callbacks.

    Coordinates multiple observers and provides centralized
    callback handling.
    """

    def __init__(self):
        """Initialize callback manager."""
        self.observers: list[OptimizationObserver] = []

    def add_observer(self, observer: OptimizationObserver):
        """
        Add an observer to the manager.

        Args:
            observer: Observer to add
        """
        self.observers.append(observer)
        logger.debug(f"Added observer: {observer.__class__.__name__}")

    def remove_observer(self, observer: OptimizationObserver):
        """
        Remove an observer from the manager.

        Args:
            observer: Observer to remove
        """
        if observer in self.observers:
            self.observers.remove(observer)
            logger.debug(f"Removed observer: {observer.__class__.__name__}")

    def notify_all(self, trial, value: float):
        """
        Notify all observers of trial completion.

        Args:
            trial: Optuna trial object
            value: Trial's objective value
        """
        for observer in self.observers:
            try:
                observer.on_trial_complete(trial, value)
            except Exception as e:
                logger.error(f"Observer {observer.__class__.__name__} failed: {e}")

    def get_observer_names(self) -> list[str]:
        """Get names of all registered observers."""
        return [obs.__class__.__name__ for obs in self.observers]

    def clear(self):
        """Remove all observers."""
        self.observers.clear()
        logger.debug("Cleared all observers")

#!/usr/bin/env python3
"""
MLflow Integration - Complete Experiment Tracking

Provides automatic MLflow integration for hyperparameter optimization:
- Creates experiment for each optimization run
- Tracks parent run with summary metrics
- Tracks each trial as nested run with params and metrics
- Logs artifacts (plots, best_params.json, etc.)
- Provides context manager for easy use
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable
from contextlib import contextmanager

from pff.utils import logger
from .strategies.base import OptimizationResult, TrialResult


class MLflowTracker:
    """
    MLflow integration for hyperparameter optimization.

    Features:
    - Automatic experiment creation
    - Parent run with summary
    - Nested runs for each trial
    - Parameter and metric tracking
    - Artifact logging
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking URI (default: ./mlruns)
            artifact_location: Artifact location (default: ./mlartifacts)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location

        self.mlflow = None
        self.experiment_id = None
        self.parent_run_id = None
        self.active_run = None

        self._check_mlflow_availability()

    def _check_mlflow_availability(self) -> None:
        """Check if MLflow is installed."""
        try:
            import mlflow
            self.mlflow = mlflow

            # Set tracking URI
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            # Get or create experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment:
                    self.experiment_id = experiment.experiment_id
                else:
                    self.experiment_id = mlflow.create_experiment(
                        name=self.experiment_name,
                        artifact_location=self.artifact_location,
                    )
                logger.success(f"Experimento MLflow pronto: {self.experiment_name}")
            except Exception as e:
                logger.warning(f"MLflow experiment setup warning: {e}")

        except ImportError:
            logger.warning(
                "MLflow not installed. Install with: pip install mlflow\n"
                "Optimization will run without experiment tracking."
            )

    @contextmanager
    def start_run(self, run_name: str | None = None):
        """Context manager for MLflow run."""
        if not self.mlflow:
            # No MLflow, just yield None
            yield None
            return

        try:
            with self.mlflow.start_run(run_name=run_name) as run:
                self.active_run = run
                yield run
        except Exception as e:
            logger.warning(f"MLflow run error: {e}")
            yield None
        finally:
            self.active_run = None

    def log_optimization_start(
        self,
        n_trials: int,
        strategy_name: str,
        search_space: dict[str, Any],
    ) -> str:
        """
        Log optimization start.

        Args:
            n_trials: Number of trials
            strategy_name: Strategy name
            search_space: Search space definition

        Returns:
            Run ID
        """
        if not self.mlflow or not self.experiment_id:
            return ""

        try:
            with self.mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name="optimization_parent",
            ) as run:
                self.parent_run_id = run.info.run_id

                # Log parameters
                self.mlflow.log_param("n_trials", n_trials)
                self.mlflow.log_param("strategy", strategy_name)

                # Log search space
                search_space_size = self._estimate_search_space_size(search_space)
                self.mlflow.log_param("search_space_size", search_space_size)
                self.mlflow.log_param("search_space_keys", len(search_space))

                # Log search space as artifact (AGENTS.md §4.1)
                search_space_file = Path("search_space.json")
                self.file_manager.save(search_space, search_space_file)
                self.mlflow.log_artifact(search_space_file, "search_space")

                logger.info(f"Run MLflow pai iniciado: {run.info.run_id}")

                return run.info.run_id

        except Exception as e:
            logger.warning(f"Failed to log optimization start: {e}")
            return ""

    def log_trial(
        self,
        trial: TrialResult,
        trial_idx: int,
    ) -> None:
        """
        Log a single trial as nested run.

        Args:
            trial: Trial result
            trial_idx: Trial index
        """
        if not self.mlflow or not self.parent_run_id:
            return

        try:
            # Start nested run
            with self.mlflow.start_run(
                run_name=f"trial_{trial.trial_number}",
                nested=True,
            ) as run:
                # Log parameters
                for key, value in trial.params.items():
                    if isinstance(value, (int, float, str)):
                        self.mlflow.log_param(key, value)

                # Log metrics
                self.mlflow.log_metric("value", trial.value, step=trial_idx)

                # Log trial state
                self.mlflow.log_param("state", trial.state)
                self.mlflow.log_param("trial_number", trial.trial_number)

                # Log intermediate values if available
                if trial.intermediate_values:
                    for step, value in trial.intermediate_values.items():
                        self.mlflow.log_metric(
                            f"intermediate_{step}",
                            value,
                            step=trial_idx
                        )

        except Exception as e:
            logger.debug(f"Failed to log trial {trial.trial_number}: {e}")

    def log_optimization_end(
        self,
        result: OptimizationResult,
    ) -> None:
        """
        Log optimization completion.

        Args:
            result: Complete optimization result
        """
        if not self.mlflow or not self.parent_run_id:
            return

        try:
            # Log summary metrics to parent run
            with self.mlflow.start_run(
                run_id=self.parent_run_id,
                nested=False,
            ):
                # Best metrics
                self.mlflow.log_metric("best_value", result.best_value)
                self.mlflow.log_param("best_params", json.dumps(result.best_params))

                # Optimization stats
                self.mlflow.log_param("n_trials_completed", result.n_trials)
                self.mlflow.log_metric("optimization_time_sec", result.optimization_time)
                self.mlflow.log_param("framework", result.framework)

                # Trial statistics
                n_completed = len([t for t in result.trials if t.state == 'COMPLETE'])
                n_pruned = len([t for t in result.trials if t.state == 'PRUNED'])
                n_failed = result.n_trials - n_completed - n_pruned

                self.mlflow.log_param("n_completed", n_completed)
                self.mlflow.log_param("n_pruned", n_pruned)
                self.mlflow.log_param("n_failed", n_failed)

                # Save best params as artifact (AGENTS.md §4.1)
                best_params_file = Path("best_params.json")
                self.file_manager.save(result.best_params, best_params_file)
                self.mlflow.log_artifact(best_params_file, "best_params")

                logger.success(
                    f"Otimização MLflow concluída: {result.best_value:.4f}"
                )

        except Exception as e:
            logger.warning(f"Failed to log optimization end: {e}")

    def log_artifacts(
        self,
        artifacts: dict[str, Path],
        artifact_path: str | None = None,
    ) -> None:
        """
        Log artifacts to MLflow.

        Args:
            artifacts: Dictionary mapping artifact names to file paths
            artifact_path: Artifact subdirectory in MLflow
        """
        if not self.mlflow:
            return

        try:
            for name, path in artifacts.items():
                if path.exists():
                    self.mlflow.log_artifact(str(path), artifact_path or "artifacts")
                    logger.debug(f"Logged artifact: {name}")
        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")

    def get_tracking_uri(self) -> str | None:
        """Get MLflow tracking URI."""
        if not self.mlflow:
            return None

        try:
            return self.mlflow.get_tracking_uri()
        except Exception:
            return None

    def get_experiment_url(self) -> str | None:
        """
        Get URL to experiment in MLflow UI.

        Returns:
            URL string or None if not available
        """
        if not self.mlflow or not self.get_tracking_uri():
            return None

        tracking_uri = self.get_tracking_uri()

        # Convert file:// URL to web URL
        if tracking_uri.startswith("file:"):
            # Default MLflow UI port
            return "http://localhost:5000"

        return tracking_uri

    def _estimate_search_space_size(self, search_space: dict[str, Any]) -> int:
        """
        Estimate total search space size (very rough approximation).

        Args:
            search_space: Search space definition

        Returns:
            Estimated number of combinations
        """
        total = 1
        for param_name, param_config in search_space.items():
            if isinstance(param_config, (list, tuple)):
                if len(param_config) == 2:
                    # Numeric range: estimate 100 values
                    total *= 100
                else:
                    # Categorical
                    total *= len(param_config)
            elif isinstance(param_config, dict):
                param_type = param_config.get('type', 'float')
                if param_type == 'categorical':
                    total *= len(param_config.get('choices', []))
                else:
                    total *= 100  # Numeric: estimate 100 values
            else:
                total *= 100  # Default estimate

        return total

    def log_model_comparison(self, results: list[OptimizationResult]) -> None:
        """
        Log comparison of multiple optimization results.

        Args:
            results: List of optimization results
        """
        if not self.mlflow or len(results) < 2:
            return

        try:
            comparison_data = []
            for result in results:
                comparison_data.append(
                    {
                        'strategy': result.framework,
                        'best_value': result.best_value,
                        'n_trials': result.n_trials,
                        'optimization_time': result.optimization_time,
                    }
                )

            # Save comparison as CSV
            import polars as pl
            df = pl.DataFrame(comparison_data)
            comparison_file = Path("strategy_comparison.csv")
            df.to_csv(comparison_file, index=False)

            # Log to MLflow
            with self.mlflow.start_run(run_name="strategy_comparison", nested=True):
                self.mlflow.log_artifact(str(comparison_file), "comparison")

            logger.success("Comparação de estratégias registrada")

        except Exception as e:
            logger.warning(f"Failed to log model comparison: {e}")

    def create_model_registry_entry(
        self,
        model_name: str,
        result: OptimizationResult,
        model_path: Path | None = None,
    ) -> None:
        """
        Create entry in MLflow Model Registry.

        Args:
            model_name: Name for the model
            result: Optimization result with best params
            model_path: Optional path to model artifact
        """
        if not self.mlflow or not model_path or not model_path.exists():
            return

        try:
            # Register model with best parameters as metadata
            self.mlflow.register_model(
                model_uri=str(model_path),
                name=model_name,
            )

            # Log best params as model version properties
            self.mlflow.log_params(result.best_params)

            logger.success(f"Modelo registrado no MLflow: {model_name}")

        except Exception as e:
            logger.warning(f"Failed to register model: {e}")

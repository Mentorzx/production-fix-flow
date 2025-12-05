#!/usr/bin/env python3
"""
MLflow Integration - Complete Experiment Tracking

Provides automatic MLflow integration for hyperparameter optimization:
- Creates experiment for each optimization run
- Tracks parent run with summary metrics
- Tracks each trial as nested run with params and metrics
- Logs artifacts (plots, best_params.json, etc.)
- Provides context manager for easy use

Design Patterns:
- Adapter Pattern: Adapts MLflow API to HPO workflow
- Context Manager: Safe resource management for runs
- Observer Pattern: Observes trial/optimization events
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable
from contextlib import contextmanager

from pff import settings
from pff.config import ENSEMBLE_HPO_CONFIG_PATH
from pff.utils import logger
from pff.utils.core.file_manager import FileManager
from .strategies.base import OptimizationResult, TrialResult


def _load_mlflow_config() -> dict[str, Any]:
    """
    Load MLflow configuration from ensemble_hpo.yaml with environment overrides.

    Returns:
        Dict with enabled flag, tracking URI, and experiment name.
    """
    fm = FileManager()
    defaults = {
        "enabled": True,
        "tracking_uri": str(settings.OUTPUTS_DIR / "optimization" / "mlruns"),
        "experiment_name": "pff_hpo",
    }
    try:
        config = fm.read(ENSEMBLE_HPO_CONFIG_PATH) or {}
        mlflow_config = config.get("mlflow", {})
        merged = {**defaults, **mlflow_config}
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Failed to load MLflow config; using defaults: {exc}")
        merged = dict(defaults)

    env_enabled = os.getenv("PFF_MLFLOW_ENABLED")
    if env_enabled is not None:
        merged["enabled"] = env_enabled.strip().lower() not in {"0", "false", "no", "off"}

    env_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_tracking_uri:
        merged["tracking_uri"] = env_tracking_uri

    return merged


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
        experiment_name: str | None = None,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
        mlflow_config: dict[str, Any] | None = None,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of MLflow experiment (defaults to config)
            tracking_uri: MLflow tracking URI (default: outputs/optimization/mlruns or config)
            artifact_location: Artifact location (default: config or ./mlartifacts)
            mlflow_config: Optional pre-loaded MLflow config (for testing)
        """
        self.file_manager = FileManager()  # AGENTS.md §5 - route I/O through utils
        self.mlflow_config = mlflow_config or _load_mlflow_config()
        self.enabled = bool(self.mlflow_config.get("enabled", True))
        self.experiment_name = experiment_name or self.mlflow_config.get("experiment_name", "pff_hpo")
        default_tracking = settings.OUTPUTS_DIR / "optimization" / "mlruns"
        self.tracking_uri = tracking_uri or self.mlflow_config.get("tracking_uri") or str(default_tracking)
        self.artifact_location = artifact_location or self.mlflow_config.get("artifact_location")

        self.mlflow = None
        self.experiment_id = None
        self.parent_run_id = None
        self.active_run = None

        if not self.enabled:
            logger.info("Rastreamento MLflow desabilitado via config/env; prosseguindo sem tracking")
            return

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

                search_space_size = self._estimate_search_space_size(search_space)
                self.mlflow.log_params({
                    "n_trials": n_trials,
                    "strategy": strategy_name,
                    "search_space_size": search_space_size,
                    "search_space_keys": len(search_space),
                })

                search_space_file = settings.OUTPUTS_DIR / "mlflow" / "search_space.json"
                search_space_file.parent.mkdir(parents=True, exist_ok=True)
                self.file_manager.save(search_space, search_space_file)
                self.mlflow.log_artifact(str(search_space_file), "search_space")

                logger.info(f"Execucao MLflow pai iniciada: {run.info.run_id}")

                return run.info.run_id

        except Exception as e:
            logger.warning(f"Failed to log optimization start: {e}")
            return ""

    @staticmethod
    def _flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
        """Flatten numeric metrics, expanding nested dicts with dot notation."""
        flat: dict[str, float] = {}
        for key, value in (metrics or {}).items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (int, float)):
                flat[full_key] = float(value)
            elif isinstance(value, dict):
                flat.update(MLflowTracker._flatten_metrics(value, prefix=full_key))
        return flat

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
                params_to_log = {
                    k: v for k, v in trial.params.items()
                    if isinstance(v, (int, float, str, bool))
                }
                params_to_log["state"] = trial.state
                params_to_log["trial_number"] = trial.trial_number
                self.mlflow.log_params(params_to_log)

                # Log metrics
                self.mlflow.log_metric("value", trial.value, step=trial_idx)
                if trial.user_attrs:
                    flattened = self._flatten_metrics(trial.user_attrs)
                    for key, value in flattened.items():
                        self.mlflow.log_metric(key, value, step=trial_idx)

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
                self.mlflow.log_metric("optimization_time_sec", result.optimization_time)

                # Trial statistics
                n_completed = len([t for t in result.trials if t.state == 'COMPLETE'])
                n_pruned = len([t for t in result.trials if t.state == 'PRUNED'])
                n_failed = result.n_trials - n_completed - n_pruned

                self.mlflow.log_params({
                    "best_params": json.dumps(result.best_params),
                    "n_trials_completed": result.n_trials,
                    "framework": result.framework,
                    "n_completed": n_completed,
                    "n_pruned": n_pruned,
                    "n_failed": n_failed,
                })

                best_params_file = settings.OUTPUTS_DIR / "mlflow" / "best_params.json"
                best_params_file.parent.mkdir(parents=True, exist_ok=True)
                self.file_manager.save(result.best_params, best_params_file)
                self.mlflow.log_artifact(str(best_params_file), "best_params")

                # Log best-trial metrics if available via user_attrs
                best_trial_attrs = {}
                try:
                    best_trial = next(
                        (t for t in result.trials if t.trial_number == result.best_trial_number),
                        None,
                    )
                    if best_trial and best_trial.user_attrs:
                        best_trial_attrs = best_trial.user_attrs
                        flattened = self._flatten_metrics(best_trial.user_attrs, prefix="best")
                        for key, value in flattened.items():
                            self.mlflow.log_metric(key, value)
                except Exception as attr_exc:  # noqa: BLE001
                    logger.warning(f"Failed to log best trial metrics: {attr_exc}")

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

            # Save comparison as CSV (AGENTS.md §3.1 - use settings.OUTPUTS_DIR)
            import polars as pl
            df = pl.DataFrame(comparison_data)
            comparison_file = settings.OUTPUTS_DIR / "mlflow" / "strategy_comparison.csv"
            comparison_file.parent.mkdir(parents=True, exist_ok=True)
            df.write_csv(comparison_file)

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

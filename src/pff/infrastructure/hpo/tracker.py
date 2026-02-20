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

import logging
import os
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pff.infrastructure.hpo.config_loader import load_optimization_config
from pff.shared import logger
from pff.shared.core.config import settings
from pff.shared.core.file_manager import FileManager, ParquetBundle

from .strategies.base import OptimizationResult, TrialResult


def _coerce_text(value: Any) -> str | None:
    """Normalize config values to plain strings for MLflow compatibility."""
    if value is None:
        return None
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return text


def _suppress_mlflow_noise() -> None:
    """Reduce noisy MLflow/Alembic INFO logs during initialization."""
    for logger_name in (
        "alembic",
        "alembic.runtime.migration",
        "mlflow.store.db.utils",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _load_mlflow_config() -> dict[str, Any]:
    """
    Load MLflow configuration from config/hpo/optimization.yaml with environment overrides.

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
        config = load_optimization_config(file_manager=fm)
        mlflow_config = config.get("mlflow", {})
        merged = {**defaults, **mlflow_config}
    except Exception as exc:
        logger.debug(f"Failed to load MLflow config; using defaults: {exc}")
        merged = dict(defaults)

    env_enabled = os.getenv("PFF_MLFLOW_ENABLED")
    if env_enabled is not None:
        merged["enabled"] = env_enabled.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    env_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_tracking_uri:
        merged["tracking_uri"] = env_tracking_uri

    merged["tracking_uri"] = _coerce_text(merged.get("tracking_uri"))
    merged["experiment_name"] = _coerce_text(merged.get("experiment_name")) or "pff_hpo"
    merged["artifact_location"] = _coerce_text(merged.get("artifact_location"))

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
        self.file_manager = FileManager()
        self.mlflow_config = mlflow_config or _load_mlflow_config()
        self.enabled = bool(self.mlflow_config.get("enabled", True))
        self.experiment_name: str = (
            _coerce_text(experiment_name or self.mlflow_config.get("experiment_name") or "pff_hpo")
            or "pff_hpo"
        )
        default_tracking = settings.OUTPUTS_DIR / "optimization" / "mlruns"
        self.tracking_uri: str = _coerce_text(
            tracking_uri or self.mlflow_config.get("tracking_uri") or str(default_tracking)
        ) or str(default_tracking)
        self.artifact_location: str | None = _coerce_text(
            artifact_location or self.mlflow_config.get("artifact_location")
        )

        self.mlflow: Any = None
        self.experiment_id: str | None = None
        self.parent_run_id: str | None = None
        self.active_run: Any = None

        if not self.enabled:
            logger.info(
                "Rastreamento MLflow desabilitado via config/env; prosseguindo sem tracking"
            )
            return

        self._check_mlflow_availability()

    def _check_mlflow_availability(self) -> None:
        """Check if MLflow is installed."""
        try:
            import mlflow

            _suppress_mlflow_noise()
            self.mlflow = mlflow

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            try:
                had_corruption = self._sanitize_tracking_store()
                if had_corruption:
                    raise RuntimeError("MLflow tracking store corruption detected")
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
                logger.error(f"MLflow experiment setup failed: {e}")
                raise

        except ImportError:
            logger.warning(
                "MLflow not installed. Install with: pip install mlflow\n"
                "Optimization will run without experiment tracking."
            )

    def _resolve_tracking_path(self) -> Path | None:
        """Resolve local tracking directory when using MLflow file store."""
        if not self.tracking_uri:
            return None
        parsed = urlparse(self.tracking_uri)
        if parsed.scheme and parsed.scheme not in {"file"}:
            return None
        if parsed.scheme == "file":
            return Path(parsed.path)
        return Path(self.tracking_uri)

    def _activate_fallback_store(self, mlflow: Any) -> bool:
        raise RuntimeError("Fallback MLflow store is disabled; fix the primary tracking URI")

    def _sanitize_tracking_store(self) -> bool:
        """Quarantine corrupt MLflow metadata to avoid file store crashes."""
        tracking_path = self._resolve_tracking_path()
        if tracking_path is None:
            return False
        tracking_path = tracking_path.expanduser()
        if not tracking_path.exists():
            return False
        quarantine_root = tracking_path.parent / f"{tracking_path.name}_quarantine"
        FileManager.ensure_dir(quarantine_root)
        had_corruption = False
        required_keys = {
            "artifact_location",
            "creation_time",
            "experiment_id",
            "last_update_time",
            "lifecycle_stage",
            "name",
        }

        for meta_path in tracking_path.rglob("meta.yaml"):
            if not meta_path.is_file():
                continue
            if self._is_mlflow_meta_valid(meta_path, required_keys=required_keys):
                continue
            self._quarantine_dir(meta_path.parent, quarantine_root=quarantine_root)
            had_corruption = True
        return had_corruption

    def _is_mlflow_meta_valid(
        self,
        meta_path: Path,
        *,
        required_keys: set[str],
    ) -> bool:
        """Execute is mlflow meta valid.



        Args:

            meta_path: Input value used by this callable.

            required_keys: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        try:
            raw = FileManager.read_text(meta_path).strip()
        except Exception:
            return False
        if not raw:
            return False
        try:
            payload = FileManager.read(meta_path)
            meta = payload.to_native() if isinstance(payload, ParquetBundle) else payload
        except Exception:
            return False
        if not isinstance(meta, dict):
            return False
        if not required_keys.issubset(meta.keys()):
            return False
        expected_id = meta_path.parent.name
        if str(meta.get("experiment_id")) != expected_id:
            return False
        return True

    def _quarantine_dir(self, corrupt_dir: Path, *, quarantine_root: Path) -> None:
        """Execute quarantine dir.



        Args:

            corrupt_dir: Input value used by this callable.

            quarantine_root: Input value used by this callable.

        """

        suffix = time.strftime("%Y%m%d%H%M%S")
        dest = quarantine_root / f"{corrupt_dir.name}_corrupt_{suffix}"
        try:
            shutil.move(str(corrupt_dir), dest)
            logger.warning(f"MLflow experiment quarantined: {corrupt_dir.name} -> {dest}")
        except Exception as exc:
            logger.warning(f"Quarantine failed for experiment {corrupt_dir.name}: {exc}")

    @contextmanager
    def start_run(self, run_name: str | None = None):
        """Context manager for MLflow run."""
        if not self.mlflow:
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
                self.mlflow.log_params(
                    {
                        "n_trials": n_trials,
                        "strategy": strategy_name,
                        "search_space_size": search_space_size,
                        "search_space_keys": len(search_space),
                    }
                )

                search_space_file = settings.OUTPUTS_DIR / "mlflow" / "search_space.json"
                FileManager.ensure_dir(search_space_file.parent)
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

    @staticmethod
    def _sanitize_metric_name(name: str) -> str:
        """Ensure metric names follow MLflow naming constraints."""
        if not name:
            return "metric"
        sanitized = re.sub(r"[^A-Za-z0-9_\-\. :/]", "_", str(name))
        sanitized = re.sub(r"_+", "_", sanitized).strip()
        return sanitized or "metric"

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
            with self.mlflow.start_run(
                run_name=f"trial_{trial.trial_number}",
                nested=True,
            ):
                params_to_log = {
                    k: v for k, v in trial.params.items() if isinstance(v, (int, float, str, bool))
                }
                params_to_log["state"] = trial.state
                params_to_log["trial_number"] = trial.trial_number
                self.mlflow.log_params(params_to_log)

                self.mlflow.log_metric("value", trial.value, step=trial_idx)
                if trial.user_attrs:
                    flattened = self._flatten_metrics(trial.user_attrs)
                    for key, value in flattened.items():
                        safe_key = self._sanitize_metric_name(key)
                        self.mlflow.log_metric(safe_key, value, step=trial_idx)

                if trial.intermediate_values:
                    for step, value in trial.intermediate_values.items():
                        self.mlflow.log_metric(f"intermediate_{step}", value, step=trial_idx)

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
            with self.mlflow.start_run(
                run_id=self.parent_run_id,
                nested=False,
            ):
                self.mlflow.log_metric("best_value", result.best_value)
                self.mlflow.log_metric("optimization_time_sec", result.optimization_time)

                n_completed = len([t for t in result.trials if t.state == "COMPLETE"])
                n_pruned = len([t for t in result.trials if t.state == "PRUNED"])
                n_failed = result.n_trials - n_completed - n_pruned

                self.mlflow.log_params(
                    {
                        "best_params": self.file_manager.json_dumps(
                            result.best_params, sort_keys=True
                        ),
                        "n_trials_completed": result.n_trials,
                        "framework": result.framework,
                        "n_completed": n_completed,
                        "n_pruned": n_pruned,
                        "n_failed": n_failed,
                    }
                )

                best_params_file = settings.OUTPUTS_DIR / "mlflow" / "best_params.json"
                try:
                    FileManager.ensure_dir(best_params_file.parent)
                    self.file_manager.save(result.best_params, best_params_file)
                    if best_params_file.exists():
                        self.mlflow.log_artifact(str(best_params_file), "best_params")
                except Exception as bp_exc:
                    logger.warning(f"Failed to save/log best_params artifact: {bp_exc}")

                try:
                    best_trial = next(
                        (t for t in result.trials if t.trial_number == result.best_trial_number),
                        None,
                    )
                    if best_trial and best_trial.user_attrs:
                        flattened = self._flatten_metrics(best_trial.user_attrs, prefix="best")
                        for key, value in flattened.items():
                            safe_key = self._sanitize_metric_name(key)
                            self.mlflow.log_metric(safe_key, value)
                except Exception as attr_exc:
                    logger.warning(f"Failed to log best trial metrics: {attr_exc}")

                logger.success(f"Otimização MLflow concluída: {result.best_value:.4f}")

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
                if self.file_manager.exists(path):
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
        if tracking_uri is None:
            return None
        if tracking_uri.startswith("file:"):
            mlflow_cfg = getattr(settings, "MLFLOW_CONFIG", {})
            return os.getenv(
                "MLFLOW_UI_URL",
                mlflow_cfg.get("ui_url", "http://localhost:5000"),
            )

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
        for _param_name, param_config in search_space.items():
            if isinstance(param_config, (list, tuple)):
                if len(param_config) == 2:
                    total *= 100
                else:
                    total *= len(param_config)
            elif isinstance(param_config, dict):
                param_type = param_config.get("type", "float")
                if param_type == "categorical":
                    total *= len(param_config.get("choices", []))
                else:
                    total *= 100
            else:
                total *= 100

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
                        "strategy": result.framework,
                        "best_value": result.best_value,
                        "n_trials": result.n_trials,
                        "optimization_time": result.optimization_time,
                    }
                )

            import polars as pl

            df = pl.DataFrame(comparison_data)
            comparison_file = settings.OUTPUTS_DIR / "mlflow" / "strategy_comparison.parquet"
            self.file_manager.save(df, comparison_file)

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
        if not self.mlflow or not model_path or not self.file_manager.exists(model_path):
            return

        try:
            self.mlflow.register_model(
                model_uri=str(model_path),
                name=model_name,
            )

            self.mlflow.log_params(result.best_params)

            logger.success(f"Modelo registrado no MLflow: {model_name}")

        except Exception as e:
            logger.warning(f"Failed to register model: {e}")

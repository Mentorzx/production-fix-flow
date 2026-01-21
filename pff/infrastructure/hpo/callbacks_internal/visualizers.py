"""Visualization callbacks for optimization."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from optuna.distributions import distribution_to_json
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.trial import TrialState

from pff.domain.learning.ml.training_observer import TrainingEvent, TrainingObserver
from pff.shared import logger
from pff.shared.core.config import settings
from pff.shared.core.file_manager import FileManager

from .collectors import flatten_trial_metrics


def _coerce_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _coerce_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        return str(value)


def _coerce_json_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _coerce_json_safe(v) for k, v in payload.items()}


def _serialize_search_space(trials: list) -> dict[str, Any]:
    """Serialize search space from trial distributions using Optuna API.

    Args:
        trials: List of Optuna FrozenTrial objects.

    Returns:
        Dict mapping parameter names to their distribution definitions.
    """
    if not trials:
        return {}
    search_space: dict[str, Any] = {}
    first_trial = trials[0]
    for name, dist in getattr(first_trial, "distributions", {}).items():
        try:
            search_space[name] = distribution_to_json(dist)
        except Exception:
            search_space[name] = {"type": type(dist).__name__}
    return search_space


class LiveTrainingObserver(TrainingObserver):
    """Observer that captures real-time training data for the dashboard."""

    def __init__(
        self,
        output_dir: Path,
        trial_number: int,
        params: dict[str, Any] | None = None,
        cv_fold_id: int | None = None,
    ):
        self.output_dir = output_dir
        self.status_path = output_dir / "live_status.json"
        self.trial_number = trial_number
        self.cv_fold_id = cv_fold_id
        self.params = params or {}
        self.start_time = time.time()
        self.epoch_history: list[dict[str, Any]] = []
        self.logs: list[dict[str, Any]] = []
        self.current_epoch = 0
        self.total_epochs = 0
        self._last_write = 0.0

        self._sink_id = logger.add(self._log_sink, level="INFO", format="{message}")
        self._write_status()

    def _log_sink(self, message):
        record = message.record
        self.logs.append(
            {
                "timestamp": record["time"].strftime("%H:%M:%S"),
                "level": record["level"].name,
                "message": record["message"],
            }
        )
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]

        if time.time() - self._last_write > 0.5:
            self._write_status()

    def on_event(self, event: TrainingEvent) -> None:
        if event.event_type == "training_start":
            config = event.metadata.get("config")
            if config:
                self.total_epochs = getattr(config, "epochs", 0)

        elif event.event_type == "epoch_end":
            self.current_epoch = event.epoch + 1
            metrics = event.metrics.copy()
            metrics["epoch"] = self.current_epoch
            metrics["timestamp"] = time.time()
            self.epoch_history.append(metrics)
            self._write_status()

        elif event.event_type == "training_end":
            self._write_status()
            if hasattr(self, "_sink_id"):
                logger.remove(self._sink_id)

    def _write_status(self):
        now = time.time()
        self._last_write = now
        elapsed = now - self.start_time

        status = {
            "trial_number": self.trial_number,
            "cv_fold_id": self.cv_fold_id,
            "params": self.params,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "elapsed_seconds": elapsed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "epoch_history": self.epoch_history,
            "recent_logs": self.logs,
            "progress": (
                (self.current_epoch / self.total_epochs * 100) if self.total_epochs > 0 else 0
            ),
        }

        if self.epoch_history:
            last = self.epoch_history[-1]
            status["elbo_recon"] = last.get("elbo_recon")
            status["elbo_kl"] = last.get("elbo_kl")
            status["kl_weight"] = last.get("kl_weight")
            status["pc2_rules"] = last.get("pc2_rules")
            status["pc2_contexts"] = last.get("pc2_contexts")
            status["pc2_latency"] = last.get("pc2_latency")
            status["pc2_density"] = last.get("pc2_density")
            status["structuralMetrics"] = {
                "latentEntropy": last.get("latentEntropy", 0.0),
                "communityOverlap": last.get("communityOverlap", 0.0),
                "graphDensity": last.get("graphDensity", 0.0),
                "numClusters": last.get("numClusters", 0),
            }

        try:
            FileManager().save(status, self.status_path)
        except Exception:
            pass


class RealTimeVisualizer:
    """Legacy visualizer stub. Kept for compatibility."""

    def __init__(self):
        pass

    def on_trial_complete(self, trial: Any, value: float) -> None:
        pass

    def close(self):
        pass


class LivePlotCallback:
    """Optuna callback that writes dashboard data JSON to disk."""

    def __init__(
        self,
        output_dir: Path,
        max_trials_axis: float = 50.0,
        expected_trials: int | None = None,
        enable_optuna_dashboard: bool = False,
        dashboard_interval: int = 5,
        dashboard_top_n: int = 12,
    ):
        self.output_dir = Path(output_dir)
        FileManager.ensure_dir(self.output_dir)
        output_dir_resolved = self.output_dir.resolve()
        outputs_root = settings.OUTPUTS_DIR.resolve()
        if output_dir_resolved.is_relative_to(outputs_root):
            self.cache_dir = settings.CACHE_DIR / "hpo"
        else:
            parents = self.output_dir.parents
            cache_root = parents[1] if len(parents) > 1 else self.output_dir
            self.cache_dir = cache_root / ".cache" / "hpo"
        FileManager.ensure_dir(self.cache_dir)

        self.data_path = self.cache_dir / "dashboard_data.json"

        self.dashboard_path = self.output_dir / "live_dashboard.html"
        self.status_path = self.output_dir / "live_status.json"

        self.dashboard_interval = dashboard_interval
        self.dashboard_top_n = dashboard_top_n
        self.expected_trials = expected_trials
        self._dashboard_last_update = 0.0

        self._initialize_data_file()

    def _initialize_data_file(self):
        """Create initial dashboard data file, clearing previous state if necessary."""

        payload = {
            "studyName": "Initializing Study...",
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "bestValue": 0,
            "trials": [],
            "totalTrials": self.expected_trials,
        }
        try:
            status_candidates = [
                self.status_path,
                self.output_dir.parent / "live_status.json",
                settings.OUTPUTS_DIR / "optimization" / "plots" / "live_status.json",
            ]
            for p in status_candidates:
                if p.exists():
                    p.unlink()
                    logger.info(f"Stale status file removed: {p}")

            FileManager().save(payload, self.data_path)
            logger.info(f"Dashboard data initialized at {self.data_path}")
        except Exception as e:
            logger.warning(f"Failed to init dashboard data: {e}")

    def __call__(self, study: Any, trial: Any) -> None:
        self._maybe_update_dashboard(study)

    def initialize_dashboard(self, study: Any) -> None:
        try:
            self._export_dashboard_data(study)
        except Exception as exc:
            logger.warning(f"Failed to initialize dashboard data: {exc}")

    def _maybe_update_dashboard(self, study: Any) -> None:
        now = time.monotonic()
        if now - self._dashboard_last_update < self.dashboard_interval:
            return
        self._dashboard_last_update = now
        try:
            self._export_dashboard_data(study)
        except Exception as exc:
            logger.warning(f"Failed to export dashboard data: {exc}")

    def _export_dashboard_data(self, study: Any) -> None:
        """Export study data to JSON for the dashboard."""
        if hasattr(study, "get_trials"):
            trials = list(study.get_trials(deepcopy=False))
        else:
            trials = list(getattr(study, "trials", []) or [])
        completed_trials = [t for t in trials if t.state == TrialState.COMPLETE]

        trials_data = []
        for t in trials:
            m = flatten_trial_metrics(t)
            primary_value = self._trial_primary_value(t)

            mrr = m.get("mrr", m.get("kge_mrr", m.get("best_val_mrr", 0.0)))
            if mrr == 0.0 and 0.0 < primary_value <= 1.0:
                mrr = primary_value

            mcc = m.get("mcc", 0.0)
            if mcc == 0.0 and -1.0 <= primary_value <= 1.0 and mrr != primary_value:
                mcc = primary_value

            duration = 0.0
            if t.datetime_complete and t.datetime_start:
                duration = (t.datetime_complete - t.datetime_start).total_seconds()
            if duration <= 0:
                duration = m.get("duration", 0.0)

            trials_data.append(
                {
                    "id": t.number + 1,
                    "value": primary_value,
                    "state": str(t.state.name),
                    "params": t.params if hasattr(t, "params") else {},
                    "duration": duration,
                    "mrr": mrr,
                    "best_mrr": max(mrr, m.get("best_mrr", 0.0)),
                    "mcc": mcc,
                    "auc": m.get("auc", 0.0),
                    "hits1": m.get("hits1", m.get("hits@1", 0.0)),
                    "hits3": m.get("hits3", m.get("hits@3", 0.0)),
                    "hits10": m.get("hits10", m.get("hits@10", 0.0)),
                    "inference_latency": m.get("inference_latency"),
                    "warmstart": bool(
                        t.system_attrs.get("warmstart_seed") or t.user_attrs.get("warmstart")
                    ),
                    "metrics": m,
                }
            )

        if completed_trials:
            try:
                best_value = float(study.best_value)
            except Exception:
                best_value = max(self._trial_primary_value(t) for t in completed_trials)
        else:
            best_value = 0.0

        study_name = str(getattr(study, "study_name", "optuna_study"))
        updated_at = datetime.now(timezone.utc).isoformat()

        study_attrs = getattr(study, "user_attrs", {})
        objective_name = study_attrs.get("objective_name", "Score")
        secondary_metric = study_attrs.get("multi_objective_secondary", "mcc")

        param_importances = {}
        if len(completed_trials) > 3:
            try:
                evaluator = FanovaImportanceEvaluator(n_trees=32, seed=42)
                importances = get_param_importances(study, evaluator=evaluator)
                param_importances = {k: float(v) for k, v in importances.items()}
            except Exception:
                pass

        payload = {
            "studyName": study_name,
            "updatedAt": updated_at,
            "bestValue": best_value,
            "trials": trials_data,
            "importances": param_importances,
            "totalTrials": self.expected_trials,
            "searchSpace": _serialize_search_space(completed_trials),
            "sampler": type(study.sampler).__name__ if hasattr(study, "sampler") else "Unknown",
            "direction": study.direction.name
            if hasattr(study, "direction") and hasattr(study.direction, "name")
            else "maximize",
            "objectiveName": objective_name,
            "secondaryMetric": secondary_metric,
        }

        try:
            FileManager().save(payload, self.data_path)
        except Exception as e:
            logger.debug(f"Failed to write dashboard data: {e}")

    @staticmethod
    def _trial_primary_value(trial: Any) -> float:
        value = getattr(trial, "value", None)
        if value is None:
            values = getattr(trial, "values", None)
            if isinstance(values, (list, tuple)) and values:
                value = values[0]
        try:
            return float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

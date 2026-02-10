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
        warmstart: bool = False,
    ):
        self.output_dir = output_dir
        self.status_path = output_dir / "live_status.json"
        self.fold_history_path = output_dir / "fold_history.json"
        self.trial_number = trial_number
        self.cv_fold_id = cv_fold_id
        self.params = params or {}
        self.warmstart = bool(warmstart)
        self.start_time = time.time()
        self.epoch_history: list[dict[str, Any]] = []
        self.logs: list[dict[str, Any]] = []
        self.current_epoch = 0
        self.total_epochs = 0
        self._last_write = 0.0
        self._last_epoch_elapsed = 0.0

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
            elapsed = time.time() - self.start_time
            duration = metrics.get("duration")
            if duration is None:
                duration = max(elapsed - self._last_epoch_elapsed, 0.0)
            metrics.setdefault("duration", duration)
            metrics.setdefault("elapsed_seconds", elapsed)
            # Normalize Train Loss
            if "train_loss" not in metrics:
                metrics["train_loss"] = (
                    metrics.get("loss")
                    or metrics.get("binary_loss")
                    or metrics.get("train_binary_loss")
                )

            # Normalize Val Loss
            if "val_loss" not in metrics:
                metrics["val_loss"] = (
                    metrics.get("eval_loss")
                    or metrics.get("val_binary_loss")
                    or metrics.get("binary_loss")
                    or metrics.get("test_loss")
                )

            if "loss" not in metrics:
                metrics["loss"] = (
                    metrics.get("train_loss")
                    or metrics.get("val_loss")
                    or metrics.get("binary_loss")
                )
            score_candidate = (
                metrics.get("score")
                or metrics.get("mrr")
                or metrics.get("mcc")
                or metrics.get("accuracy")
            )
            if score_candidate is not None and duration:
                try:
                    metrics["efficiency"] = float(score_candidate) / float(duration)
                except (TypeError, ValueError):
                    pass
            metrics["epoch"] = self.current_epoch
            metrics["timestamp"] = time.time()
            self.epoch_history.append(metrics)
            self._last_epoch_elapsed = elapsed
            self._write_status()

        elif event.event_type == "training_end":
            self._write_status()
            self._save_fold_to_history()
            if hasattr(self, "_sink_id"):
                logger.remove(self._sink_id)

    def _save_fold_to_history(self):
        """Save completed fold data to history file for dashboard to show multiple folds."""
        try:
            # Load existing history
            history = []
            if self.fold_history_path.exists():
                try:
                    existing = FileManager().read(self.fold_history_path)
                    if isinstance(existing, list):
                        history = existing
                except Exception:
                    pass

            # Get last confusion matrix from epoch history
            cm = None
            last_epoch = None
            if self.epoch_history:
                last = self.epoch_history[-1]
                last_val = next(
                    (
                        e
                        for e in reversed(self.epoch_history)
                        if ("vp" in e) or ("tp" in e) or ("fp" in e) or ("fn" in e)
                    ),
                    last,
                )

                def _get_cm_val(key: str, alt: str) -> int:
                    val = last_val.get(key)
                    if val is None:
                        val = last_val.get(alt)
                    try:
                        return int(val) if val is not None else 0
                    except (TypeError, ValueError):
                        return 0

                cm = {
                    "vp": _get_cm_val("vp", "tp"),
                    "vn": _get_cm_val("vn", "tn"),
                    "fp": _get_cm_val("fp", "fp"),
                    "fn": _get_cm_val("fn", "fn"),
                }
                last_epoch = last_val.get("epoch", self.current_epoch)

            if cm:
                # Add current fold to history
                entry = {
                    "trial_number": self.trial_number,
                    "cv_fold_id": self.cv_fold_id,
                    "epoch": last_epoch,
                    "timestamp": time.time(),
                    "confusion_matrix": cm,
                    "params": self.params,
                }
                history.append(entry)

                # Keep only last 10 folds to prevent file from growing too large
                history = history[-10:]

                FileManager().save(history, self.fold_history_path)
        except Exception:
            pass

    def _write_status(self):
        now = time.time()
        self._last_write = now
        elapsed = now - self.start_time

        status = {
            "trial_number": self.trial_number,
            "cv_fold_id": self.cv_fold_id,
            "params": self.params,
            "warmstart": self.warmstart,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "elapsed_seconds": elapsed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "epoch_history": self.epoch_history,
            "recent_logs": self.logs,
            "progress": (
                (self.current_epoch / self.total_epochs * 100)
                if self.total_epochs > 0
                else 0
            ),
        }

        if self.epoch_history:
            last = self.epoch_history[-1]
            last_val = next(
                (
                    e
                    for e in reversed(self.epoch_history)
                    if ("vp" in e) or ("tp" in e) or ("fp" in e) or ("fn" in e)
                ),
                last,
            )

            def _get_cm_val(key: str, alt: str) -> int:
                val = last_val.get(key)
                if val is None:
                    val = last_val.get(alt)
                try:
                    return int(val) if val is not None else 0
                except (TypeError, ValueError):
                    return 0

            cm = {
                "vp": _get_cm_val("vp", "tp"),
                "vn": _get_cm_val("vn", "tn"),
                "fp": _get_cm_val("fp", "fp"),
                "fn": _get_cm_val("fn", "fn"),
            }
            status["elbo_recon"] = last.get("elbo_recon")
            status["elbo_kl"] = last.get("elbo_kl")
            status["kl_weight"] = last.get("kl_weight")
            status["pc2_rules"] = last.get("pc2_rules")
            status["pc2_contexts"] = last.get("pc2_contexts")
            status["pc2_latency"] = last.get("pc2_latency")
            status["pc2_density"] = last.get("pc2_density")
            status["confusion_matrix"] = cm
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
        dashboard_data_path: Path | None = None,
    ):
        self.output_dir = Path(output_dir)
        FileManager.ensure_dir(self.output_dir)
        output_dir_resolved = self.output_dir.resolve()
        outputs_root = settings.OUTPUTS_DIR.resolve()
        if dashboard_data_path is not None:
            resolved_path = Path(dashboard_data_path)
            if not resolved_path.is_absolute():
                resolved_path = settings.ROOT_DIR / resolved_path
            self.data_path = resolved_path
            self.cache_dir = self.data_path.parent
        else:
            if output_dir_resolved.is_relative_to(outputs_root):
                self.cache_dir = settings.CACHE_DIR / "hpo"
            else:
                parents = self.output_dir.parents
                cache_root = parents[1] if len(parents) > 1 else self.output_dir
                self.cache_dir = cache_root / ".cache" / "hpo"
            self.data_path = self.cache_dir / "dashboard_data.json"
        FileManager.ensure_dir(self.cache_dir)

        self.dashboard_path = self.output_dir / "live_dashboard.html"
        self.status_path = self.output_dir / "live_status.json"

        self.dashboard_interval = dashboard_interval
        self.dashboard_top_n = dashboard_top_n
        self.expected_trials = expected_trials
        self._dashboard_last_update = 0.0

        self._initialize_data_file()

    def _initialize_data_file(self):
        """Ensure the dashboard data file exists.

        This must be non-destructive: if a previous dashboard_data.json exists with trials,
        we keep it to avoid wiping historical metrics on resume/restart.
        """

        if self.data_path.exists():
            try:
                existing = FileManager().read(self.data_path)
                existing_native = (
                    existing.to_native() if hasattr(existing, "to_native") else existing
                )
                if (
                    isinstance(existing_native, dict)
                    and isinstance(existing_native.get("trials"), list)
                    and existing_native["trials"]
                ):
                    return
            except Exception:
                pass

        payload = {
            "studyName": "Initializing Study...",
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "bestValue": 0,
            "trials": [],
            "totalTrials": self.expected_trials,
        }
        try:
            FileManager().save(payload, self.data_path)
            logger.info(f"Dados do dashboard inicializados em {self.data_path}")
            mirror_path = self.output_dir / "dashboard_data.json"
            if mirror_path != self.data_path:
                FileManager().save(payload, mirror_path)
        except Exception as e:
            timestamp = datetime.now(timezone.utc).isoformat()
            logger.warning(
                f"timestamp={timestamp} component_name=hpo_dashboard stop_reason=init_dashboard_data_failed key_parameters={{'file': str(self.data_path)}} message='Failed to init dashboard data: {e}'"
            )

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
        """Export study data to JSON for dashboard."""
        if hasattr(study, "get_trials"):
            trials = list(study.get_trials(deepcopy=False))
        else:
            trials = list(getattr(study, "trials", []) or [])

        logger.debug(
            "component_name=hpo_dashboard key_parameters={'total_trials': len(trials)} message='Trials found in study'"
        )

        for t in trials:
            logger.debug(
                "component_name=hpo_dashboard key_parameters={'trial_id': t.number, 'state': str(t.state), 'value': t.value} message='Trial details'"
            )

        completed_trials = [t for t in trials if t.state == TrialState.COMPLETE]

        trials_data = []
        max_trial_id = max([t.number for t in trials]) if trials else -1

        for t in trials:
            m = flatten_trial_metrics(t)
            primary_value = self._trial_primary_value(t)
            t_state = str(t.state.name)

            if t_state == "RUNNING" and t.number < max_trial_id:
                t_state = "PRUNED"

            user_attrs = t.user_attrs

            mrr = m.get("mrr", m.get("kge_mrr", m.get("best_val_mrr", 0.0)))
            if mrr == 0.0 and 0.0 < primary_value <= 1.0:
                mrr = primary_value

            best_mrr = user_attrs.get(
                "best_val_mrr", user_attrs.get("best_mrr", m.get("best_mrr"))
            )

            mcc = m.get("mcc", user_attrs.get("mcc"))

            best_mcc = user_attrs.get(
                "best_val_mcc", user_attrs.get("best_mcc", m.get("best_mcc"))
            )

            duration = 0.0
            if t.datetime_complete and t.datetime_start:
                duration = (t.datetime_complete - t.datetime_start).total_seconds()
            if duration <= 0:
                duration = m.get("duration", 0.0)

            loss_value = (
                m.get("loss")
                or m.get("val_loss")
                or m.get("train_loss")
                or m.get("binary_loss")
            )
            if loss_value is not None:
                m.setdefault("loss", loss_value)
            m.setdefault("duration", duration)
            efficiency = None
            if duration:
                try:
                    efficiency = float(primary_value) / float(duration)
                    m.setdefault("efficiency", efficiency)
                except (TypeError, ValueError):
                    efficiency = None

            trials_data.append(
                {
                    "id": t.number + 1,
                    "value": primary_value,
                    "state": t_state,
                    "params": t.params if hasattr(t, "params") else {},
                    "duration": duration,
                    "loss": loss_value,
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "efficiency": efficiency,
                    "mrr": mrr,
                    "best_mrr": float(best_mrr) if best_mrr is not None else None,
                    "mcc": mcc,
                    "best_mcc": float(best_mcc) if best_mcc is not None else None,
                    "auc": m.get("auc"),
                    "hits1": m.get("hits1", m.get("hits@1", user_attrs.get("hits@1"))),
                    "hits3": m.get("hits3", m.get("hits@3", user_attrs.get("hits@3"))),
                    "hits10": m.get(
                        "hits10", m.get("hits@10", user_attrs.get("hits@10"))
                    ),
                    "inference_latency": m.get("inference_latency"),
                    "warmstart": bool(
                        t.system_attrs.get("warmstart_seed")
                        or t.user_attrs.get("warmstart")
                        or t.user_attrs.get("warmstart_seed")
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

        live_status = {}
        status_candidates = [
            self.status_path,
            settings.OUTPUTS_DIR / "optimization" / "plots" / "live_status.json",
        ]
        for candidate in status_candidates:
            try:
                if candidate.exists():
                    payload = FileManager().read(candidate)
                    live_status = (
                        payload.to_native()
                        if hasattr(payload, "to_native")
                        else payload
                    )
                    logger.debug(
                        f"component_name=hpo_dashboard message='Loaded live_status from {candidate}'"
                    )
                    break
            except Exception:
                pass

        live_history_best = {}
        if live_status.get("trial_number") is not None:
            hist = live_status.get("epoch_history", [])
            if hist:
                live_history_best = {
                    "mrr": max((e.get("mrr", 0.0) for e in hist), default=0.0),
                    "mcc": max((e.get("mcc", 0.0) for e in hist), default=0.0),
                    "id": live_status["trial_number"],
                }

        for i, t in enumerate(trials):
            if t.number == live_history_best.get("id"):
                pass

        charts = {}
        confusion_matrices: list[dict[str, Any]] = []

        # Current live fold identifiers
        current_trial = live_status.get("trial_number") if live_status else None
        current_fold = live_status.get("cv_fold_id") if live_status else None

        # Load fold history — previous completed folds (excludes current live fold)
        fold_history_path = (
            settings.OUTPUTS_DIR / "optimization" / "plots" / "fold_history.json"
        )
        if fold_history_path.exists():
            try:
                history_data = FileManager().read(fold_history_path)
                if isinstance(history_data, list):
                    # Deduplicate by trial:fold keeping latest entry per combo
                    seen: dict[str, dict[str, Any]] = {}
                    for entry in history_data:
                        if not isinstance(entry, dict) or not entry.get(
                            "confusion_matrix"
                        ):
                            continue
                        t_num = entry.get("trial_number")
                        f_id = entry.get("cv_fold_id")
                        # Skip entries matching the current live fold
                        if t_num == current_trial and f_id == current_fold:
                            continue
                        combo = f"{t_num}:{f_id}"
                        seen[combo] = entry

                    for entry in list(seen.values())[-2:]:
                        confusion_matrices.append(
                            {
                                "timestamp": entry.get("timestamp"),
                                "epoch": entry.get("epoch"),
                                "trial_number": entry.get("trial_number"),
                                "cv_fold_id": entry.get("cv_fold_id"),
                                "confusion_matrix": entry["confusion_matrix"],
                            }
                        )
            except Exception:
                pass

        # Add ONLY the latest validation event for the current live fold
        if live_status.get("epoch_history"):
            epoch_history = live_status["epoch_history"]
            if epoch_history:
                val_events = [
                    e
                    for e in epoch_history
                    if isinstance(e, dict)
                    and (("vp" in e) or ("tp" in e) or ("fp" in e) or ("fn" in e))
                ]

                def _get_cm_val(obj: dict[str, Any], key: str, alt: str) -> int:
                    v = obj.get(key)
                    if v is None:
                        v = obj.get(alt)
                    try:
                        return int(v) if v is not None else 0
                    except (TypeError, ValueError):
                        return 0

                if val_events:
                    last_val = val_events[-1]
                    cm = {
                        "vp": _get_cm_val(last_val, "vp", "tp"),
                        "vn": _get_cm_val(last_val, "vn", "tn"),
                        "fp": _get_cm_val(last_val, "fp", "fp"),
                        "fn": _get_cm_val(last_val, "fn", "fn"),
                    }
                    charts["confusion_matrix"] = cm

                    confusion_matrices.append(
                        {
                            "timestamp": last_val.get("timestamp"),
                            "epoch": last_val.get("epoch"),
                            "trial_number": current_trial,
                            "cv_fold_id": current_fold,
                            "confusion_matrix": cm,
                        }
                    )

        param_importances = {}
        if len(completed_trials) > 3:
            try:
                evaluator = FanovaImportanceEvaluator(n_trees=32, seed=42)
                importances = get_param_importances(study, evaluator=evaluator)
                param_importances = {k: float(v) for k, v in importances.items()}
            except Exception:
                pass

        trials_data = []
        max_trial_id = max([t.number for t in trials]) if trials else -1

        for t in trials:
            m = flatten_trial_metrics(t)
            primary_value = self._trial_primary_value(t)
            t_state = str(t.state.name)

            if t_state == "RUNNING" and t.number < max_trial_id:
                t_state = "PRUNED"

            mrr = m.get("mrr", m.get("kge_mrr", m.get("best_val_mrr", 0.0)))
            if mrr == 0.0 and 0.0 < primary_value <= 1.0:
                mrr = primary_value

            user_attrs = t.user_attrs

            best_mrr = user_attrs.get(
                "best_val_mrr", user_attrs.get("best_mrr", m.get("best_mrr"))
            )

            if best_mrr is None and t.number == live_history_best.get("id"):
                best_mrr = live_history_best["mrr"]

            mcc = m.get("mcc", user_attrs.get("mcc"))

            best_mcc = user_attrs.get(
                "best_val_mcc", user_attrs.get("best_mcc", m.get("best_mcc"))
            )
            if best_mcc is None and t.number == live_history_best.get("id"):
                best_mcc = live_history_best["mcc"]

            duration = 0.0
            if t.datetime_complete and t.datetime_start:
                duration = (t.datetime_complete - t.datetime_start).total_seconds()
            if duration <= 0:
                duration = m.get("duration", 0.0)

            loss_value = m.get("loss") or m.get("val_loss") or m.get("train_loss")
            if loss_value is not None:
                m.setdefault("loss", loss_value)
            m.setdefault("duration", duration)
            efficiency = None
            if duration:
                try:
                    efficiency = float(primary_value) / float(duration)
                    m.setdefault("efficiency", efficiency)
                except (TypeError, ValueError):
                    efficiency = None

            trials_data.append(
                {
                    "id": t.number + 1,
                    "value": primary_value,
                    "state": t_state,
                    "params": t.params if hasattr(t, "params") else {},
                    "duration": duration,
                    "loss": loss_value,
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "efficiency": efficiency,
                    "mrr": mrr,
                    "best_mrr": float(best_mrr) if best_mrr is not None else None,
                    "mcc": mcc,
                    "best_mcc": float(best_mcc) if best_mcc is not None else None,
                    "auc": m.get("auc"),
                    "hits1": m.get("hits1", m.get("hits@1", user_attrs.get("hits@1"))),
                    "hits3": m.get("hits3", m.get("hits@3", user_attrs.get("hits@3"))),
                    "hits10": m.get(
                        "hits10", m.get("hits@10", user_attrs.get("hits@10"))
                    ),
                    "inference_latency": m.get("inference_latency"),
                    "warmstart": bool(
                        t.system_attrs.get("warmstart_seed")
                        or t.user_attrs.get("warmstart")
                        or t.user_attrs.get("warmstart_seed")
                    ),
                    "metrics": m,
                }
            )

        payload = {
            "studyName": study_name,
            "updatedAt": updated_at,
            "bestValue": best_value,
            "trials": trials_data,
            "importances": param_importances,
            "totalTrials": self.expected_trials,
            "searchSpace": _serialize_search_space(completed_trials),
            "sampler": (
                type(study.sampler).__name__ if hasattr(study, "sampler") else "Unknown"
            ),
            "direction": (
                study.direction.name
                if hasattr(study, "direction") and hasattr(study.direction, "name")
                else "maximize"
            ),
            "objectiveName": objective_name,
            "secondaryMetric": secondary_metric,
            "liveStatus": live_status,
            "charts": charts,
        }

        if confusion_matrices:
            payload["charts"]["confusion_matrices"] = confusion_matrices

        try:
            FileManager().save(payload, self.data_path)
            logger.debug(
                "component_name=hpo_dashboard key_parameters={'trials_count': len(trials_data), 'file': str(self.data_path)} message='Dashboard data written successfully'"
            )
            mirror_path = self.output_dir / "dashboard_data.json"
            if mirror_path != self.data_path:
                FileManager().save(payload, mirror_path)
        except Exception as e:
            timestamp = datetime.now(timezone.utc).isoformat()
            logger.warning(
                f"timestamp={timestamp} component_name=hpo_dashboard stop_reason=write_dashboard_data_failed key_parameters={{'file': str(self.data_path)}} message='Failed to write dashboard data: {e}'"
            )

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

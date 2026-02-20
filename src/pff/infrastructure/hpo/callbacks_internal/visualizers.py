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
    """Execute coerce json safe.



    Args:

        value: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

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


def _merge_fold_history_entries(
    existing: list[dict[str, Any]],
    incoming: dict[str, Any],
    *,
    max_entries: int = 10,
) -> list[dict[str, Any]]:
    """Merge a fold-history entry without losing previous folds."""
    merged: dict[tuple[Any, Any], dict[str, Any]] = {}

    def _safe_key(row: dict[str, Any], idx: int) -> tuple[Any, Any]:
        trial = row.get("trial_number")
        fold = row.get("cv_fold_id")
        if trial is not None and fold is not None:
            return (trial, fold)
        return (f"unknown-{idx}", row.get("timestamp", idx))

    for idx, row in enumerate(existing):
        if not isinstance(row, dict) or not isinstance(row.get("confusion_matrix"), dict):
            continue
        merged[_safe_key(row, idx)] = row

    next_key = _safe_key(incoming, len(merged))
    previous = merged.get(next_key)
    if previous is None:
        merged[next_key] = incoming
    else:
        prev_epoch = float(previous.get("epoch") or -1)
        new_epoch = float(incoming.get("epoch") or -1)
        prev_ts = float(previous.get("timestamp") or -1)
        new_ts = float(incoming.get("timestamp") or -1)
        merged[next_key] = incoming if (new_epoch, new_ts) >= (prev_epoch, prev_ts) else previous

    items = list(merged.values())
    items.sort(key=lambda row: (float(row.get("timestamp") or 0.0), int(row.get("epoch") or 0)))
    return items[-max_entries:]


def _serialize_search_space(trials: list) -> tuple[dict[str, Any], dict[str, Any]]:
    """Serialize search space from all trial distributions and compute coverage metadata."""
    if not trials:
        return {}, {
            "search_space_coverage_ratio": 0.0,
            "missing_params": [],
            "distribution_conflicts": [],
        }

    search_space: dict[str, Any] = {}
    conflict_map: dict[str, set[str]] = {}
    all_trial_params: set[str] = set()

    for trial in trials:
        params = getattr(trial, "params", {}) or {}
        if isinstance(params, dict):
            all_trial_params.update(str(k) for k in params)
        for name, dist in getattr(trial, "distributions", {}).items():
            param_name = str(name)
            try:
                serialized = distribution_to_json(dist)
            except Exception:
                serialized = {"type": type(dist).__name__}
            if param_name not in search_space:
                search_space[param_name] = serialized
                continue
            if search_space[param_name] != serialized:
                conflict_map.setdefault(param_name, set()).update(
                    {
                        FileManager.json_dumps(search_space[param_name]),
                        FileManager.json_dumps(serialized),
                    }
                )

    for param_name in sorted(all_trial_params):
        if param_name in search_space:
            continue
        values = [
            trial.params.get(param_name)
            for trial in trials
            if isinstance(getattr(trial, "params", {}), dict) and param_name in trial.params
        ]
        values = [v for v in values if v is not None]
        if not values:
            continue
        if all(isinstance(v, (int, float)) for v in values):
            low = min(float(v) for v in values)
            high = max(float(v) for v in values)
            if abs(high - low) <= 1e-12:
                search_space[param_name] = {"type": "fixed", "value": low}
            else:
                all_int = all(isinstance(v, int) for v in values)
                search_space[param_name] = {
                    "type": "int" if all_int else "float",
                    "low": int(low) if all_int else low,
                    "high": int(high) if all_int else high,
                }
        else:
            choices = sorted({str(v) for v in values})
            search_space[param_name] = {"type": "categorical", "choices": choices[:32]}

    missing_params = sorted(p for p in all_trial_params if p not in search_space)
    covered_params = len([p for p in all_trial_params if p in search_space])
    coverage_ratio = (
        float(covered_params) / float(max(1, len(all_trial_params))) if all_trial_params else 1.0
    )
    distribution_conflicts = sorted(conflict_map.keys())
    coverage_meta = {
        "search_space_coverage_ratio": round(float(coverage_ratio), 4),
        "missing_params": missing_params,
        "distribution_conflicts": distribution_conflicts,
    }
    return search_space, coverage_meta


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
        """Execute init.



        Args:

            output_dir: Input value used by this callable.

            trial_number: Input value used by this callable.

            params: Optional input value.

            cv_fold_id: Optional input value.

            warmstart: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

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

    def _fold_history_targets(self) -> list[Path]:
        """Return all fold-history targets (local + canonical dashboard path)."""
        targets: list[Path] = [self.fold_history_path]
        canonical = settings.OUTPUTS_DIR / "optimization" / "plots" / "fold_history.json"
        if canonical not in targets:
            targets.append(canonical)
        return targets

    def _log_sink(self, message):
        """Execute log sink.



        Args:

            message: Input value used by this callable.

        """

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
        """Execute on event.



        Args:

            event: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if event.event_type == "training_start":
            self._handle_training_start(event)
            return
        if event.event_type == "epoch_end":
            self._handle_epoch_end(event)
            return
        if event.event_type == "training_end":
            self._handle_training_end()

    def _handle_training_start(self, event: TrainingEvent) -> None:
        """Execute handle training start.



        Args:

            event: Input value used by this callable.

        """

        config = event.metadata.get("config")
        if config:
            self.total_epochs = getattr(config, "epochs", 0)

    def _handle_epoch_end(self, event: TrainingEvent) -> None:
        """Execute handle epoch end.



        Args:

            event: Input value used by this callable.

        """

        self.current_epoch = event.epoch + 1
        metrics = event.metrics.copy()
        elapsed = time.time() - self.start_time
        duration = self._resolve_epoch_duration(metrics, elapsed)
        self._normalize_loss_metrics(metrics)
        self._add_efficiency_metric(metrics, duration)
        metrics["epoch"] = self.current_epoch
        metrics["timestamp"] = time.time()
        self.epoch_history.append(metrics)
        self._last_epoch_elapsed = elapsed
        self._write_status()

    def _handle_training_end(self) -> None:
        """Execute handle training end."""

        self._write_status()
        self._save_fold_to_history()
        if hasattr(self, "_sink_id"):
            logger.remove(self._sink_id)

    def _resolve_epoch_duration(self, metrics: dict[str, Any], elapsed: float) -> float:
        """Execute resolve epoch duration.



        Args:

            metrics: Input value used by this callable.

            elapsed: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        duration = metrics.get("duration")
        if duration is None:
            duration = max(elapsed - self._last_epoch_elapsed, 0.0)
        metrics.setdefault("duration", duration)
        metrics.setdefault("elapsed_seconds", elapsed)
        return float(duration)

    def _normalize_loss_metrics(self, metrics: dict[str, Any]) -> None:
        """Execute normalize loss metrics.



        Args:

            metrics: Input value used by this callable.

        """

        if metrics.get("train_loss") is None:
            metrics["train_loss"] = (
                metrics.get("loss")
                or metrics.get("binary_loss")
                or metrics.get("train_binary_loss")
            )
        if metrics.get("val_loss") is None:
            metrics["val_loss"] = (
                metrics.get("eval_loss")
                or metrics.get("val_binary_loss")
                or metrics.get("test_loss")
            )
        if metrics.get("val_loss") is None and self._has_validation_signals(metrics):
            metrics["val_loss"] = metrics.get("binary_loss")
        if metrics.get("loss") is None:
            metrics["loss"] = (
                metrics.get("train_loss") or metrics.get("val_loss") or metrics.get("binary_loss")
            )

    @staticmethod
    def _has_validation_signals(metrics: dict[str, Any]) -> bool:
        """Return True when metrics indicate an evaluation/validation epoch."""
        eval_keys = (
            "mrr",
            "mcc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "pr_auc",
            "hits@1",
            "hits@3",
            "hits@10",
            "hits1",
            "hits3",
            "hits10",
            "tp",
            "vp",
            "tn",
            "vn",
            "fp",
            "fn",
            "decision_threshold",
        )
        for key in eval_keys:
            if metrics.get(key) is not None:
                return True
        return False

    @staticmethod
    def _add_efficiency_metric(metrics: dict[str, Any], duration: float) -> None:
        """Execute add efficiency metric.



        Args:

            metrics: Input value used by this callable.

            duration: Input value used by this callable.

        """

        score_candidate = (
            metrics.get("score")
            or metrics.get("mrr")
            or metrics.get("mcc")
            or metrics.get("accuracy")
        )
        if score_candidate is None or not duration:
            return
        try:
            metrics["efficiency"] = float(score_candidate) / float(duration)
        except (TypeError, ValueError):
            return

    def _save_fold_to_history(self):
        """Save completed fold data to history file for dashboard to show multiple folds."""
        try:
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
                    """Execute get cm val.



                    Args:

                        key: Input value used by this callable.

                        alt: Input value used by this callable.



                    Returns:

                        Return value produced by the callable.

                    """

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
                entry = {
                    "trial_number": self.trial_number,
                    "cv_fold_id": self.cv_fold_id,
                    "epoch": last_epoch,
                    "timestamp": time.time(),
                    "confusion_matrix": cm,
                    "params": self.params,
                }
                for target_path in self._fold_history_targets():
                    history = []
                    if target_path.exists():
                        try:
                            existing = FileManager().read(target_path)
                            if hasattr(existing, "to_native"):
                                existing = existing.to_native()
                            if isinstance(existing, list):
                                history = existing
                        except Exception as exc:
                            logger.warning(
                                f"Failed to read fold history safely from {target_path}: {exc}"
                            )
                            continue
                    history = _merge_fold_history_entries(history, entry, max_entries=10)
                    FileManager().save(history, target_path)
        except Exception:
            pass

    def _write_status(self):
        """Execute write status.



        Returns:

            Return value produced by the callable.

        """

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
                (self.current_epoch / self.total_epochs * 100) if self.total_epochs > 0 else 0
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
                """Execute get cm val.



                Args:

                    key: Input value used by this callable.

                    alt: Input value used by this callable.



                Returns:

                    Return value produced by the callable.

                """

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
        """Execute init."""

        pass

    def on_trial_complete(self, trial: Any, value: float) -> None:
        """Execute on trial complete.



        Args:

            trial: Input value used by this callable.

            value: Input value used by this callable.

        """

        pass

    def close(self):
        """Execute close."""

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
        """Execute init.



        Args:

            output_dir: Input value used by this callable.

            max_trials_axis: Optional input value.

            expected_trials: Optional input value.

            enable_optuna_dashboard: Optional input value.

            dashboard_interval: Optional input value.

            dashboard_top_n: Optional input value.

            dashboard_data_path: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

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
                "timestamp={timestamp} component_name=hpo_dashboard stop_reason=init_dashboard_data_failed key_parameters=file={file_path!r} message='Failed to init dashboard data: {error}'",
                timestamp=timestamp,
                file_path=str(self.data_path),
                error=e,
            )

    def __call__(self, study: Any, trial: Any) -> None:
        self._maybe_update_dashboard(study)

    def initialize_dashboard(self, study: Any) -> None:
        """Execute initialize dashboard.



        Args:

            study: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        try:
            self._export_dashboard_data(study)
        except Exception as exc:
            logger.warning(f"Failed to initialize dashboard data: {exc}")

    def _maybe_update_dashboard(self, study: Any) -> None:
        """Execute maybe update dashboard.



        Args:

            study: Input value used by this callable.

        """

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
        trials = self._collect_trials(study)
        self._log_trials(trials)
        completed_trials = [t for t in trials if t.state == TrialState.COMPLETE]
        best_value = self._resolve_best_value(study, completed_trials)
        study_name = str(getattr(study, "study_name", "optuna_study"))
        updated_at = datetime.now(timezone.utc).isoformat()
        objective_name, secondary_metric = self._resolve_objective_labels(study)
        live_status = self._load_live_status_payload()
        live_history_best = self._resolve_live_history_best(live_status)
        charts, confusion_matrices = self._collect_chart_payloads(live_status)
        param_importances = self._compute_param_importances(study, completed_trials)
        trials_data = self._build_trials_data(trials, live_history_best, study=study)
        serialized_search_space, search_space_coverage = _serialize_search_space(completed_trials)
        payload: dict[str, Any] = {
            "studyName": study_name,
            "updatedAt": updated_at,
            "bestValue": best_value,
            "trials": trials_data,
            "importances": param_importances,
            "totalTrials": self.expected_trials,
            "searchSpace": serialized_search_space,
            "searchSpaceCoverage": search_space_coverage,
            "sampler": (type(study.sampler).__name__ if hasattr(study, "sampler") else "Unknown"),
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

        search_space_advice = self._compute_search_space_advice(
            search_space=payload["searchSpace"],
            trials_data=trials_data,
            importances=param_importances,
            direction=payload["direction"],
            study_name=study_name,
            study=study,
            search_space_coverage=search_space_coverage,
        )
        if search_space_advice:
            payload["searchSpaceAdvice"] = search_space_advice

        self._save_dashboard_payload(payload, len(trials_data))

    def _collect_trials(self, study: Any) -> list[Any]:
        """Execute collect trials.



        Args:

            study: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if hasattr(study, "get_trials"):
            return list(study.get_trials(deepcopy=False))
        return list(getattr(study, "trials", []) or [])

    @staticmethod
    def _log_trials(trials: list[Any]) -> None:
        """Execute log trials.



        Args:

            trials: Input value used by this callable.

        """

        logger.debug(
            "component_name=hpo_dashboard key_parameters=total_trials={total_trials} message='Trials found in study'",
            total_trials=len(trials),
        )
        for trial in trials:
            logger.debug(
                "component_name=hpo_dashboard key_parameters=trial_id={trial_id}, state={state!r}, value={value} message='Trial details'",
                trial_id=getattr(trial, "number", None),
                state=str(getattr(trial, "state", "unknown")),
                value=getattr(trial, "value", None),
            )

    def _resolve_best_value(self, study: Any, completed_trials: list[Any]) -> float:
        """Execute resolve best value.



        Args:

            study: Input value used by this callable.

            completed_trials: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not completed_trials:
            return 0.0
        try:
            return float(study.best_value)
        except Exception:
            return max(self._trial_primary_value(t) for t in completed_trials)

    @staticmethod
    def _resolve_objective_labels(study: Any) -> tuple[str, str]:
        """Execute resolve objective labels.



        Args:

            study: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        study_attrs = getattr(study, "user_attrs", {})
        objective_name = study_attrs.get("objective_name", "Score")
        secondary_metric = study_attrs.get("multi_objective_secondary", "mcc")
        return objective_name, secondary_metric

    def _load_live_status_payload(self) -> dict[str, Any]:
        """Execute load live status payload.



        Returns:

            Return value produced by the callable.

        """

        status_candidates = [
            self.status_path,
            settings.OUTPUTS_DIR / "optimization" / "plots" / "live_status.json",
        ]
        for candidate in status_candidates:
            try:
                if not candidate.exists():
                    continue
                payload = FileManager().read(candidate)
                live_status = payload.to_native() if hasattr(payload, "to_native") else payload
                logger.debug(
                    f"component_name=hpo_dashboard message='Loaded live_status from {candidate}'"
                )
                if isinstance(live_status, dict):
                    return live_status
            except Exception:
                continue
        return {}

    @staticmethod
    def _resolve_live_history_best(live_status: dict[str, Any]) -> dict[str, Any]:
        """Execute resolve live history best.



        Args:

            live_status: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if live_status.get("trial_number") is None:
            return {}
        hist = live_status.get("epoch_history", [])
        if not hist:
            return {}
        return {
            "mrr": max((e.get("mrr", 0.0) for e in hist), default=0.0),
            "mcc": max((e.get("mcc", 0.0) for e in hist), default=0.0),
            "id": live_status["trial_number"],
        }

    def _collect_chart_payloads(
        self, live_status: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Execute collect chart payloads.



        Args:

            live_status: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        charts: dict[str, Any] = {}
        confusion_matrices = self._load_fold_history_confusion_matrices(live_status)
        latest_confusion = self._extract_latest_live_confusion(live_status)
        if latest_confusion is not None:
            charts["confusion_matrix"] = latest_confusion["confusion_matrix"]
            confusion_matrices.append(latest_confusion)
        return charts, confusion_matrices

    def _load_fold_history_confusion_matrices(
        self, live_status: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Execute load fold history confusion matrices.



        Args:

            live_status: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        confusion_matrices: list[dict[str, Any]] = []
        current_trial = live_status.get("trial_number")
        current_fold = live_status.get("cv_fold_id")
        fold_history_path = settings.OUTPUTS_DIR / "optimization" / "plots" / "fold_history.json"
        if not fold_history_path.exists():
            return confusion_matrices
        try:
            history_data = FileManager().read(fold_history_path)
            if hasattr(history_data, "to_native"):
                history_data = history_data.to_native()
        except Exception:
            return confusion_matrices
        if not isinstance(history_data, list):
            return confusion_matrices
        seen: dict[str, dict[str, Any]] = {}
        for entry in history_data:
            if not isinstance(entry, dict) or not entry.get("confusion_matrix"):
                continue
            t_num = entry.get("trial_number")
            f_id = entry.get("cv_fold_id")
            if t_num == current_trial and f_id == current_fold:
                continue
            seen[f"{t_num}:{f_id}"] = entry
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
        return confusion_matrices

    def _extract_latest_live_confusion(self, live_status: dict[str, Any]) -> dict[str, Any] | None:
        """Execute extract latest live confusion.



        Args:

            live_status: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        epoch_history = live_status.get("epoch_history")
        if not epoch_history:
            return None
        val_events = [
            e
            for e in epoch_history
            if isinstance(e, dict) and (("vp" in e) or ("tp" in e) or ("fp" in e) or ("fn" in e))
        ]
        if not val_events:
            return None
        last_val = val_events[-1]
        cm = {
            "vp": self._get_cm_val(last_val, "vp", "tp"),
            "vn": self._get_cm_val(last_val, "vn", "tn"),
            "fp": self._get_cm_val(last_val, "fp", "fp"),
            "fn": self._get_cm_val(last_val, "fn", "fn"),
        }
        return {
            "timestamp": last_val.get("timestamp"),
            "epoch": last_val.get("epoch"),
            "trial_number": live_status.get("trial_number"),
            "cv_fold_id": live_status.get("cv_fold_id"),
            "confusion_matrix": cm,
        }

    @staticmethod
    def _get_cm_val(obj: dict[str, Any], key: str, alt: str) -> int:
        """Execute get cm val.



        Args:

            obj: Input value used by this callable.

            key: Input value used by this callable.

            alt: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        value = obj.get(key)
        if value is None:
            value = obj.get(alt)
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _compute_search_space_advice(
        search_space: dict[str, Any],
        trials_data: list[dict[str, Any]],
        importances: dict[str, float],
        direction: str,
        study_name: str,
        study: Any | None = None,
        search_space_coverage: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not search_space or not trials_data:
            return None
        try:
            from pff.infrastructure.hpo.search_space_advisor import (
                SearchSpaceAdvisor,
                compute_dataset_profile_fingerprint,
            )

            advisor = SearchSpaceAdvisor()
            dataset_fingerprint, dataset_profile = compute_dataset_profile_fingerprint()
            objective_directions: list[str] | None = None
            if study is not None:
                raw_dirs = getattr(study, "directions", None)
                if isinstance(raw_dirs, (list, tuple)) and raw_dirs:
                    objective_directions = [
                        str(getattr(item, "name", item)).lower() for item in raw_dirs
                    ]
            return advisor.advise(
                search_space=search_space,
                trials_data=trials_data,
                importances=importances,
                direction=direction,
                study_name=study_name,
                dataset_fingerprint=dataset_fingerprint,
                dataset_profile=dataset_profile,
                study=study,
                objective_directions=objective_directions,
                advisor_config={
                    "distribution_conflicts": (
                        search_space_coverage.get("distribution_conflicts", [])
                        if isinstance(search_space_coverage, dict)
                        else []
                    ),
                    "search_space_coverage_ratio": (
                        search_space_coverage.get("search_space_coverage_ratio")
                        if isinstance(search_space_coverage, dict)
                        else None
                    ),
                },
            )
        except Exception:
            return None

    @staticmethod
    def _compute_param_importances(study: Any, completed_trials: list[Any]) -> dict[str, float]:
        """Execute compute param importances.



        Args:

            study: Input value used by this callable.

            completed_trials: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if len(completed_trials) <= 3:
            return {}
        try:
            evaluator = FanovaImportanceEvaluator(n_trees=32, seed=42)
            importances = get_param_importances(study, evaluator=evaluator)
            return {k: float(v) for k, v in importances.items()}
        except Exception:
            return {}

    def _build_trials_data(
        self,
        trials: list[Any],
        live_history_best: dict[str, Any],
        *,
        study: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Execute build trials data.



        Args:

            trials: Input value used by this callable.

            live_history_best: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        trials_data: list[dict[str, Any]] = []
        max_trial_id = max((t.number for t in trials), default=-1)
        for trial in trials:
            trial_data = self._build_single_trial_payload(
                trial=trial,
                max_trial_id=max_trial_id,
                live_history_best=live_history_best,
                study=study,
            )
            trials_data.append(trial_data)
        return trials_data

    def _build_single_trial_payload(
        self,
        *,
        trial: Any,
        max_trial_id: int,
        live_history_best: dict[str, Any],
        study: Any | None = None,
    ) -> dict[str, Any]:
        """Execute build single trial payload.



        Args:

            trial: Input value used by this callable.

            max_trial_id: Input value used by this callable.

            live_history_best: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        metrics = flatten_trial_metrics(trial)
        primary_value = self._trial_primary_value(trial)
        trial_state = self._resolve_trial_state(trial, max_trial_id)
        user_attrs = trial.user_attrs
        mrr = metrics.get("mrr", metrics.get("kge_mrr", metrics.get("best_val_mrr", 0.0)))
        if mrr == 0.0 and 0.0 < primary_value <= 1.0:
            mrr = primary_value
        best_mrr = user_attrs.get(
            "best_val_mrr", user_attrs.get("best_mrr", metrics.get("best_mrr"))
        )
        if best_mrr is None and trial.number == live_history_best.get("id"):
            best_mrr = live_history_best["mrr"]
        mcc = metrics.get("mcc", user_attrs.get("mcc"))
        best_mcc = user_attrs.get(
            "best_val_mcc", user_attrs.get("best_mcc", metrics.get("best_mcc"))
        )
        if best_mcc is None and trial.number == live_history_best.get("id"):
            best_mcc = live_history_best["mcc"]
        duration = self._resolve_duration(trial, metrics)
        loss_value = metrics.get("loss") or metrics.get("val_loss") or metrics.get("train_loss")
        if loss_value is not None:
            metrics.setdefault("loss", loss_value)
        metrics.setdefault("duration", duration)
        efficiency = self._resolve_efficiency(primary_value, duration, metrics)
        raw_values = getattr(trial, "values", None)
        values_payload = (
            [float(v) for v in raw_values if isinstance(v, (int, float))]
            if isinstance(raw_values, (list, tuple))
            else None
        )
        return {
            "id": trial.number + 1,
            "value": primary_value,
            "values": values_payload,
            "state": trial_state,
            "params": trial.params if hasattr(trial, "params") else {},
            "duration": duration,
            "loss": loss_value,
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "efficiency": efficiency,
            "mrr": mrr,
            "best_mrr": float(best_mrr) if best_mrr is not None else None,
            "mcc": mcc,
            "best_mcc": float(best_mcc) if best_mcc is not None else None,
            "auc": metrics.get("auc"),
            "hits1": metrics.get("hits1", metrics.get("hits@1", user_attrs.get("hits@1"))),
            "hits3": metrics.get("hits3", metrics.get("hits@3", user_attrs.get("hits@3"))),
            "hits10": metrics.get("hits10", metrics.get("hits@10", user_attrs.get("hits@10"))),
            "inference_latency": metrics.get("inference_latency"),
            "warmstart": self._is_warmstart_trial(trial, study=study),
            "metrics": metrics,
        }

    @staticmethod
    def _load_trial_system_attrs(trial: Any, *, study: Any | None = None) -> dict[str, Any]:
        """Load system attributes via Optuna storage API without deprecated Trial.system_attrs."""
        storage = getattr(trial, "_storage", None)
        if storage is None and study is not None:
            storage = getattr(study, "_storage", None)
        trial_id = getattr(trial, "_trial_id", None)
        if storage is None or trial_id is None or not hasattr(storage, "get_trial_system_attrs"):
            return {}
        try:
            loaded = storage.get_trial_system_attrs(trial_id)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    @classmethod
    def _is_warmstart_trial(cls, trial: Any, *, study: Any | None = None) -> bool:
        """Resolve warmstart flag from user attributes and storage-backed system attributes."""
        user_attrs = getattr(trial, "user_attrs", {}) or {}
        system_attrs = cls._load_trial_system_attrs(trial, study=study)
        return bool(
            system_attrs.get("warmstart_seed")
            or user_attrs.get("warmstart")
            or user_attrs.get("warmstart_seed")
        )

    @staticmethod
    def _resolve_trial_state(trial: Any, max_trial_id: int) -> str:
        """Execute resolve trial state.



        Args:

            trial: Input value used by this callable.

            max_trial_id: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        trial_state = str(trial.state.name)
        if trial_state == "RUNNING" and trial.number < max_trial_id:
            return "PRUNED"
        return trial_state

    @staticmethod
    def _resolve_duration(trial: Any, metrics: dict[str, Any]) -> float:
        """Execute resolve duration.



        Args:

            trial: Input value used by this callable.

            metrics: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        duration = 0.0
        if trial.datetime_complete and trial.datetime_start:
            duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
        if duration <= 0:
            duration = metrics.get("duration", 0.0)
        return duration

    @staticmethod
    def _resolve_efficiency(
        primary_value: float, duration: float, metrics: dict[str, Any]
    ) -> float | None:
        """Execute resolve efficiency.



        Args:

            primary_value: Input value used by this callable.

            duration: Input value used by this callable.

            metrics: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not duration:
            return None
        try:
            efficiency = float(primary_value) / float(duration)
            metrics.setdefault("efficiency", efficiency)
            return efficiency
        except (TypeError, ValueError):
            return None

    def _save_dashboard_payload(self, payload: dict[str, Any], trial_count: int) -> None:
        """Execute save dashboard payload.



        Args:

            payload: Input value used by this callable.

            trial_count: Input value used by this callable.

        """

        try:
            FileManager().save(payload, self.data_path)
            logger.debug(
                "component_name=hpo_dashboard key_parameters=trials_count={trial_count}, file={file_path!r} message='Dashboard data written successfully'",
                trial_count=trial_count,
                file_path=str(self.data_path),
            )
            mirror_path = self.output_dir / "dashboard_data.json"
            if mirror_path != self.data_path:
                FileManager().save(payload, mirror_path)
        except Exception as e:
            timestamp = datetime.now(timezone.utc).isoformat()
            logger.warning(
                "timestamp={timestamp} component_name=hpo_dashboard stop_reason=write_dashboard_data_failed key_parameters=file={file_path!r} message='Failed to write dashboard data: {error}'",
                timestamp=timestamp,
                file_path=str(self.data_path),
                error=e,
            )

    @staticmethod
    def _trial_primary_value(trial: Any) -> float:
        """Execute trial primary value.



        Args:

            trial: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        value = getattr(trial, "value", None)
        if value is None:
            values = getattr(trial, "values", None)
            if isinstance(values, (list, tuple)) and values:
                value = values[0]
        try:
            return float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

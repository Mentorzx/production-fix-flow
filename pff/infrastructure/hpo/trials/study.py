from __future__ import annotations

import gc
import os
import time
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import optuna  # noqa: E402

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
try:
    from optuna.exceptions import ExperimentalWarning as _OptunaExperimentalWarning
except ImportError:
    _OptunaExperimentalWarning: Any = None  # type: ignore[no-redef,misc]
else:
    warnings.filterwarnings("ignore", category=_OptunaExperimentalWarning)

from pff.infrastructure.hpo.callbacks import (  # noqa: E402
    BestScoreObserver,
    CallbackManager,
    LivePlotCallback,
    LoggingObserver,
    MaxTrialsCallback,
    MLflowTrialObserver,
)
from pff.infrastructure.hpo.storage import create_optuna_storage  # noqa: E402
from pff.shared import logger  # noqa: E402
from pff.shared.core.config import settings  # noqa: E402
from pff.shared.core.file_manager import FileManager  # noqa: E402
from pff.shared.ops.global_interrupt_manager import (  # noqa: E402
    PRIORITY_HIGH,
    check_interruption,
    get_interrupt_manager,
)

from .artifacts import TrialArtifactManager  # noqa: E402
from .config_loader import (  # noqa: E402
    load_live_plot_settings,
    load_multi_objective_settings,
    load_optuna_settings,
)


def create_study_and_run(
    *,
    study_name: str,
    storage_path: Path,
    checkpoint_path: Path | None,
    checkpoint_key: str | None,
    checkpoint_store,
    output_dir: Path,
    work_dir: Path,
    n_trials: int,
    expected_trials: int,
    resume_mode: bool,
    checkpoint_data: dict[str, Any] | None,
    hpo_memory_config: Any,
    trial_memory,
    warmstart_callback: Callable[[optuna.Study], Any] | None,
    objective_fn: Callable[[optuna.trial.Trial], float],
    artifact_manager: TrialArtifactManager,
    enable_mlflow: bool,
    file_manager,
) -> dict[str, Any]:
    """Create Optuna study, handle resume/checkpoint, and run optimization."""
    fm = file_manager or FileManager()
    from pff.infrastructure.hpo.runner import (
        BestModelSaverCallback,
        _delete_directory,
        _write_checkpoint,
    )

    loaded_settings = load_optuna_settings(file_manager)
    sampler_settings = loaded_settings.get("sampler", {})
    hyperband_settings = loaded_settings.get("pruner", {}).get("hyperband", {})
    pruner_settings = loaded_settings.get("pruner", {})
    multi_objective = load_multi_objective_settings(file_manager)
    multi_enabled = bool(multi_objective.get("enabled", False))
    secondary_metric = str(multi_objective.get("secondary_metric", "mcc"))
    tertiary_metric = str(multi_objective.get("tertiary_metric", "duration"))
    directions = multi_objective.get("directions", ["maximize"])
    storage, storage_url = create_optuna_storage(storage_path=storage_path, file_manager=fm)

    configured_startup = max(1, int(sampler_settings.get("n_startup_trials", 5)))
    dynamic_startup = max(5, n_trials // 10)
    n_startup = min(n_trials, max(configured_startup, dynamic_startup))
    min_resource = max(1, int(hyperband_settings.get("min_resource", 5)))
    max_resource = max(min_resource + 1, int(hyperband_settings.get("max_resource", 50)))
    reduction_factor = max(2, int(hyperband_settings.get("reduction_factor", 3)))
    live_plot_settings = load_live_plot_settings(file_manager)

    if not resume_mode and fm.exists(storage_path):
        fm.delete_file(storage_path, ignore_errors=True)

    trial_runs_dir = work_dir / "trials"
    best_models_dir = work_dir / "best_models"
    if resume_mode:
        fm.ensure_dir(trial_runs_dir)
    else:
        _delete_directory(trial_runs_dir)
        fm.ensure_dir(trial_runs_dir)
        _delete_directory(best_models_dir)

    model_saver_callback = BestModelSaverCallback(
        output_dir,
        memory=trial_memory,
        artifact_manager=artifact_manager,
        trial_runs_dir=trial_runs_dir,
        study_name=study_name,
        store=checkpoint_store,
    )
    live_plot_callback = None
    if live_plot_settings.get("enabled", True):
        plot_subdir = live_plot_settings.get("output_subdir", "optimization/plots/live")
        live_plot_dir = settings.OUTPUTS_DIR / Path(plot_subdir)
        live_plot_callback = LivePlotCallback(
            output_dir=live_plot_dir,
            max_trials_axis=live_plot_settings.get("max_trials_axis", 50),
            expected_trials=expected_trials,
            enable_optuna_dashboard=live_plot_settings.get("enable_optuna_dashboard", False),
            dashboard_interval=live_plot_settings.get("dashboard_interval", 5),
            dashboard_top_n=live_plot_settings.get("dashboard_top_n", 12),
        )

    sampler_seed = int(sampler_settings.get("seed", 42))
    burn_in_epochs = int(hyperband_settings.get("burn_in_epochs", 10))

    sampler_type = str(sampler_settings.get("type", "tpe")).lower()

    sampler: Any = None
    pruner: Any = None

    sampler_name = str(multi_objective.get("sampler", "motpe")).lower()
    if multi_enabled and sampler_name == "nsga2":
        sampler = optuna.samplers.NSGAIISampler(
            seed=sampler_seed,
            population_size=int(multi_objective.get("population_size", 50)),
            crossover_prob=float(multi_objective.get("crossover_prob", 0.9)),
            mutation_prob=float(multi_objective.get("mutation_prob", 0.1)),
        )
    elif sampler_type == "auto":
        if multi_enabled:
            logger.warning(
                "component_name=hpo_study key_parameters={'multi_enabled': True} message='AutoSampler unavailable for multi-objective; using TPE'"
            )
            sampler_type = "tpe"
        else:
            try:
                import optunahub

                module = optunahub.load_module(package="samplers/auto_sampler")
                sampler = module.AutoSampler()
                logger.info(
                    "component_name=hpo_study key_parameters={'sampler_type': 'auto'} message='AutoSampler optunahub habilitado'"
                )
            except Exception as exc:
                logger.warning(
                    f"component_name=hpo_study message='AutoSampler unavailable ({exc}); using TPE'"
                )
                sampler_type = "tpe"
    if sampler_type == "tpe":
        sampler_kwargs: dict[str, Any] = {
            "seed": sampler_seed,
            "n_startup_trials": n_startup,
            "n_ei_candidates": int(sampler_settings.get("n_ei_candidates", 48)),
            "constant_liar": bool(sampler_settings.get("constant_liar", True)),
            "consider_prior": bool(sampler_settings.get("consider_prior", True)),
            "consider_magic_clip": bool(sampler_settings.get("consider_magic_clip", True)),
            "warn_independent_sampling": bool(
                sampler_settings.get("warn_independent_sampling", True)
            ),
        }
        if "multivariate" in sampler_settings:
            multivariate = bool(sampler_settings.get("multivariate", True))
            if multivariate:
                sampler_kwargs["multivariate"] = True
        if "group" in sampler_settings:
            group = bool(sampler_settings.get("group", True))
            if group:
                sampler_kwargs["group"] = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
            if _OptunaExperimentalWarning is not None:
                warnings.filterwarnings("ignore", category=_OptunaExperimentalWarning)
            sampler = optuna.samplers.TPESampler(**sampler_kwargs)
    elif sampler_type not in {"tpe", "auto"} and not (multi_enabled and sampler_name == "nsga2"):
        logger.warning(
            f"component_name=hpo_study key_parameters={{'sampler_type': '{sampler_type}'}} message='Unknown sampler type, using TPE'"
        )
        sampler_kwargs = {
            "seed": sampler_seed,
            "n_startup_trials": n_startup,
            "n_ei_candidates": int(sampler_settings.get("n_ei_candidates", 48)),
            "constant_liar": bool(sampler_settings.get("constant_liar", True)),
            "consider_prior": bool(sampler_settings.get("consider_prior", True)),
            "consider_magic_clip": bool(sampler_settings.get("consider_magic_clip", True)),
            "warn_independent_sampling": bool(
                sampler_settings.get("warn_independent_sampling", True)
            ),
        }
        if "multivariate" in sampler_settings:
            multivariate = bool(sampler_settings.get("multivariate", True))
            if multivariate:
                sampler_kwargs["multivariate"] = True
        if "group" in sampler_settings:
            group = bool(sampler_settings.get("group", True))
            if group:
                sampler_kwargs["group"] = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
            if _OptunaExperimentalWarning is not None:
                warnings.filterwarnings("ignore", category=_OptunaExperimentalWarning)
            sampler = optuna.samplers.TPESampler(**sampler_kwargs)

    pruner_type = str(pruner_settings.get("type", "hyperband")).lower()
    if pruner_type in {"hyperband", "asha", "patient", "patient_hyperband"}:
        base_pruner = optuna.pruners.HyperbandPruner(
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=reduction_factor,
        )
        patient_cfg = (
            pruner_settings.get("patient", {}) if isinstance(pruner_settings, dict) else {}
        )
        patience = int(patient_cfg.get("patience", 0))
        min_delta = float(patient_cfg.get("min_delta", 0.0))
        if patience > 0:
            pruner = optuna.pruners.PatientPruner(
                base_pruner, patience=patience, min_delta=min_delta
            )
        else:
            pruner = base_pruner
    elif pruner_type == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner_settings.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruner_settings.get("n_warmup_steps", 10)),
            interval_steps=int(pruner_settings.get("interval_steps", 1)),
        )
    elif pruner_type == "wilcoxon":
        try:
            from optuna.pruners import WilcoxonPruner

            p_threshold = float(pruner_settings.get("wilcoxon", {}).get("p_threshold", 0.1))
            n_startup_steps = int(pruner_settings.get("wilcoxon", {}).get("n_startup_steps", 2))
            pruner = WilcoxonPruner(
                p_threshold=p_threshold,
                n_startup_steps=n_startup_steps,
            )
        except Exception:
            logger.warning(
                "component_name=hpo_study message='WilcoxonPruner unavailable; using HyperbandPruner'"
            )
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=min_resource,
                max_resource=max_resource,
                reduction_factor=reduction_factor,
            )
    else:
        logger.warning(
            f"component_name=hpo_study key_parameters={{'pruner_type': '{pruner_type}'}} message='Unknown pruner type, using HyperbandPruner'"
        )
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=reduction_factor,
        )

    if multi_enabled and isinstance(directions, list) and len(directions) >= 2:
        study = optuna.create_study(
            study_name=study_name,
            directions=directions,
            sampler=sampler,
            pruner=pruner,
            storage=storage if storage is not None else storage_url,
            load_if_exists=resume_mode,
        )
        study.set_user_attr("multi_objective", True)
        study.set_user_attr("multi_objective_directions", list(directions))
        study.set_user_attr("multi_objective_secondary", secondary_metric)
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=storage if storage is not None else storage_url,
            load_if_exists=resume_mode,
        )

    study.set_user_attr("sampler_seed", sampler_seed)
    study.set_user_attr("burn_in_epochs", burn_in_epochs)

    warmstart_injected = 0
    if warmstart_callback:
        warmstart_injected = warmstart_callback(study) or 0

    def _is_warmstart(trial: optuna.trial.FrozenTrial) -> bool:
        user_attrs = getattr(trial, "user_attrs", {}) or {}
        system_attrs = getattr(trial, "system_attrs", {}) or {}
        return bool(
            system_attrs.get("warmstart_seed")
            or user_attrs.get("warmstart")
            or user_attrs.get("warmstart_seed")
        )

    completed_trials_count = sum(
        1
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and not _is_warmstart(t)
    )

    effective_completed = 0 if not resume_mode else completed_trials_count

    total_target_trials = max(expected_trials, n_trials)
    max_trials_callback = MaxTrialsCallback(total_target_trials)
    remaining_trials = max(total_target_trials - effective_completed, 0)

    checkpoint_payload = {
        "status": "running",
        "study_name": study_name,
        "expected_trials": total_target_trials,
        "completed_trials": completed_trials_count,
        "resume_mode": resume_mode,
        "last_update": datetime.now(timezone.utc).isoformat(),
    }
    _write_checkpoint(
        checkpoint_path,
        checkpoint_payload,
        store=checkpoint_store,
        checkpoint_key=checkpoint_key,
    )

    interrupt_manager = get_interrupt_manager()
    interrupt_checkpoint_label = None

    def _persist_interrupt_checkpoint() -> None:
        """Persist a minimal checkpoint snapshot during coordinated shutdown."""
        try:
            checkpoint_snapshot = dict(checkpoint_payload)
            checkpoint_snapshot["status"] = "interrupted"
            checkpoint_snapshot["completed_trials"] = len(getattr(study, "trials", []) or [])
            checkpoint_snapshot["last_update"] = datetime.now(timezone.utc).isoformat()
            _write_checkpoint(
                checkpoint_path,
                checkpoint_snapshot,
                store=checkpoint_store,
                checkpoint_key=checkpoint_key,
            )
        except Exception as exc:
            logger.warning(
                f"component_name=hpo_study message='Interrupt checkpoint write failed: {exc}'"
            )

    try:
        interrupt_checkpoint_label = interrupt_manager.register_callback(
            _persist_interrupt_checkpoint,
            priority=PRIORITY_HIGH,
            label=f"hpo_checkpoint_{study_name}",
        )
    except Exception as exc:
        logger.warning(
            f"component_name=hpo_study message='Failed to register interrupt checkpoint callback: {exc}'"
        )

    logger.info(
        f"component_name=hpo_study key_parameters={{'study_name': '{study_name}'}} message='Estudo Optuna criado'"
    )
    logger.info(
        f"component_name=hpo_study key_parameters={{'sampler': '{study.sampler.__class__.__name__}'}} message='Amostrador ativo'"
    )
    logger.info(
        f"component_name=hpo_study key_parameters={{'pruner': '{study.pruner.__class__.__name__}'}} message='Pruner configurado'"
    )
    logger.info(
        f"component_name=hpo_study key_parameters={{'dir': '{best_models_dir}'}} message='Modelos serão salvos no diretório especificado'"
    )

    if remaining_trials > 0:
        logger.info(
            f"component_name=hpo_study key_parameters={{'remaining': {remaining_trials}, 'total': {total_target_trials}}} "
            "message='Iniciando otimização com trials pendentes'"
        )
        if warmstart_injected > 0:
            logger.info(
                f"component_name=hpo_study key_parameters={{'warmstart_seeds': {warmstart_injected}}} "
                "message='Seeds de warm-start carregados'"
            )
    else:
        logger.info(
            "component_name=hpo_study stop_reason=target_reached message='Nenhum trial pendente. Resultados existentes já atingem o alvo.'"
        )
        logger.info(
            "component_name=hpo_study stop_reason=no_trials_pending message='Sem trials pendentes'"
        )

    start_time = time.time()

    callback_manager = CallbackManager()
    callback_manager.add_observer(LoggingObserver())
    callback_manager.add_observer(BestScoreObserver())
    if enable_mlflow:
        try:
            from pff.infrastructure.hpo.tracker import MLflowTracker

            tracker = MLflowTracker()
            callback_manager.add_observer(MLflowTrialObserver(tracker))
        except Exception as exc:
            logger.warning(
                f"component_name=hpo_study message='Failed to initialize MLflow tracker: {exc}'"
            )

    if callback_manager.observers:
        callback_manager.notify_start(study_name, total_target_trials)

    if live_plot_callback and live_plot_settings.get("enable_optuna_dashboard", False):
        dashboard_url = os.getenv(
            "OPTUNA_DASHBOARD_URL",
            settings.HPO_CONFIG.get("optuna", {}).get(
                "dashboard_url", "http://localhost:8080/dashboard"
            ),
        )
        refresh_sec = int(live_plot_settings.get("dashboard_interval", 5))
        logger.info(
            f"component_name=hpo_study key_parameters={{'url': '{dashboard_url}', 'refresh_s': {refresh_sec}}} message='Dashboards HPO iniciados'"
        )
        live_plot_callback.initialize_dashboard(study)

    def cleanup_after_trial(study_obj, trial_obj):
        """Force cleanup after each trial to prevent segfaults."""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        try:
            if hasattr(study_obj, "_storage") and hasattr(study_obj._storage, "_engine"):
                study_obj._storage._engine.dispose()
        except Exception as exc:
            logger.warning(
                f"component_name=hpo_study message='Trial cleanup failed to dispose Optuna storage engine: {exc}'"
            )

    def interruptible_objective(trial: optuna.trial.Trial) -> float | list[float]:
        check_interruption()
        try:
            raw_value = objective_fn(trial)
            if isinstance(raw_value, (list, tuple)) and raw_value:
                primary_value = float(raw_value[0])
            else:
                primary_value = float(raw_value)
            if multi_enabled and isinstance(directions, list) and len(directions) >= 2:
                attrs = getattr(trial, "user_attrs", {}) or {}
                secondary_value = attrs.get(secondary_metric)
                if secondary_value is None and secondary_metric != "duration":
                    secondary_value = attrs.get("duration")
                if secondary_value is None:
                    secondary_value = 0.0
                tertiary_value: float = 0.0
                if len(directions) >= 3:
                    ter_val = attrs.get(tertiary_metric)
                    if ter_val is None and tertiary_metric != "duration":
                        ter_val = attrs.get("duration")
                    tertiary_value = float(ter_val) if ter_val is not None else 0.0
                if len(directions) >= 3:
                    return [
                        primary_value,
                        float(secondary_value),
                        tertiary_value,
                    ]
                return [primary_value, float(secondary_value)]
            return primary_value
        except RuntimeError as exc:
            message = str(exc)
            if "Non-finite" in message or "NaN/Inf" in message:
                logger.warning(
                    f"component_name=hpo_study key_parameters={{'trial': {trial.number}}} message='Trial pruned due to numeric instability: {message!r}'"
                )
                trial.set_user_attr("pruned_reason", "numeric_instability")
                trial.set_user_attr("pruned_error", message)
                raise optuna.TrialPruned(message) from exc
            raise

    def _trial_primary_value(trial: Any) -> float | None:
        value = getattr(trial, "value", None)
        if value is None:
            values = getattr(trial, "values", None)
            if isinstance(values, (list, tuple)) and values:
                value = values[0]
        if value is None:
            return None
        return float(value)

    def _observer_callback(study_obj, trial_obj) -> None:
        value = _trial_primary_value(trial_obj)
        if value is None:
            return
        if callback_manager.observers:
            callback_manager.notify_all(trial_obj, value)

    interrupted = False
    try:
        if remaining_trials > 0:
            try:
                study.optimize(
                    interruptible_objective,
                    n_trials=remaining_trials,
                    n_jobs=1,
                    callbacks=cast(
                        Any,
                        [
                            cb
                            for cb in [
                                model_saver_callback,
                                live_plot_callback,
                                cleanup_after_trial,
                                max_trials_callback,
                                _observer_callback,
                            ]
                            if cb
                        ],
                    ),
                    gc_after_trial=True,
                )
            except KeyboardInterrupt:
                interrupted = True
                logger.warning(
                    "component_name=hpo_study stop_reason=user_interrupted message='Optuna study interrupted by user; returning partial results'"
                )
    finally:
        if interrupt_checkpoint_label is not None:
            interrupt_manager.unregister_callback(interrupt_checkpoint_label)

    if callback_manager.observers:
        best_value_for_log: float | None = None
        best_params_for_log: dict[str, Any] = {}
        try:
            if multi_enabled and isinstance(directions, list) and len(directions) >= 2:
                best_trials = getattr(study, "best_trials", []) or []
                if best_trials:
                    values = list(getattr(best_trials[0], "values", []) or [])
                    if values:
                        best_value_for_log = float(values[0])
                    best_params_for_log = dict(getattr(best_trials[0], "params", {}) or {})
            else:
                best_value_for_log = float(getattr(study, "best_value", 0.0))
                best_params_for_log = dict(getattr(study, "best_params", {}) or {})
        except Exception as exc:
            logger.warning(
                f"component_name=hpo_study message='Failed to get best_value for observers: {exc}'"
            )
        callback_manager.notify_end(best_value_for_log or 0.0, best_params_for_log)

    checkpoint_payload["status"] = "completed" if not interrupted else "interrupted"
    checkpoint_payload["completed_trials"] = len(study.trials)
    checkpoint_payload["last_update"] = datetime.now(timezone.utc).isoformat()
    _write_checkpoint(
        checkpoint_path,
        checkpoint_payload,
        store=checkpoint_store,
        checkpoint_key=checkpoint_key,
    )

    if interrupted:
        logger.info(
            "component_name=hpo_study stop_reason=user_interrupted message='Otimização interrompida pelo usuário'"
        )
    else:
        logger.success(
            "component_name=hpo_study stop_reason=completed message='Otimização HPO concluída com sucesso'"
        )

    optimization_time = time.time() - start_time
    pareto_front: list[dict[str, Any]] = []
    try:
        if multi_enabled and isinstance(directions, list) and len(directions) >= 2:
            best_trials = getattr(study, "best_trials", []) or []
            for trial in best_trials:
                pareto_front.append(
                    {
                        "number": trial.number,
                        "values": list(getattr(trial, "values", []) or []),
                        "params": dict(getattr(trial, "params", {}) or {}),
                    }
                )
            if best_trials:
                best_params = dict(best_trials[0].params)
                values = list(getattr(best_trials[0], "values", []) or [])
                best_value = values[0] if values else None
            else:
                best_params, best_value = {}, None
        else:
            best_params = study.best_params
            best_value = study.best_value
    except Exception:
        logger.warning(
            "component_name=hpo_study stop_reason=no_trials message='No completed trials; returning empty best_params'"
        )
        best_params, best_value = {}, None
    else:
        completed_trials = [
            trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        completed_trials.sort(key=lambda t: t.number)
        if completed_trials and best_value is not None:
            first_value = _trial_primary_value(completed_trials[0])
            if first_value is not None:
                if multi_enabled and isinstance(directions, list) and directions:
                    direction_label = str(directions[0])
                else:
                    direction_label = str(getattr(study, "direction", "maximize"))
                direction_label = direction_label.lower().replace("studydirection.", "")
                if direction_label == "minimize":
                    delta = float(first_value) - float(best_value)
                else:
                    delta = float(best_value) - float(first_value)
                logger.info(
                    f"component_name=hpo_study key_parameters={{'best': {float(best_value):.4f}, 'delta': {delta:.4f}, 'direction': '{direction_label}'}} message='Métricas delta calculadas'"
                )

    result: dict[str, Any] = {
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": len(study.trials),
        "optimization_time": optimization_time,
        "framework": "optuna",
        "study": study,
        "trials": [],
        "interrupted": interrupted,
        "multi_objective": bool(multi_enabled),
    }
    if pareto_front:
        result["pareto_front"] = pareto_front
    if live_plot_callback:
        result["live_plot_dir"] = str(live_plot_callback.output_dir)
        if getattr(live_plot_callback, "enable_optuna_dashboard", False):
            result["live_dashboard"] = str(live_plot_callback.dashboard_path)
    for trial in study.trials:
        result["trials"].append(
            {
                "number": trial.number,
                "value": trial.value,
                "values": list(getattr(trial, "values", []) or []),
                "params": trial.params,
                "state": str(trial.state),
            }
        )
    return result

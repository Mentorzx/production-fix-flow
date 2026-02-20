"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/infrastructure/hpo/trials/study.py

"""

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

try:
    from optuna.exceptions import ExperimentalWarning as _OptunaExperimentalWarning
except ImportError:
    _OptunaExperimentalWarning: Any = None  # type: ignore[no-redef,misc]


def _configure_optuna_warnings() -> None:
    """Execute configure optuna warnings."""

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    if _OptunaExperimentalWarning is not None:
        warnings.filterwarnings("ignore", category=_OptunaExperimentalWarning)


from pff.infrastructure.hpo.callbacks_internal.observers import (  # noqa: E402
    AdaptiveSamplerController,
    BestScoreObserver,
    CallbackManager,
    LoggingObserver,
    MaxTrialsCallback,
    MLflowTrialObserver,
    StagnationDetector,
)
from pff.infrastructure.hpo.callbacks_internal.visualizers import (  # noqa: E402
    LivePlotCallback,
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


def _build_tpe_sampler_kwargs(
    sampler_settings: dict[str, Any],
    *,
    sampler_seed: int,
    n_startup: int,
) -> dict[str, Any]:
    """Execute build tpe sampler kwargs.



    Args:

        sampler_settings: Input value used by this callable.

        sampler_seed: Input value used by this callable.

        n_startup: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    sampler_kwargs: dict[str, Any] = {
        "seed": sampler_seed,
        "n_startup_trials": n_startup,
        "n_ei_candidates": int(sampler_settings.get("n_ei_candidates", 48)),
        "constant_liar": bool(sampler_settings.get("constant_liar", True)),
        "consider_prior": bool(sampler_settings.get("consider_prior", True)),
        "consider_magic_clip": bool(sampler_settings.get("consider_magic_clip", True)),
        "warn_independent_sampling": bool(sampler_settings.get("warn_independent_sampling", True)),
    }
    if bool(sampler_settings.get("multivariate", False)):
        sampler_kwargs["multivariate"] = True
    if bool(sampler_settings.get("group", False)):
        sampler_kwargs["group"] = True
    return sampler_kwargs


def _build_tpe_sampler(
    sampler_settings: dict[str, Any],
    *,
    sampler_seed: int,
    n_startup: int,
) -> Any:
    """Execute build tpe sampler.



    Args:

        sampler_settings: Input value used by this callable.

        sampler_seed: Input value used by this callable.

        n_startup: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    sampler_kwargs = _build_tpe_sampler_kwargs(
        sampler_settings=sampler_settings,
        sampler_seed=sampler_seed,
        n_startup=n_startup,
    )
    with warnings.catch_warnings():
        _configure_optuna_warnings()
        return optuna.samplers.TPESampler(**sampler_kwargs)


def _build_cmaes_sampler(
    sampler_settings: dict[str, Any],
    *,
    sampler_seed: int,
) -> Any:
    """Execute build cmaes sampler.



    Args:

        sampler_settings: Input value used by this callable.

        sampler_seed: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    cmaes_settings = sampler_settings.get("alternatives", {}).get("cmaes", {})
    sampler_kwargs = {
        "seed": sampler_seed,
        "n_startup_trials": int(cmaes_settings.get("n_startup_trials", 5)),
        "warn_independent_sampling": bool(cmaes_settings.get("warn_independent_sampling", False)),
    }
    with warnings.catch_warnings():
        _configure_optuna_warnings()
        sampler = optuna.samplers.CmaEsSampler(**sampler_kwargs)
    logger.bind(
        component="hpo_study",
        stop_reason="sampler_configured",
        key_parameters={
            "sampler_type": "cmaes",
            "n_startup": sampler_kwargs["n_startup_trials"],
        },
    ).info("Sampler CMA-ES habilitado para diversidade.")
    return sampler


def _resolve_sampler(
    *,
    sampler_settings: dict[str, Any],
    multi_objective: dict[str, Any],
    multi_enabled: bool,
    sampler_seed: int,
    n_startup: int,
) -> Any:
    """Execute resolve sampler.



    Args:

        sampler_settings: Input value used by this callable.

        multi_objective: Input value used by this callable.

        multi_enabled: Input value used by this callable.

        sampler_seed: Input value used by this callable.

        n_startup: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    sampler_type = str(sampler_settings.get("type", "tpe")).lower()
    sampler_name = str(multi_objective.get("sampler", "motpe")).lower()
    if multi_enabled and sampler_name == "nsga2":
        return optuna.samplers.NSGAIISampler(
            seed=sampler_seed,
            population_size=int(multi_objective.get("population_size", 50)),
            crossover_prob=float(multi_objective.get("crossover_prob", 0.9)),
            mutation_prob=float(multi_objective.get("mutation_prob", 0.1)),
        )
    if sampler_type == "auto":
        auto_sampler = _try_build_auto_sampler(multi_enabled)
        if auto_sampler is not None:
            return auto_sampler
        sampler_type = "tpe"
    if sampler_type == "cmaes":
        return _build_cmaes_sampler(sampler_settings, sampler_seed=sampler_seed)
    if sampler_type not in {"tpe", "auto", "cmaes"}:
        logger.bind(
            component="hpo_study",
            stop_reason="sampler_unknown",
            key_parameters={"sampler_type": sampler_type},
        ).warning("Unknown sampler type, using TPE.")
    return _build_tpe_sampler(
        sampler_settings=sampler_settings,
        sampler_seed=sampler_seed,
        n_startup=n_startup,
    )


def _try_build_auto_sampler(multi_enabled: bool) -> Any | None:
    """Execute try build auto sampler.



    Args:

        multi_enabled: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    if multi_enabled:
        logger.bind(
            component="hpo_study",
            stop_reason="autosampler_unavailable",
            key_parameters={"multi_enabled": True},
        ).warning("AutoSampler unavailable for multi-objective; using TPE.")
        return None
    try:
        import optunahub

        module = optunahub.load_module(package="samplers/auto_sampler")
        sampler = module.AutoSampler()
        logger.bind(
            component="hpo_study",
            stop_reason="sampler_configured",
            key_parameters={"sampler_type": "auto"},
        ).info("AutoSampler do optunahub habilitado.")
        return sampler
    except Exception as exc:
        logger.bind(
            component="hpo_study",
            stop_reason="autosampler_failed",
            key_parameters={"error": repr(exc)},
        ).warning("AutoSampler unavailable; using TPE.")
        return None


def _resolve_pruner(
    *,
    pruner_settings: dict[str, Any],
    min_resource: int,
    max_resource: int,
    reduction_factor: int,
) -> Any:
    """Execute resolve pruner.



    Args:

        pruner_settings: Input value used by this callable.

        min_resource: Input value used by this callable.

        max_resource: Input value used by this callable.

        reduction_factor: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    pruner_type = str(pruner_settings.get("type", "hyperband")).lower()
    if pruner_type in {"hyperband", "asha", "patient", "patient_hyperband"}:
        return _build_hyperband_patient_pruner(
            pruner_settings=pruner_settings,
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=reduction_factor,
        )
    if pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner_settings.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruner_settings.get("n_warmup_steps", 10)),
            interval_steps=int(pruner_settings.get("interval_steps", 1)),
        )
    if pruner_type == "wilcoxon":
        return _build_wilcoxon_or_hyperband_pruner(
            pruner_settings=pruner_settings,
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=reduction_factor,
        )
    logger.bind(
        component="hpo_study",
        stop_reason="pruner_unknown",
        key_parameters={"pruner_type": pruner_type},
    ).warning("Unknown pruner type, using HyperbandPruner.")
    return optuna.pruners.HyperbandPruner(
        min_resource=min_resource,
        max_resource=max_resource,
        reduction_factor=reduction_factor,
    )


def _build_hyperband_patient_pruner(
    *,
    pruner_settings: dict[str, Any],
    min_resource: int,
    max_resource: int,
    reduction_factor: int,
) -> Any:
    """Execute build hyperband patient pruner.



    Args:

        pruner_settings: Input value used by this callable.

        min_resource: Input value used by this callable.

        max_resource: Input value used by this callable.

        reduction_factor: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    base_pruner = optuna.pruners.HyperbandPruner(
        min_resource=min_resource,
        max_resource=max_resource,
        reduction_factor=reduction_factor,
    )
    patient_cfg = pruner_settings.get("patient", {})
    patience = int(patient_cfg.get("patience", 0))
    min_delta = float(patient_cfg.get("min_delta", 0.0))
    if patience <= 0:
        return base_pruner
    return optuna.pruners.PatientPruner(base_pruner, patience=patience, min_delta=min_delta)


def _build_wilcoxon_or_hyperband_pruner(
    *,
    pruner_settings: dict[str, Any],
    min_resource: int,
    max_resource: int,
    reduction_factor: int,
) -> Any:
    """Execute build wilcoxon or hyperband pruner.



    Args:

        pruner_settings: Input value used by this callable.

        min_resource: Input value used by this callable.

        max_resource: Input value used by this callable.

        reduction_factor: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    try:
        from optuna.pruners import WilcoxonPruner

        p_threshold = float(pruner_settings.get("wilcoxon", {}).get("p_threshold", 0.1))
        n_startup_steps = int(pruner_settings.get("wilcoxon", {}).get("n_startup_steps", 2))
        return WilcoxonPruner(
            p_threshold=p_threshold,
            n_startup_steps=n_startup_steps,
        )
    except Exception:
        logger.bind(
            component="hpo_study",
            stop_reason="wilcoxon_unavailable",
            key_parameters={},
        ).warning("WilcoxonPruner unavailable; using HyperbandPruner.")
        return optuna.pruners.HyperbandPruner(
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=reduction_factor,
        )


def _create_study(
    *,
    study_name: str,
    storage: Any | None,
    storage_url: str | None,
    resume_mode: bool,
    sampler: Any,
    pruner: Any,
    multi_enabled: bool,
    directions: Any,
    secondary_metric: str,
) -> Any:
    """Execute create study.



    Args:

        study_name: Input value used by this callable.

        storage: Input value used by this callable.

        storage_url: Input value used by this callable.

        resume_mode: Input value used by this callable.

        sampler: Input value used by this callable.

        pruner: Input value used by this callable.

        multi_enabled: Input value used by this callable.

        directions: Input value used by this callable.

        secondary_metric: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    storage_obj = storage if storage is not None else storage_url
    if multi_enabled and isinstance(directions, list) and len(directions) >= 2:
        study = optuna.create_study(
            study_name=study_name,
            directions=directions,
            sampler=sampler,
            pruner=pruner,
            storage=storage_obj,
            load_if_exists=resume_mode,
        )
        study.set_user_attr("multi_objective", True)
        study.set_user_attr("multi_objective_directions", list(directions))
        study.set_user_attr("multi_objective_secondary", secondary_metric)
        return study
    return optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_obj,
        load_if_exists=resume_mode,
    )


def _load_trial_system_attrs(
    trial: optuna.trial.FrozenTrial,
    *,
    study: Any | None = None,
) -> dict[str, Any]:
    """Load trial system attributes via storage API without touching deprecated Trial.system_attrs."""
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


def _is_warmstart_trial(
    trial: optuna.trial.FrozenTrial,
    *,
    study: Any | None = None,
) -> bool:
    """Execute is warmstart trial.



    Args:

        trial: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    user_attrs = getattr(trial, "user_attrs", {}) or {}
    system_attrs = _load_trial_system_attrs(trial, study=study)
    return bool(
        system_attrs.get("warmstart_seed")
        or user_attrs.get("warmstart")
        or user_attrs.get("warmstart_seed")
    )


def _count_completed_trials(
    study: Any, *, resume_mode: bool
) -> tuple[int, int, int, int, MaxTrialsCallback]:
    completed_trials_count = sum(
        1
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and not _is_warmstart_trial(trial, study=study)
    )
    effective_completed = 0 if not resume_mode else completed_trials_count
    return completed_trials_count, effective_completed, 0, 0, MaxTrialsCallback(0)


def _prepare_trial_counts(
    *,
    study: Any,
    n_trials: int,
    expected_trials: int,
    resume_mode: bool,
) -> tuple[int, int, int, MaxTrialsCallback]:
    """Execute prepare trial counts.



    Args:

        study: Input value used by this callable.

        n_trials: Input value used by this callable.

        expected_trials: Input value used by this callable.

        resume_mode: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    completed_trials_count = sum(
        1
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and not _is_warmstart_trial(trial)
    )
    effective_completed = 0 if not resume_mode else completed_trials_count
    if resume_mode and int(expected_trials) > 0:
        total_target_trials = max(int(expected_trials), completed_trials_count)
    else:
        total_target_trials = int(n_trials) if int(n_trials) > 0 else int(expected_trials)
    remaining_trials = max(total_target_trials - effective_completed, 0)
    return (
        completed_trials_count,
        total_target_trials,
        remaining_trials,
        MaxTrialsCallback(total_target_trials),
    )


def _write_running_checkpoint_payload(
    *,
    checkpoint_path: Path | None,
    checkpoint_key: str | None,
    checkpoint_store: Any,
    write_checkpoint: Callable[..., None],
    study_name: str,
    total_target_trials: int,
    completed_trials_count: int,
    resume_mode: bool,
) -> dict[str, Any]:
    """Execute write running checkpoint payload.



    Args:

        checkpoint_path: Input value used by this callable.

        checkpoint_key: Input value used by this callable.

        checkpoint_store: Input value used by this callable.

        write_checkpoint: Input value used by this callable.

        study_name: Input value used by this callable.

        total_target_trials: Input value used by this callable.

        completed_trials_count: Input value used by this callable.

        resume_mode: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    checkpoint_payload = {
        "status": "running",
        "study_name": study_name,
        "expected_trials": total_target_trials,
        "completed_trials": completed_trials_count,
        "resume_mode": resume_mode,
        "last_update": datetime.now(timezone.utc).isoformat(),
    }
    write_checkpoint(
        checkpoint_path,
        checkpoint_payload,
        store=checkpoint_store,
        checkpoint_key=checkpoint_key,
    )
    return checkpoint_payload


def _register_interrupt_checkpoint_callback(
    *,
    study_name: str,
    study: Any,
    checkpoint_payload: dict[str, Any],
    checkpoint_path: Path | None,
    checkpoint_key: str | None,
    checkpoint_store: Any,
    write_checkpoint: Callable[..., None],
) -> tuple[Any, str | None]:
    """Execute register interrupt checkpoint callback.



    Args:

        study_name: Input value used by this callable.

        study: Input value used by this callable.

        checkpoint_payload: Input value used by this callable.

        checkpoint_path: Input value used by this callable.

        checkpoint_key: Input value used by this callable.

        checkpoint_store: Input value used by this callable.

        write_checkpoint: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    interrupt_manager = get_interrupt_manager()

    def _persist_interrupt_checkpoint() -> None:
        """Execute persist interrupt checkpoint."""

        try:
            checkpoint_snapshot = dict(checkpoint_payload)
            checkpoint_snapshot["status"] = "interrupted"
            checkpoint_snapshot["completed_trials"] = len(getattr(study, "trials", []) or [])
            checkpoint_snapshot["last_update"] = datetime.now(timezone.utc).isoformat()
            write_checkpoint(
                checkpoint_path,
                checkpoint_snapshot,
                store=checkpoint_store,
                checkpoint_key=checkpoint_key,
            )
        except Exception as exc:
            logger.bind(
                component="hpo_study",
                stop_reason="checkpoint_write_failed",
                key_parameters={"error": repr(exc)},
            ).warning("Interrupt checkpoint write failed.")

    try:
        label = interrupt_manager.register_callback(
            _persist_interrupt_checkpoint,
            priority=PRIORITY_HIGH,
            label=f"hpo_checkpoint_{study_name}",
        )
        return interrupt_manager, label
    except Exception as exc:
        logger.bind(
            component="hpo_study",
            stop_reason="checkpoint_register_failed",
            key_parameters={"error": repr(exc)},
        ).warning("Failed to register interrupt checkpoint callback.")
        return interrupt_manager, None


def _log_study_configuration(
    *,
    study_name: str,
    study: Any,
    best_models_dir: Path,
    remaining_trials: int,
    total_target_trials: int,
    warmstart_injected: int,
) -> None:
    """Execute log study configuration.



    Args:

        study_name: Input value used by this callable.

        study: Input value used by this callable.

        best_models_dir: Input value used by this callable.

        remaining_trials: Input value used by this callable.

        total_target_trials: Input value used by this callable.

        warmstart_injected: Input value used by this callable.

    """

    logger.bind(
        component="hpo_study",
        stop_reason="study_created",
        key_parameters={"study_name": study_name},
    ).info("Estudo Optuna criado.")
    logger.bind(
        component="hpo_study",
        stop_reason="sampler_active",
        key_parameters={"sampler": study.sampler.__class__.__name__},
    ).info("Amostrador ativo.")
    logger.bind(
        component="hpo_study",
        stop_reason="pruner_active",
        key_parameters={"pruner": study.pruner.__class__.__name__},
    ).info("Pruner configurado.")
    logger.bind(
        component="hpo_study",
        stop_reason="models_dir_set",
        key_parameters={"dir": str(best_models_dir)},
    ).info("Modelos serao salvos no diretorio especificado.")
    if remaining_trials <= 0:
        logger.bind(
            component="hpo_study",
            stop_reason="target_reached",
            key_parameters={},
        ).info("Nenhum trial pendente. Resultados existentes ja atingem o alvo.")
        logger.bind(
            component="hpo_study",
            stop_reason="no_trials_pending",
            key_parameters={},
        ).info("Sem trials pendentes.")
        return
    logger.bind(
        component="hpo_study",
        stop_reason="trials_pending",
        key_parameters={"remaining": remaining_trials, "total": total_target_trials},
    ).info("Iniciando otimizacao com trials pendentes.")
    if warmstart_injected > 0:
        logger.bind(
            component="hpo_study",
            stop_reason="warmstart_loaded",
            key_parameters={"warmstart_seeds": warmstart_injected},
        ).info("Seeds de warm-start carregados.")


def _configure_callback_manager(
    *,
    study: Any,
    study_name: str,
    total_target_trials: int,
    sampler_settings: dict[str, Any],
    multi_enabled: bool,
    enable_mlflow: bool,
) -> CallbackManager:
    """Execute configure callback manager.



    Args:

        study: Input value used by this callable.

        study_name: Input value used by this callable.

        total_target_trials: Input value used by this callable.

        sampler_settings: Input value used by this callable.

        multi_enabled: Input value used by this callable.

        enable_mlflow: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    callback_manager = CallbackManager()
    callback_manager.add_observer(LoggingObserver())
    callback_manager.add_observer(BestScoreObserver())
    callback_manager.add_observer(
        StagnationDetector(window_size=7, min_trials=10, improvement_threshold=0.02)
    )
    _maybe_add_adaptive_sampler_observer(
        callback_manager=callback_manager,
        study=study,
        sampler_settings=sampler_settings,
        multi_enabled=multi_enabled,
    )
    _maybe_add_mlflow_observer(callback_manager, enable_mlflow)
    if callback_manager.observers:
        callback_manager.notify_start(study_name, total_target_trials)
    return callback_manager


def _maybe_add_adaptive_sampler_observer(
    *,
    callback_manager: CallbackManager,
    study: Any,
    sampler_settings: dict[str, Any],
    multi_enabled: bool,
) -> None:
    """Execute maybe add adaptive sampler observer.



    Args:

        callback_manager: Input value used by this callable.

        study: Input value used by this callable.

        sampler_settings: Input value used by this callable.

        multi_enabled: Input value used by this callable.

    """

    adaptive_enabled = sampler_settings.get("adaptive_switching", True)
    if not adaptive_enabled or multi_enabled:
        return
    adaptive_controller = AdaptiveSamplerController(
        study=study,
        sampler_settings=sampler_settings,
        window_size=sampler_settings.get("adaptive_window_size", 7),
        min_trials=sampler_settings.get("adaptive_min_trials", 10),
        improvement_threshold=sampler_settings.get("adaptive_threshold", 0.02),
        max_switches=sampler_settings.get("adaptive_max_switches", 3),
    )
    callback_manager.add_observer(adaptive_controller)
    logger.bind(
        component="hpo_study",
        stop_reason="adaptive_sampler_enabled",
        key_parameters={
            "adaptive_enabled": True,
            "primary_sampler": sampler_settings.get("type", "tpe"),
            "alternative": "gp",
        },
    ).info("Troca adaptativa de amostrador habilitada (TPE ↔ GP).")


def _maybe_add_mlflow_observer(callback_manager: CallbackManager, enable_mlflow: bool) -> None:
    """Execute maybe add mlflow observer.



    Args:

        callback_manager: Input value used by this callable.

        enable_mlflow: Input value used by this callable.

    """

    if not enable_mlflow:
        return
    try:
        from pff.infrastructure.hpo.tracker import MLflowTracker

        tracker = MLflowTracker()
        callback_manager.add_observer(MLflowTrialObserver(tracker))
    except Exception as exc:
        logger.bind(
            component="hpo_study",
            stop_reason="mlflow_init_failed",
            key_parameters={"error": repr(exc)},
        ).warning("Failed to initialize MLflow tracker.")


def _maybe_initialize_dashboard(
    *,
    live_plot_callback: Any | None,
    live_plot_settings: dict[str, Any],
    study: Any,
) -> None:
    """Execute maybe initialize dashboard.



    Args:

        live_plot_callback: Input value used by this callable.

        live_plot_settings: Input value used by this callable.

        study: Input value used by this callable.

    """

    if not live_plot_callback:
        return
    if not live_plot_settings.get("enable_optuna_dashboard", False):
        return
    dashboard_url = os.getenv(
        "OPTUNA_DASHBOARD_URL",
        settings.HPO_CONFIG.get("optuna", {}).get(
            "dashboard_url", "http://localhost:8080/dashboard"
        ),
    )
    refresh_sec = int(live_plot_settings.get("dashboard_interval", 5))
    logger.bind(
        component="hpo_study",
        stop_reason="dashboard_started",
        key_parameters={"url": dashboard_url, "refresh_s": refresh_sec},
    ).info("Dashboards HPO iniciados.")
    live_plot_callback.initialize_dashboard(study)


def _cleanup_after_trial(study_obj: Any, trial_obj: Any) -> None:
    """Execute cleanup after trial.



    Args:

        study_obj: Input value used by this callable.

        trial_obj: Input value used by this callable.

    """

    del trial_obj
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
        logger.bind(
            component="hpo_study",
            stop_reason="cleanup_failed",
            key_parameters={"error": repr(exc)},
        ).warning("Trial cleanup failed to dispose Optuna storage engine.")


def _resolve_primary_value(raw_value: Any) -> float:
    """Execute resolve primary value.



    Args:

        raw_value: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Raises:

        Exception: Propagates domain-specific failures with context.

    """

    if isinstance(raw_value, (list, tuple)):
        if not raw_value:
            raise ValueError("Primary objective value list is empty")
        raw_value = raw_value[0]
    return float(raw_value)


def _resolve_secondary_objective(attrs: dict[str, Any], secondary_metric: str) -> float:
    """Execute resolve secondary objective.



    Args:

        attrs: Input value used by this callable.

        secondary_metric: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    secondary_value = attrs.get(secondary_metric)
    if secondary_value is None and secondary_metric != "duration":
        secondary_value = attrs.get("duration")
    return float(secondary_value) if secondary_value is not None else 0.0


def _resolve_tertiary_objective(attrs: dict[str, Any], tertiary_metric: str) -> float:
    """Execute resolve tertiary objective.



    Args:

        attrs: Input value used by this callable.

        tertiary_metric: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    tertiary_value = attrs.get(tertiary_metric)
    if tertiary_value is None and tertiary_metric != "duration":
        tertiary_value = attrs.get("duration")
    return float(tertiary_value) if tertiary_value is not None else 0.0


def _resolve_objective_output(
    *,
    trial: optuna.trial.Trial,
    raw_value: Any,
    multi_enabled: bool,
    directions: Any,
    secondary_metric: str,
    tertiary_metric: str,
) -> float | list[float]:
    """Execute resolve objective output.



    Args:

        trial: Input value used by this callable.

        raw_value: Input value used by this callable.

        multi_enabled: Input value used by this callable.

        directions: Input value used by this callable.

        secondary_metric: Input value used by this callable.

        tertiary_metric: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    primary_value = _resolve_primary_value(raw_value)
    if not (multi_enabled and isinstance(directions, list) and len(directions) >= 2):
        return primary_value
    attrs = getattr(trial, "user_attrs", {}) or {}
    output: list[float] = [primary_value, _resolve_secondary_objective(attrs, secondary_metric)]
    if len(directions) >= 3:
        output.append(_resolve_tertiary_objective(attrs, tertiary_metric))
    return output


def _build_interruptible_objective(
    *,
    objective_fn: Callable[[optuna.trial.Trial], float],
    multi_enabled: bool,
    directions: Any,
    secondary_metric: str,
    tertiary_metric: str,
) -> Callable[[optuna.trial.Trial], float | list[float]]:
    """Execute build interruptible objective.



    Args:

        objective_fn: Input value used by this callable.

        multi_enabled: Input value used by this callable.

        directions: Input value used by this callable.

        secondary_metric: Input value used by this callable.

        tertiary_metric: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Raises:

        Exception: Propagates domain-specific failures with context.

    """

    def interruptible_objective(trial: optuna.trial.Trial) -> float | list[float]:
        """Execute interruptible objective.



        Args:

            trial: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        try:
            check_interruption()
            raw_value = objective_fn(trial)
            return _resolve_objective_output(
                trial=trial,
                raw_value=raw_value,
                multi_enabled=multi_enabled,
                directions=directions,
                secondary_metric=secondary_metric,
                tertiary_metric=tertiary_metric,
            )
        except KeyboardInterrupt as exc:
            logger.bind(
                component="hpo_study",
                stop_reason="user_interrupted",
                key_parameters={"trial": trial.number},
            ).warning("Trial interrupted by user.")
            trial.set_user_attr("pruned_reason", "user_interrupted")
            trial.set_user_attr("pruned_error", str(exc))
            raise optuna.TrialPruned("User interrupted the trial") from exc
        except RuntimeError as exc:
            message = str(exc)
            if "Non-finite" in message or "NaN/Inf" in message:
                logger.bind(
                    component="hpo_study",
                    stop_reason="numeric_instability",
                    key_parameters={"trial": trial.number, "prune_message": message},
                ).warning("Trial pruned due to numeric instability.")
                trial.set_user_attr("pruned_reason", "numeric_instability")
                trial.set_user_attr("pruned_error", message)
                raise optuna.TrialPruned(message) from exc
            raise

    return interruptible_objective


def _trial_primary_value(trial: Any) -> float | None:
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
    if value is None:
        return None
    return float(value)


def _build_observer_callback(callback_manager: CallbackManager) -> Callable[[Any, Any], None]:
    """Execute build observer callback.



    Args:

        callback_manager: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    def _observer_callback(study_obj: Any, trial_obj: Any) -> None:
        """Execute observer callback.



        Args:

            study_obj: Input value used by this callable.

            trial_obj: Input value used by this callable.

        """

        del study_obj
        value = _trial_primary_value(trial_obj)
        if value is None:
            return
        if callback_manager.observers:
            callback_manager.notify_all(trial_obj, value)

    return _observer_callback


def _run_optuna_optimization(
    *,
    study: Any,
    objective: Callable[[optuna.trial.Trial], float | list[float]],
    remaining_trials: int,
    callbacks: list[Any],
) -> bool:
    """Execute run optuna optimization.



    Args:

        study: Input value used by this callable.

        objective: Input value used by this callable.

        remaining_trials: Input value used by this callable.

        callbacks: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    if remaining_trials <= 0:
        return False
    try:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            n_jobs=1,
            callbacks=cast(Any, [cb for cb in callbacks if cb]),
            gc_after_trial=True,
        )
        return False
    except KeyboardInterrupt:
        logger.bind(
            component="hpo_study",
            stop_reason="user_interrupted",
            key_parameters={},
        ).warning("Optuna study interrupted by user; returning partial results.")
        return True


def _notify_callback_manager_end(
    *,
    callback_manager: CallbackManager,
    study: Any,
    multi_enabled: bool,
    directions: Any,
) -> None:
    """Execute notify callback manager end.



    Args:

        callback_manager: Input value used by this callable.

        study: Input value used by this callable.

        multi_enabled: Input value used by this callable.

        directions: Input value used by this callable.

    """

    if not callback_manager.observers:
        return
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
        logger.bind(
            component="hpo_study",
            stop_reason="best_value_failed",
            key_parameters={"error": repr(exc)},
        ).warning("Failed to get best_value for observers.")
    callback_manager.notify_end(best_value_for_log or 0.0, best_params_for_log)


def _resolve_best_params_and_value(
    *,
    study: Any,
    multi_enabled: bool,
    directions: Any,
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    """Execute resolve best params and value.



    Args:

        study: Input value used by this callable.

        multi_enabled: Input value used by this callable.

        directions: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    pareto_front: list[dict[str, Any]] = []
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
            values = list(getattr(best_trials[0], "values", []) or [])
            best_value = values[0] if values else None
            return dict(best_trials[0].params), best_value, pareto_front
        return {}, None, pareto_front
    return study.best_params, study.best_value, pareto_front


def _log_delta_metric(
    *,
    study: Any,
    best_value: Any,
    multi_enabled: bool,
    directions: Any,
) -> None:
    """Execute log delta metric.



    Args:

        study: Input value used by this callable.

        best_value: Input value used by this callable.

        multi_enabled: Input value used by this callable.

        directions: Input value used by this callable.

    """

    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    completed_trials.sort(key=lambda t: t.number)
    if not completed_trials or best_value is None:
        return
    first_value = _trial_primary_value(completed_trials[0])
    if first_value is None:
        return
    if multi_enabled and isinstance(directions, list) and directions:
        direction_label = str(directions[0])
    else:
        direction_label = str(getattr(study, "direction", "maximize"))
    direction_label = direction_label.lower().replace("studydirection.", "")
    if direction_label == "minimize":
        delta = float(first_value) - float(best_value)
    else:
        delta = float(best_value) - float(first_value)
    logger.bind(
        component="hpo_study",
        stop_reason="delta_computed",
        key_parameters={
            "best": round(float(best_value), 4),
            "delta": round(delta, 4),
            "direction": direction_label,
        },
    ).info("Metricas delta calculadas.")


def _finalize_checkpoint_and_log(
    *,
    checkpoint_payload: dict[str, Any],
    study: Any,
    interrupted: bool,
    checkpoint_path: Path | None,
    checkpoint_store: Any,
    checkpoint_key: str | None,
    write_checkpoint: Callable[..., None],
) -> None:
    """Execute finalize checkpoint and log.



    Args:

        checkpoint_payload: Input value used by this callable.

        study: Input value used by this callable.

        interrupted: Input value used by this callable.

        checkpoint_path: Input value used by this callable.

        checkpoint_store: Input value used by this callable.

        checkpoint_key: Input value used by this callable.

        write_checkpoint: Input value used by this callable.

    """

    checkpoint_payload["status"] = "completed" if not interrupted else "interrupted"
    checkpoint_payload["completed_trials"] = len(study.trials)
    checkpoint_payload["last_update"] = datetime.now(timezone.utc).isoformat()
    write_checkpoint(
        checkpoint_path,
        checkpoint_payload,
        store=checkpoint_store,
        checkpoint_key=checkpoint_key,
    )
    if interrupted:
        logger.bind(
            component="hpo_study",
            stop_reason="user_interrupted",
            key_parameters={},
        ).info("Otimizacao interrompida pelo usuario.")
    else:
        logger.bind(
            component="hpo_study",
            stop_reason="completed",
            key_parameters={},
        ).success("Otimizacao HPO concluida com sucesso.")


def _build_study_result_payload(
    *,
    study: Any,
    best_params: dict[str, Any],
    best_value: Any,
    optimization_time: float,
    interrupted: bool,
    multi_enabled: bool,
    pareto_front: list[dict[str, Any]],
    live_plot_callback: Any | None,
) -> dict[str, Any]:
    """Execute build study result payload.



    Args:

        study: Input value used by this callable.

        best_params: Input value used by this callable.

        best_value: Input value used by this callable.

        optimization_time: Input value used by this callable.

        interrupted: Input value used by this callable.

        multi_enabled: Input value used by this callable.

        pareto_front: Input value used by this callable.

        live_plot_callback: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

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
    result["trials"] = [
        {
            "number": trial.number,
            "value": trial.value,
            "values": list(getattr(trial, "values", []) or []),
            "params": trial.params,
            "state": str(trial.state),
        }
        for trial in study.trials
    ]
    return result


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
    storage: Any | None = None,
    storage_url: str | None = None,
) -> dict[str, Any]:
    """Create Optuna study, handle resume/checkpoint, and run optimization."""
    _configure_optuna_warnings()
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
    if storage is None and storage_url is None:
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
            dashboard_data_path=live_plot_settings.get("dashboard_data_path"),
        )

    sampler_seed = int(sampler_settings.get("seed", 42))
    burn_in_epochs = int(hyperband_settings.get("burn_in_epochs", 10))

    sampler = _resolve_sampler(
        sampler_settings=sampler_settings,
        multi_objective=multi_objective,
        multi_enabled=multi_enabled,
        sampler_seed=sampler_seed,
        n_startup=n_startup,
    )
    pruner = _resolve_pruner(
        pruner_settings=pruner_settings,
        min_resource=min_resource,
        max_resource=max_resource,
        reduction_factor=reduction_factor,
    )
    study = _create_study(
        study_name=study_name,
        storage=storage,
        storage_url=storage_url,
        resume_mode=resume_mode,
        sampler=sampler,
        pruner=pruner,
        multi_enabled=multi_enabled,
        directions=directions,
        secondary_metric=secondary_metric,
    )

    study.set_user_attr("sampler_seed", sampler_seed)
    study.set_user_attr("burn_in_epochs", burn_in_epochs)

    warmstart_injected = warmstart_callback(study) or 0 if warmstart_callback else 0
    completed_trials_count, total_target_trials, remaining_trials, max_trials_callback = (
        _prepare_trial_counts(
            study=study,
            n_trials=n_trials,
            expected_trials=expected_trials,
            resume_mode=resume_mode,
        )
    )
    checkpoint_payload = _write_running_checkpoint_payload(
        checkpoint_path=checkpoint_path,
        checkpoint_key=checkpoint_key,
        checkpoint_store=checkpoint_store,
        write_checkpoint=_write_checkpoint,
        study_name=study_name,
        total_target_trials=total_target_trials,
        completed_trials_count=completed_trials_count,
        resume_mode=resume_mode,
    )
    interrupt_manager, interrupt_checkpoint_label = _register_interrupt_checkpoint_callback(
        study_name=study_name,
        study=study,
        checkpoint_payload=checkpoint_payload,
        checkpoint_path=checkpoint_path,
        checkpoint_key=checkpoint_key,
        checkpoint_store=checkpoint_store,
        write_checkpoint=_write_checkpoint,
    )
    _log_study_configuration(
        study_name=study_name,
        study=study,
        best_models_dir=best_models_dir,
        remaining_trials=remaining_trials,
        total_target_trials=total_target_trials,
        warmstart_injected=warmstart_injected,
    )
    callback_manager = _configure_callback_manager(
        study=study,
        study_name=study_name,
        total_target_trials=total_target_trials,
        sampler_settings=sampler_settings,
        multi_enabled=multi_enabled,
        enable_mlflow=enable_mlflow,
    )
    _maybe_initialize_dashboard(
        live_plot_callback=live_plot_callback,
        live_plot_settings=live_plot_settings,
        study=study,
    )
    objective = _build_interruptible_objective(
        objective_fn=objective_fn,
        multi_enabled=multi_enabled,
        directions=directions,
        secondary_metric=secondary_metric,
        tertiary_metric=tertiary_metric,
    )
    observer_callback = _build_observer_callback(callback_manager)
    start_time = time.time()
    try:
        interrupted = _run_optuna_optimization(
            study=study,
            objective=objective,
            remaining_trials=remaining_trials,
            callbacks=[
                model_saver_callback,
                live_plot_callback,
                _cleanup_after_trial,
                max_trials_callback,
                observer_callback,
            ],
        )
    finally:
        if interrupt_checkpoint_label is not None:
            interrupt_manager.unregister_callback(interrupt_checkpoint_label)
    _notify_callback_manager_end(
        callback_manager=callback_manager,
        study=study,
        multi_enabled=multi_enabled,
        directions=directions,
    )
    _finalize_checkpoint_and_log(
        checkpoint_payload=checkpoint_payload,
        study=study,
        interrupted=interrupted,
        checkpoint_path=checkpoint_path,
        checkpoint_store=checkpoint_store,
        checkpoint_key=checkpoint_key,
        write_checkpoint=_write_checkpoint,
    )
    optimization_time = time.time() - start_time
    try:
        best_params, best_value, pareto_front = _resolve_best_params_and_value(
            study=study,
            multi_enabled=multi_enabled,
            directions=directions,
        )
    except Exception:
        logger.bind(
            component="hpo_study",
            stop_reason="no_trials",
            key_parameters={},
        ).warning("No completed trials; returning empty best_params.")
        best_params, best_value, pareto_front = {}, None, []
    else:
        _log_delta_metric(
            study=study,
            best_value=best_value,
            multi_enabled=multi_enabled,
            directions=directions,
        )
    return _build_study_result_payload(
        study=study,
        best_params=best_params,
        best_value=best_value,
        optimization_time=optimization_time,
        interrupted=interrupted,
        multi_enabled=multi_enabled,
        pareto_front=pareto_front,
        live_plot_callback=live_plot_callback,
    )

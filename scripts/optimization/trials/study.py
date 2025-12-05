from __future__ import annotations

import gc
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import optuna

# Suppress Optuna experimental feature warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

from pff import settings
from pff.utils import logger
from pff.utils.ops.global_interrupt_manager import check_interruption

from .artifacts import TrialArtifactManager
from scripts.optimization.callbacks import LivePlotCallback
from .config_loader import load_live_plot_settings, load_optuna_settings


def create_study_and_run(
    *,
    study_name: str,
    storage_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    n_trials: int,
    expected_trials: int,
    resume_mode: bool,
    checkpoint_data: dict[str, Any] | None,
    hpo_memory_config: dict[str, Any],
    trial_memory,
    warmstart_callback: Callable[[optuna.Study], Any] | None,
    objective_fn: Callable[[optuna.trial.Trial], float],
    artifact_manager: TrialArtifactManager,
    enable_mlflow: bool,
    file_manager,
) -> dict[str, Any]:
    """Create Optuna study, handle resume/checkpoint, and run optimization."""
    from scripts.optimization.core import _load_checkpoint, _write_checkpoint, _delete_directory  # noqa: SLF001
    from scripts.optimization.core import BestModelSaverCallback  # noqa: SLF001

    storage_url = f"sqlite:///{storage_path}"
    optuna_settings = load_optuna_settings(file_manager)
    tpe_settings = optuna_settings["tpe"]
    hyperband_settings = optuna_settings["hyperband"]

    configured_startup = max(1, int(tpe_settings.get("n_startup_trials", 5)))
    dynamic_startup = max(5, n_trials // 10)
    n_startup = min(n_trials, max(configured_startup, dynamic_startup))
    min_resource = max(1, int(hyperband_settings.get("min_resource", 5)))
    max_resource = max(min_resource + 1, int(hyperband_settings.get("max_resource", 50)))
    reduction_factor = max(2, int(hyperband_settings.get("reduction_factor", 3)))
    live_plot_settings = load_live_plot_settings(file_manager)

    if not resume_mode and storage_path.exists() and checkpoint_data is None:
        storage_path.unlink()

    trial_runs_dir = output_dir / "trials"
    best_models_dir = output_dir / "best_models"
    if resume_mode:
        trial_runs_dir.mkdir(parents=True, exist_ok=True)
    else:
        _delete_directory(trial_runs_dir)
        trial_runs_dir.mkdir(parents=True, exist_ok=True)
        _delete_directory(best_models_dir)

    model_saver_callback = BestModelSaverCallback(output_dir, memory=trial_memory, artifact_manager=artifact_manager)
    live_plot_callback = None
    if live_plot_settings.get("enabled", True):
        plot_subdir = live_plot_settings.get("output_subdir", "optimization/plots/live")
        live_plot_dir = settings.OUTPUTS_DIR / Path(plot_subdir)
        live_plot_callback = LivePlotCallback(
            output_dir=live_plot_dir,
            max_trials_axis=live_plot_settings.get("max_trials_axis", 50),
            expected_trials=expected_trials,
        )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=42,
            multivariate=bool(tpe_settings.get("multivariate", True)),
            group=bool(tpe_settings.get("group", True)),
            n_startup_trials=n_startup,
            constant_liar=bool(tpe_settings.get("constant_liar", False)),
        ),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=reduction_factor,
        ),
        storage=storage_url,
        load_if_exists=True,
    )

    warmstart_injected = 0
    if warmstart_callback:
        warmstart_injected = warmstart_callback(study) or 0

    completed_trials_count = sum(
        1
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and not t.system_attrs.get("warmstart_seed")
    )

    # For fresh runs (resume_mode=False), do not subtract completed trials; honor requested n_trials
    effective_completed = 0 if not resume_mode else completed_trials_count

    total_target_trials = max(expected_trials, n_trials)
    remaining_trials = max(total_target_trials - effective_completed, 0)

    checkpoint_payload = {
        "status": "running",
        "study_name": study_name,
        "expected_trials": total_target_trials,
        "completed_trials": completed_trials_count,
        "resume_mode": resume_mode,
        "last_update": datetime.now(timezone.utc).isoformat(),
    }
    _write_checkpoint(checkpoint_path, checkpoint_payload)

    logger.info(f"Estudo Optuna criado: {study_name}")
    logger.info(f"Amostrador ativo: {study.sampler.__class__.__name__}")
    logger.info(f"Pruner configurado: {study.pruner.__class__.__name__}")
    logger.info(f"Modelos serão salvos em: {output_dir / 'best_models'}")

    if remaining_trials > 0:
        logger.info(
            f"Iniciando otimização com {remaining_trials} trials pendentes (alvo total: {total_target_trials})."
        )
        if warmstart_injected > 0:
            logger.info(f"Warm-start seeds carregados: {warmstart_injected} (não contam como trials completos).")
    else:
        logger.info("Nenhum trial pendente. Os resultados existentes já atingem o alvo configurado.")

    start_time = time.time()

    def cleanup_after_trial(study_obj, trial_obj):
        """Force cleanup after each trial to prevent segfaults."""
        gc.collect()
        try:
            if hasattr(study_obj, "_storage") and hasattr(study_obj._storage, "_engine"):
                study_obj._storage._engine.dispose()
        except Exception:
            pass

    def interruptible_objective(trial: optuna.trial.Trial) -> float:
        check_interruption()
        return float(objective_fn(trial))

    interrupted = False
    if remaining_trials > 0:
        try:
            study.optimize(
                interruptible_objective,
                n_trials=remaining_trials,
                n_jobs=1,
                callbacks=[cb for cb in [model_saver_callback, live_plot_callback, cleanup_after_trial] if cb],
                gc_after_trial=True,
            )
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("Optuna study interrupted by user; returning partial results")

    checkpoint_payload["status"] = "completed" if not interrupted else "interrupted"
    checkpoint_payload["completed_trials"] = len(study.trials)
    checkpoint_payload["last_update"] = datetime.now(timezone.utc).isoformat()
    _write_checkpoint(checkpoint_path, checkpoint_payload)

    optimization_time = time.time() - start_time
    try:
        best_params = study.best_params
        best_value = study.best_value
    except Exception:
        logger.warning("No completed trials; returning empty best_params")
        best_params, best_value = {}, None

    result = {
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": len(study.trials),
        "optimization_time": optimization_time,
        "framework": "optuna",
        "study": study,
        "trials": [],
        "interrupted": interrupted,
    }
    if live_plot_callback:
        result["live_plot_dir"] = str(live_plot_callback.output_dir)
    for trial in study.trials:
        result["trials"].append(
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
            }
        )
    return result

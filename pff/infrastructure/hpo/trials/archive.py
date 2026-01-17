"""Archive and reset HPO trials while preserving top-performing models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
import polars as pl

from pff.shared import logger
from pff.shared.core.file_manager import FileManager
from .config_loader import load_scoring_settings
from pff.infrastructure.hpo.config_loader import load_storage_settings
from pff.infrastructure.hpo.storage import create_optuna_storage
from .postgres_store import HpoPostgresStore
from pff.domain.hpo.scoring import (
    build_weights_from_settings,
    compute_score,
    rename_metric_keys,
)


def _load_completed_trials(
    storage_path: Path, study_name: str, file_manager: FileManager
) -> list[Any]:
    """Load completed Optuna trials from storage."""
    storage, storage_url = create_optuna_storage(
        storage_path=storage_path, file_manager=file_manager
    )
    study = optuna.load_study(
        study_name=study_name, storage=storage if storage is not None else storage_url
    )
    return [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]


def _compute_scores_for_trials(trials: list[Any], weights) -> list[dict[str, Any]]:
    """Compute scores for archived trials using the unified scoring function."""
    trial_metrics: list[dict[str, Any]] = []
    for trial in trials:
        renamed = rename_metric_keys(dict(getattr(trial, "user_attrs", {})))
        renamed["score"] = float(trial.value)
        trial_metrics.append(renamed)

    scored_trials: list[dict[str, Any]] = []
    for trial, metrics in zip(trials, trial_metrics, strict=False):
        history = [m for m in trial_metrics if m is not metrics]
        score, _, components = compute_score(metrics, history, weights=weights)
        scored_trials.append(
            {
                "number": trial.number,
                "score": score,
                "params": dict(trial.params),
                "metrics": metrics,
                "components": {
                    "rank": components.rank,
                    "classification": components.classification,
                    "efficiency": components.efficiency,
                },
            }
        )
    return scored_trials


def archive_and_reset_trials(
    output_dir: Path,
    *,
    study_name: str,
    top_n: int = 5,
    store: HpoPostgresStore | None = None,
    file_manager: FileManager | None = None,
) -> None:
    """
    Archive the best trials and reset the working directory for a fresh HPO run.

    Args:
        output_dir: Directory containing the optuna_study.db and trial outputs.
        study_name: Name of the study in the Optuna storage.
        top_n: Number of best trials to preserve in history.
        file_manager: Optional FileManager instance.
    """
    fm = file_manager or FileManager()
    storage_path = output_dir / "optuna_study.db"
    storage_backend = str(load_storage_settings(fm).get("backend", "sqlite")).lower()
    if storage_backend not in {
        "postgres",
        "postgresql",
        "rdb",
        "rdbstorage",
        "grpc",
        "grpc_proxy",
    }:
        if not fm.exists(storage_path):
            logger.info("Nenhum arquivo de estudo encontrado; reset ignorado.")
            return

    try:
        scoring_settings = load_scoring_settings(fm)
        weights = build_weights_from_settings(scoring_settings)
        trials = _load_completed_trials(storage_path, study_name, fm)
        if not trials:
            logger.info("Nenhum trial completo para arquivar; removendo estado atual.")
        scored_trials = _compute_scores_for_trials(trials, weights)
        scored_sorted = sorted(scored_trials, key=lambda t: t["score"], reverse=True)
        best_trials = scored_sorted[:top_n]

        history_dir = output_dir / "history" / fm.get_timestamp().replace(":", "-")
        fm.ensure_dir(history_dir)
        df = pl.DataFrame(best_trials)
        fm.save(df, history_dir / "top_trials.parquet")

        for trial in best_trials:
            trial_dir = output_dir / "trials" / f"trial_{trial['number']:04d}"
            if fm.exists(trial_dir):
                dest = history_dir / "trials" / f"trial_{trial['number']:04d}"
                fm.copy_directory(trial_dir, dest)

        best_models_dir = output_dir / "best_models"
        if fm.exists(best_models_dir):
            fm.copy_directory(best_models_dir, history_dir / "best_models")

        logger.success(
            f"Trials arquivados em {history_dir} e melhores modelos preservados (top {len(best_trials)})."
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to archive trials before reset: {exc}")

    fm.delete_file(storage_path, ignore_errors=True)
    fm.delete_directory(output_dir / "trials", ignore_errors=True)
    fm.delete_directory(output_dir / "best_models", ignore_errors=True)
    fm.delete_file(output_dir / "checkpoint.json", ignore_errors=True)
    fm.delete_file(output_dir / "best_params.json", ignore_errors=True)
    if store is not None:
        try:
            from pff.shared.acceleration.asyncio_runner import run_coroutine_sync

            run_coroutine_sync(store.clear_study(study_name))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to clear Postgres HPO state: {exc}")
    logger.info("Estado de HPO resetado; pronto para iniciar novos trials.")

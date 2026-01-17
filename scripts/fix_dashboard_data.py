"""
Script de recuperação para gerar manualmente o arquivo de dados do dashboard HPO.
Útil quando o callback automático falha ou o processo HPO está desconectado do dashboard.
"""

import json
from datetime import datetime, timezone
from typing import Any

import optuna
from optuna.trial import TrialState
from optuna.importance import get_param_importances, FanovaImportanceEvaluator

from pff import settings


try:
    from pff.infrastructure.hpo.callbacks_internal.collectors import (
        flatten_trial_metrics,
    )
except ImportError:

    def flatten_trial_metrics(trial: Any) -> dict[str, float]:
        """Extract metrics from trial user attrs and intermediate values."""
        metrics = {}
        # 1. User attributes (final metrics)
        for k, v in trial.user_attrs.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
        return metrics


def _infer_duration(trial: Any) -> float:
    """Try various ways to find the trial duration."""
    # 1. From timestamps
    if trial.datetime_complete and trial.datetime_start:
        d = (trial.datetime_complete - trial.datetime_start).total_seconds()
        if d > 0:
            return float(d)

    # 2. From Optuna duration attribute if it exists
    optuna_dur = getattr(trial, "duration", None)
    if optuna_dur:
        try:
            return float(optuna_dur.total_seconds())
        except Exception:  # noqa: BLE001
            pass

    # 3. From user_attrs (we often save it there as a fallback)
    attrs = trial.user_attrs or {}
    for key in ["duration", "elapsed_time", "time"]:
        if key in attrs and isinstance(attrs[key], (int, float)):
            return float(attrs[key])

    return 0.0


def _trial_primary_value(trial: Any) -> float:
    value = getattr(trial, "value", None)
    if value is None:
        values = getattr(trial, "values", None)
        if values and len(values) > 0:
            value = values[0]
    return float(value) if value is not None else 0.0


def export_dashboard_data(study_name: str):
    """Export study data to JSON for the dashboard."""

    # Connect to storage
    url = f"postgresql+psycopg2://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    study = optuna.load_study(study_name=study_name, storage=url)

    trials = list(study.get_trials(deepcopy=False))
    completed_trials = [t for t in trials if t.state == TrialState.COMPLETE]

    # Export COMPLETE and RUNNING trials to ensure we see params for current run
    valid_states = {
        TrialState.COMPLETE,
        TrialState.RUNNING,
        TrialState.PRUNED,
        TrialState.FAIL,
    }
    exportable_trials = [t for t in trials if t.state in valid_states]

    # Collect trial data
    trials_data = []
    for t in exportable_trials:
        m = flatten_trial_metrics(t)
        primary_value = _trial_primary_value(t)
        duration = _infer_duration(t)

        trials_data.append(
            {
                "id": t.number + 1,
                "value": primary_value,
                "state": str(t.state.name),
                "params": t.params if hasattr(t, "params") else {},
                "duration": duration,
                "datetime_start": (
                    t.datetime_start.isoformat() if t.datetime_start else None
                ),
                "datetime_complete": (
                    t.datetime_complete.isoformat() if t.datetime_complete else None
                ),
                # Extra metrics
                "mrr": m.get("mrr", m.get("kge_mrr", 0.0)),
                "best_mrr": m.get("best_mrr", m.get("kge_best_mrr", 0.0)),
                "mcc": m.get("mcc", 0.0),
                "auc": m.get("auc", 0.0),
                "hits1": m.get("hits@1", m.get("hits1", 0.0)),
                "hits3": m.get("hits@3", m.get("hits3", 0.0)),
                "hits10": m.get("hits@10", m.get("hits10", 0.0)),
            }
        )

    best_value = 0.0
    if completed_trials:
        try:
            best_value = float(study.best_value)
        except Exception:
            best_value = max(_trial_primary_value(t) for t in completed_trials)

    updated_at = datetime.now(timezone.utc).isoformat()

    # Calculate fANOVA Importances
    param_importances = {}
    if len(completed_trials) > 3:
        try:
            evaluator = FanovaImportanceEvaluator(n_trees=32, seed=42)
            importances = get_param_importances(study, evaluator=evaluator)
            param_importances = {k: float(v) for k, v in importances.items()}
        except Exception as e:
            print(f"Importance calc failed: {e}")
            pass

    payload = {
        "studyName": study_name,
        "updatedAt": updated_at,
        "bestValue": best_value,
        "trials": trials_data,
        "importances": param_importances,
        "totalTrials": 100,  # Hardcoded or fetch from config
    }

    # Write file
    cache_dir = settings.CACHE_DIR / "hpo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / "dashboard_data.json"

    print(f"Writing dashboard data to {data_path}")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    export_dashboard_data("pff_kg_real_dslfm_kgc")

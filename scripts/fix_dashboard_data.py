"""
Script de recuperação para gerar manualmente o arquivo de dados do dashboard HPO.
Útil quando o callback automático falha ou o processo HPO está desconectado do dashboard.
"""

from datetime import datetime, timezone
from typing import Any

import optuna
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.trial import TrialState

from pff import settings

try:
    from pff.infrastructure.hpo.callbacks_internal.collectors import (
        flatten_trial_metrics,
    )
except ImportError:

    def flatten_trial_metrics(trial: Any) -> dict[str, float]:
        """Extract metrics from trial user attrs and intermediate values."""
        metrics = {}

        for k, v in trial.user_attrs.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
        return metrics


def _infer_duration(trial: Any) -> float:
    """Try various ways to find the trial duration."""

    if trial.datetime_complete and trial.datetime_start:
        d = (trial.datetime_complete - trial.datetime_start).total_seconds()
        if d > 0:
            return float(d)

    optuna_dur = getattr(trial, "duration", None)
    if optuna_dur:
        try:
            return float(optuna_dur.total_seconds())
        except Exception:
            pass

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

    url = f"postgresql+psycopg2://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    study = optuna.load_study(study_name=study_name, storage=url)

    trials = list(study.get_trials(deepcopy=False))
    completed_trials = [t for t in trials if t.state == TrialState.COMPLETE]

    valid_states = {
        TrialState.COMPLETE,
        TrialState.RUNNING,
        TrialState.PRUNED,
        TrialState.FAIL,
    }
    exportable_trials = [t for t in trials if t.state in valid_states]

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
                "datetime_start": (t.datetime_start.isoformat() if t.datetime_start else None),
                "datetime_complete": (
                    t.datetime_complete.isoformat() if t.datetime_complete else None
                ),
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
        "totalTrials": 100,
    }

    cache_dir = settings.CACHE_DIR / "hpo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / "dashboard_data.json"

    print(f"Writing dashboard data to {data_path}")
    from pff.shared import FileManager

    FileManager().save(payload, data_path)


if __name__ == "__main__":
    export_dashboard_data("pff_kg_real_dslfm_kgc")

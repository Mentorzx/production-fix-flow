"""Metric collection helpers for optimization callbacks."""

from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    from pff.domain.hpo.scoring import rename_metric_keys
except Exception:  # pragma: no cover - fallback for alternate entrypoints
    from ..trials.scoring import rename_metric_keys


def flatten_trial_metrics(trial: Any) -> dict[str, float]:
    """Normalize trial metrics into a flat dictionary.

    Args:
        trial: Optuna trial object.

    Returns:
        Dict with normalized metric keys and float values.
    """
    attrs = dict(getattr(trial, "user_attrs", {}) or {})
    for key in ("metrics", "kge_metrics", "clf_metrics"):
        nested = attrs.get(key)
        if isinstance(nested, dict):
            attrs.update(nested)
    normalized = rename_metric_keys(attrs)
    score_value = getattr(trial, "value", None)
    if score_value is None:
        values = getattr(trial, "values", None)
        if isinstance(values, (list, tuple)) and values:
            score_value = values[0]
    normalized.setdefault("score", float(score_value or 0.0))
    duration = float(normalized.get("duration", attrs.get("duration", 0.0) or 0.0))
    if duration <= 0.0:
        fallback = _infer_trial_duration_seconds(trial)
        if fallback is not None:
            duration = float(fallback)
    normalized.setdefault("duration", duration)
    return normalized


def _infer_trial_duration_seconds(trial: Any) -> float | None:
    """Infer trial duration from Optuna timestamps when user_attrs are missing."""
    duration = getattr(trial, "duration", None)
    if duration is not None:
        try:
            return float(getattr(duration, "total_seconds", lambda: duration)())
        except Exception:
            pass
    start = getattr(trial, "datetime_start", None)
    complete = getattr(trial, "datetime_complete", None)
    if isinstance(start, datetime) and isinstance(complete, datetime):
        try:
            return max(0.0, float((complete - start).total_seconds()))
        except Exception:
            return None
    return None


def flatten_trials_batch(trials: list[Any]) -> list[dict[str, float]]:
    """Normalize a batch of trials into flat dictionaries efficiently.

    Args:
        trials: List of Optuna trial objects.

    Returns:
        List of dicts with normalized metric keys.
    """
    return [flatten_trial_metrics(t) for t in trials]


def extract_metric_series(trials: list[Any], metric_key: str) -> list[float]:
    """Extract metric values across trials efficiently.

    Args:
        trials: Iterable of Optuna trials.
        metric_key: Metric name to extract.

    Returns:
        List of metric values.
    """
    # Performance optimization: extract directly from pre-flattened if available
    # or use a single pass for lookups
    values: list[float] = []
    for trial in trials:
        # Check if the object already has pre-flattened metrics
        if hasattr(trial, "_cached_metrics"):
            metrics = trial._cached_metrics
        else:
            metrics = flatten_trial_metrics(trial)
            try:
                trial._cached_metrics = metrics
            except Exception:  # noqa: BLE001 - fallback if object is frozen
                pass

        if metric_key in metrics:
            values.append(float(metrics[metric_key]))
    return values

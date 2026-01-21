"""
Multi-objective trial selection helpers for DSLFM/PC HPO.

Chooses three champions:
- time-aware winner (uses full score with tempo)
- quality-only winner (ignores tempo in the aggregation)
- trade-off winner (best quality/tempo ratio among the two champions)
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pff.shared import logger

from .scoring import (
    ScoreComponents,
    ScoreWeights,
    build_weights_from_settings,
    compute_score,
    rename_metric_keys,
)


@dataclass(frozen=True)
class TrialSelectionEntry:
    """Container for per-trial scores used in multi-objective ranking."""

    trial_number: int
    params: dict[str, Any]
    duration: float
    score_time: float
    score_quality: float
    tradeoff_score: float
    metrics: dict[str, float]
    components: ScoreComponents


def _quality_only_weights(base: ScoreWeights) -> ScoreWeights:
    """Build a ScoreWeights variant with tempo removido (time_block=0)."""
    total = max(base.rank_block + base.clf_block, base.eps)
    rank_block = base.rank_block / total if total > 0 else 0.0
    clf_block = base.clf_block / total if total > 0 else 0.0
    return ScoreWeights(
        rank_block=rank_block,
        clf_block=clf_block,
        time_block=0.0,
        rank_metrics=dict(base.rank_metrics),
        clf_metrics=dict(base.clf_metrics),
        duration_weight=0.0,
        eps=base.eps,
        time_scale=base.time_scale,
    )


def _time_tradeoff(score_quality: float, duration: float, eps: float) -> float:
    """Compute quality/tempo ratio with soft penalty on duration."""
    safe_duration = max(duration, eps)
    return score_quality / max(math.log1p(safe_duration), eps)


def _normalize_trials(trials: Iterable[Any]) -> list[Any]:
    """Filter completed trials only (Optuna or plain objects)."""
    normalized: list[Any] = []
    try:
        from optuna.trial import TrialState

        completed_state = TrialState.COMPLETE
    except Exception:
        completed_state = None

    for trial in trials:
        state = getattr(trial, "state", None)
        if completed_state is not None:
            if state != completed_state:
                if not (isinstance(state, str) and state.lower() in {"complete", "completed"}):
                    continue
        elif isinstance(state, str) and state.lower() not in {"complete", "completed"}:
            continue
        value = getattr(trial, "value", None)
        if value is None:
            continue
        normalized.append(trial)
    return normalized


def _build_entry(
    trial: Any,
    metrics: dict[str, float],
    history: list[dict[str, float]],
    *,
    weights_time: ScoreWeights,
    weights_quality: ScoreWeights,
) -> TrialSelectionEntry:
    score_time, _, comps_time = compute_score(metrics, history, weights=weights_time)
    score_quality, _, _ = compute_score(metrics, history, weights=weights_quality)
    duration = float(metrics.get("duration", 0.0))
    tradeoff_score = _time_tradeoff(score_quality, duration, weights_time.eps)
    return TrialSelectionEntry(
        trial_number=int(getattr(trial, "number", -1)),
        params=dict(getattr(trial, "params", {})),
        duration=duration,
        score_time=score_time,
        score_quality=score_quality,
        tradeoff_score=tradeoff_score,
        metrics=metrics,
        components=comps_time,
    )


def _default_payload() -> dict[str, Any]:
    return {
        "best_time_aware": None,
        "best_quality": None,
        "best_tradeoff": None,
        "tradeoff_formula": "score_quality / log1p(duration_seconds)",
    }


def select_best_trials(
    study: Any | None,
    *,
    weights: ScoreWeights | None = None,
) -> dict[str, Any]:
    """
    Pick champions for tempo-aware, quality-only and trade-off criteria.

    Args:
        study: Optuna study ou objeto com lista de trials.
        weights: ScoreWeights para o caminho tempo-aware (opcional).

    Returns:
        Dict com best_time_aware, best_quality, best_tradeoff e formula usada.
    """
    if study is None:
        return _default_payload()

    trials = _normalize_trials(getattr(study, "trials", []))
    if not trials:
        return _default_payload()

    if weights is None:
        weights = build_weights_from_settings({})
    quality_weights = _quality_only_weights(weights)

    metrics_history: list[dict[str, float]] = []
    for trial in trials:
        renamed = rename_metric_keys(dict(getattr(trial, "user_attrs", {}) or {}))
        renamed.setdefault("score", float(getattr(trial, "value", 0.0) or 0.0))
        renamed.setdefault("duration", float(renamed.get("duration", 0.0)))
        metrics_history.append(renamed)

    entries: list[TrialSelectionEntry] = []
    for idx, trial in enumerate(trials):
        metrics = metrics_history[idx]
        history = metrics_history[:idx] + metrics_history[idx + 1 :]
        try:
            entry = _build_entry(
                trial,
                metrics,
                history,
                weights_time=weights,
                weights_quality=quality_weights,
            )
            entries.append(entry)
        except Exception as exc:
            logger.warning(f"Failed to evaluate trial {getattr(trial, 'number', '?')}: {exc}")

    if not entries:
        return _default_payload()

    best_time = max(entries, key=lambda e: e.score_time)
    best_quality = max(entries, key=lambda e: e.score_quality)
    best_tradeoff = max(entries, key=lambda e: e.tradeoff_score)

    def _payload(entry: TrialSelectionEntry) -> dict[str, Any]:
        return {
            "trial_number": entry.trial_number,
            "params": entry.params,
            "duration": entry.duration,
            "score_time": entry.score_time,
            "score_quality": entry.score_quality,
            "tradeoff_score": entry.tradeoff_score,
            "metrics": entry.metrics,
            "components": {
                "rank": entry.components.rank,
                "classification": entry.components.classification,
                "efficiency": entry.components.efficiency,
            },
        }

    return {
        "best_time_aware": _payload(best_time),
        "best_quality": _payload(best_quality),
        "best_tradeoff": _payload(best_tradeoff),
        "tradeoff_formula": "score_quality / log1p(duration_seconds)",
    }

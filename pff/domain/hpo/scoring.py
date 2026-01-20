"""Scoring utilities for DSLFM/PC HPO trials.

Implements a multi-metric score with min-max normalization across trials and an
open-interval clamp to avoid exact 0/1 results. Metrics are grouped into rank,
classification and efficiency blocks and combined with explicit weights.

The time/efficiency block uses a **physical time scale** with logarithmic decay,
ensuring that scores near 1.0 are only achievable with near-zero durations
(physically impossible), while realistic durations produce meaningfully lower scores.

Design patterns:
- Strategy/Template Method (functional composition of normalization + weighting)
- Builder-style helpers for score components
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

DEFAULT_EPS = 0.02


@dataclass(frozen=True)
class TimeScaleConfig:
    """Physical time scale configuration for efficiency scoring.

    The time score uses a logarithmic scale calibrated to these anchor points,
    ensuring that realistic durations produce meaningfully penalized scores
    while only impossible (near-zero) durations approach 1.0.

    Attributes:
        t_best: Ideal duration in seconds (e.g., 1s). Score at this point is score_at_best.
        t_target: Acceptable duration with visible penalty (e.g., 50s).
        t_worst: Poor duration threshold (e.g., 300s). Score approaches score_at_worst.
        score_at_best: Score when duration == t_best (should be < 1, e.g., 0.90).
        score_at_target: Score when duration == t_target (e.g., 0.65).
        score_at_worst: Score when duration >= t_worst (e.g., 0.05).
    """

    t_best: float = 1.0
    t_target: float = 50.0
    t_worst: float = 300.0
    score_at_best: float = 0.88
    score_at_target: float = 0.55
    score_at_worst: float = 0.20


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for score aggregation."""

    rank_block: float
    clf_block: float
    time_block: float
    rank_metrics: dict[str, float]
    clf_metrics: dict[str, float]
    duration_weight: float
    eps: float = DEFAULT_EPS
    time_scale: TimeScaleConfig = field(default_factory=TimeScaleConfig)


@dataclass(frozen=True)
class ScoreComponents:
    """Container for per-block scores."""

    rank: float
    classification: float
    efficiency: float


def rename_metric_keys(metrics: dict[str, Any]) -> dict[str, float]:
    """Normalize metric key names to the short schema used across the HPO stack.

    Args:
        metrics: Raw metrics dictionary (possibly with legacy keys).

    Returns:
        Metrics dictionary with normalized keys (score, mrr, best_mrr, hits1, hits3, hits10, auc,
        pr_auc, precision, recall, duration).
    """
    mapping = {
        "score_composto": "score",
        "kge_mrr": "mrr",
        "kge_best_mrr": "best_mrr",
        "best_val_mrr": "best_mrr",
        "hits@1": "hits1",
        "hits@3": "hits3",
        "hits@10": "hits10",
        "kge_hits@1": "hits1",
        "kge_hits@3": "hits3",
        "kge_hits@10": "hits10",
        "elapsed_time": "duration",
        "trial_time": "duration",
        "tempo": "duration",
        "ap@10": "ap10",
        "mcc": "mcc",
    }
    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        target = mapping.get(key, key)
        try:
            normalized[target] = float(value)
        except Exception:
            continue
    if "best_mrr" not in normalized and "mrr" in normalized:
        normalized["best_mrr"] = normalized["mrr"]
    return normalized


def _to_open_interval(value: float, eps: float) -> float:
    """Map [0, 1] -> (eps, 1-eps) to avoid exact 0/1 boundary values.

    This is essential for numerical stability in optimization and prevents
    degenerate behavior at score boundaries.

    Args:
        value: Input value in [0, 1] range.
        eps: Small epsilon to define open interval boundaries.

    Returns:
        Value mapped to (eps, 1-eps) open interval.
    """
    return max(eps, min(1.0 - eps, eps + (1.0 - 2 * eps) * value))


def compute_physical_time_score(
    duration: float,
    time_scale: TimeScaleConfig,
    eps: float = DEFAULT_EPS,
) -> float:
    """Compute time score using physical/absolute scale with piecewise log interpolation.

    This function implements a time penalty with **physical semantics**:
    - Scores near 1.0 are only achievable with near-zero durations (impossible)
    - Realistic durations (>= t_best) produce meaningfully lower scores
    - The function is monotonically decreasing and calibrated to all 3 anchor points

    The piecewise logarithmic scale ensures:
    - Exact calibration at t_best, t_target, and t_worst
    - Small time differences at low durations have larger score impact
    - Large time differences at high durations have diminishing impact

    Args:
        duration: Trial duration in seconds.
        time_scale: TimeScaleConfig with anchor points and target scores.
        eps: Small value to keep result in open interval (0, 1).

    Returns:
        Time score in (eps, 1-eps), where higher is better (faster).

    Example calibration with defaults (t_best=1, t_target=50, t_worst=300):
        - duration=0.1s -> score ≈ 0.95 (near theoretical max)
        - duration=1s   -> score ≈ 0.90 (score_at_best)
        - duration=50s  -> score ≈ 0.65 (score_at_target)
        - duration=300s -> score ≈ 0.05 (score_at_worst)
        - duration=600s -> score ≈ 0.02 (heavily penalized)
    """
    t_best = max(time_scale.t_best, eps)
    t_target = max(time_scale.t_target, t_best + eps)
    t_worst = max(time_scale.t_worst, t_target + eps)
    score_at_best = time_scale.score_at_best
    score_at_target = time_scale.score_at_target
    score_at_worst = time_scale.score_at_worst

    d = max(duration, eps)

    log_best = math.log(t_best)
    log_target = math.log(t_target)
    log_worst = math.log(t_worst)
    log_d = math.log(d)

    if d <= t_best:
        bonus_range = 1.0 - score_at_best
        bonus_factor = (t_best - d) / t_best
        bonus = bonus_range * bonus_factor * 0.5
        raw_score = min(score_at_best + bonus, 1.0 - eps)
    elif d <= t_target:
        log_range = log_target - log_best
        if log_range < eps:
            x = 0.5
        else:
            x = (log_d - log_best) / log_range
        raw_score = score_at_best + x * (score_at_target - score_at_best)
    elif d <= t_worst:
        log_range = log_worst - log_target
        if log_range < eps:
            x = 0.5
        else:
            x = (log_d - log_target) / log_range
        raw_score = score_at_target + x * (score_at_worst - score_at_target)
    else:
        overshoot = (log_d - log_worst) / (log_worst - log_target)
        decay = math.pow(0.5, overshoot)
        raw_score = max(score_at_worst * decay, eps)

    return _to_open_interval(raw_score, eps)


def _compute_min_max(values: list[float]) -> tuple[float, float] | None:
    """Compute min and max of a list, filtering out NaN values.

    Args:
        values: List of float values (may contain NaNs).

    Returns:
        Tuple of (min, max) or None if no valid values exist.
    """
    filtered = [v for v in values if not math.isnan(v)]
    if not filtered:
        return None
    return min(filtered), max(filtered)


def _normalize_value(
    value: float | None,
    *,
    min_v: float,
    max_v: float,
    eps: float,
    invert: bool = False,
) -> float:
    """Normalize a value to (eps, 1-eps) using min-max scaling.

    Args:
        value: Raw metric value to normalize (may be None or NaN).
        min_v: Minimum value in the reference range.
        max_v: Maximum value in the reference range.
        eps: Epsilon for open interval mapping.
        invert: If True, smaller values produce higher scores.

    Returns:
        Normalized value in (eps, 1-eps) open interval.
    """
    if value is None or math.isnan(value):
        return _to_open_interval(0.0, eps)
    if math.isclose(max_v, min_v):
        return _to_open_interval(0.0, eps)
    if invert:
        ref = max(max_v, eps)
        norm = 1.0 / (1.0 + max(value, 0.0) / ref)
        norm = max(0.0, min(1.0, norm))
        return _to_open_interval(norm, eps)
    norm = (value - min_v) / (max_v - min_v)
    norm = max(0.0, min(1.0, norm))
    return _to_open_interval(norm, eps)


def _absolute_value(value: float | None, eps: float) -> float:
    """Use absolute value for metrics already in [0, 1] range.

    Unlike min-max normalization, this preserves the actual metric value.
    A precision of 0.05 stays at ~0.05, not normalized to 1.0 because
    it's the "best" among trials.

    Args:
        value: Raw metric value (should be in [0, 1] range).
        eps: Epsilon for open interval mapping.

    Returns:
        Value mapped to (eps, 1-eps) open interval.
    """
    if value is None or math.isnan(value):
        return _to_open_interval(0.0, eps)
    clamped = max(0.0, min(1.0, value))
    return _to_open_interval(clamped, eps)


def _aggregate_block(values: dict[str, float], weights: dict[str, float]) -> float:
    """Compute weighted average of values using specified weights.

    Implements weighted mean aggregation for score blocks (rank, classification).
    Skips keys with zero or negative weights, and keys not present in values.

    Args:
        values: Dictionary mapping metric names to normalized values.
        weights: Dictionary mapping metric names to their relative weights.

    Returns:
        Weighted average of values, or 0.0 if no valid weights.
    """
    weighted_sum = 0.0
    total = 0.0
    for key, weight in weights.items():
        if weight <= 0:
            continue
        if key not in values:
            continue
        weighted_sum += values[key] * weight
        total += weight
    if total == 0.0:
        return 0.0
    return weighted_sum / total


def _default_weights() -> ScoreWeights:
    return ScoreWeights(
        rank_block=0.25,
        clf_block=0.65,
        time_block=0.10,
        rank_metrics={
            "mrr": 0.40,
            "best_mrr": 0.30,
            "hits1": 0.15,
            "hits3": 0.10,
            "hits10": 0.05,
        },
        clf_metrics={
            "mcc": 0.50,
            "ap": 0.20,
            "pr_auc": 0.20,
            "auc": 0.05,
            "precision": 0.03,
            "recall": 0.02,
        },
        duration_weight=1.0,
        eps=DEFAULT_EPS,
        time_scale=TimeScaleConfig(),
    )


def build_weights_from_settings(settings: dict[str, Any]) -> ScoreWeights:
    """Build ScoreWeights from a plain settings dict (as loaded by config_loader).

    Args:
        settings: Dict produced by `load_scoring_settings`.

    Returns:
        ScoreWeights instance.
    """
    defaults = _default_weights()

    time_scale_cfg = settings.get("time_scale", {})
    time_scale = TimeScaleConfig(
        t_best=float(time_scale_cfg.get("t_best", defaults.time_scale.t_best)),
        t_target=float(time_scale_cfg.get("t_target", defaults.time_scale.t_target)),
        t_worst=float(time_scale_cfg.get("t_worst", defaults.time_scale.t_worst)),
        score_at_best=float(time_scale_cfg.get("score_at_best", defaults.time_scale.score_at_best)),
        score_at_target=float(
            time_scale_cfg.get("score_at_target", defaults.time_scale.score_at_target)
        ),
        score_at_worst=float(
            time_scale_cfg.get("score_at_worst", defaults.time_scale.score_at_worst)
        ),
    )

    return ScoreWeights(
        rank_block=float(settings.get("weights", {}).get("rank_block", defaults.rank_block)),
        clf_block=float(settings.get("weights", {}).get("clf_block", defaults.clf_block)),
        time_block=float(settings.get("weights", {}).get("time_block", defaults.time_block)),
        rank_metrics={
            "best_mrr": float(
                settings.get("rank_metrics", {}).get("best_mrr", defaults.rank_metrics["best_mrr"])
            ),
            "mrr": float(settings.get("rank_metrics", {}).get("mrr", defaults.rank_metrics["mrr"])),
            "hits1": float(
                settings.get("rank_metrics", {}).get("hits1", defaults.rank_metrics["hits1"])
            ),
            "hits3": float(
                settings.get("rank_metrics", {}).get("hits3", defaults.rank_metrics["hits3"])
            ),
            "hits10": float(
                settings.get("rank_metrics", {}).get("hits10", defaults.rank_metrics["hits10"])
            ),
        },
        clf_metrics={
            "auc": float(settings.get("clf_metrics", {}).get("auc", defaults.clf_metrics["auc"])),
            "pr_auc": float(
                settings.get("clf_metrics", {}).get("pr_auc", defaults.clf_metrics["pr_auc"])
            ),
            "mcc": float(settings.get("clf_metrics", {}).get("mcc", defaults.clf_metrics["mcc"])),
            "ap": float(settings.get("clf_metrics", {}).get("ap", defaults.clf_metrics["ap"])),
            "precision": float(
                settings.get("clf_metrics", {}).get("precision", defaults.clf_metrics["precision"])
            ),
            "recall": float(
                settings.get("clf_metrics", {}).get("recall", defaults.clf_metrics["recall"])
            ),
        },
        duration_weight=float(settings.get("time_metric_weight", defaults.duration_weight)),
        eps=float(settings.get("eps", defaults.eps)),
        time_scale=time_scale,
    )


def _collect_metric_span(
    all_metrics: Iterable[dict[str, float]],
    keys: Iterable[str],
) -> dict[str, tuple[float, float]]:
    """Collect min-max spans for each metric key across all trial metrics.

    Used for min-max normalization of rank and classification metrics.
    Duration uses physical scaling instead of this span-based approach.

    Args:
        all_metrics: Iterable of metric dictionaries from all trials.
        keys: Metric keys to compute spans for.

    Returns:
        Dictionary mapping metric keys to (min, max) tuples.
    """
    spans: dict[str, tuple[float, float]] = {}
    for key in keys:
        values = []
        for metrics in all_metrics:
            try:
                val = metrics.get(key)
                if val is not None:
                    values.append(float(val))
            except Exception:
                continue
        span = _compute_min_max(values)
        if span is not None:
            spans[key] = span
    return spans


def compute_score(
    current_metrics: dict[str, Any],
    history_metrics: list[dict[str, Any]],
    *,
    weights: ScoreWeights | None = None,
) -> tuple[float, dict[str, float], ScoreComponents]:
    """Compute the final score for a trial using all relevant metrics.

    Uses **physical time scaling** for the efficiency block: the time score is
    computed using absolute anchor points (t_best, t_target, t_worst) rather than
    relative min-max normalization. This ensures that:
    - Scores near 1.0 are only achievable with near-zero durations (impossible)
    - Realistic durations produce meaningfully lower scores
    - The score has physical/semantic meaning, not just relative ranking

    Args:
        current_metrics: Metrics for the current trial (raw values).
        history_metrics: Metrics of previous trials (raw values) for min-max normalization.
        weights: ScoreWeights configuration (defaults are used if None).

    Returns:
        Tuple with (score in (0,1), normalized metrics, score components).
    """
    weights = weights or _default_weights()
    renamed_current = rename_metric_keys(current_metrics)
    [rename_metric_keys(item) for item in history_metrics]

    normalized: dict[str, float] = {}

    for key in weights.rank_metrics:
        normalized[key] = _absolute_value(renamed_current.get(key), weights.eps)

    for key in weights.clf_metrics:
        normalized[key] = _absolute_value(renamed_current.get(key), weights.eps)

    rank_block = _aggregate_block(normalized, weights.rank_metrics)
    clf_block = _aggregate_block(normalized, weights.clf_metrics)

    duration = renamed_current.get("duration", 0.0)
    efficiency_block = compute_physical_time_score(
        duration=duration,
        time_scale=weights.time_scale,
        eps=weights.eps,
    )
    efficiency_block *= weights.duration_weight

    normalized["duration"] = efficiency_block

    score_raw = (
        weights.rank_block * rank_block
        + weights.clf_block * clf_block
        + weights.time_block * efficiency_block
    )
    score = _to_open_interval(score_raw, weights.eps)
    components = ScoreComponents(
        rank=rank_block, classification=clf_block, efficiency=efficiency_block
    )
    return score, normalized, components

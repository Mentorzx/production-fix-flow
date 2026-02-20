"""Importance normalization and blending helpers."""

from __future__ import annotations

import math
from typing import Any, Callable


def normalize_importances(
    raw_importances: dict[str, float],
    *,
    search_space: dict[str, Any],
) -> dict[str, float]:
    """Normalize non-negative importances over current search-space keys."""
    filtered: dict[str, float] = {}
    for param_name in search_space:
        value = raw_importances.get(param_name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0:
            filtered[param_name] = float(value)
    total = sum(filtered.values())
    if total <= 0:
        return {}
    return {name: float(value / total) for name, value in filtered.items()}


def compute_internal_importances(
    *,
    completed_trials: list[Any],
    direction: str,
    search_space: dict[str, Any],
    spearman_rho_fn: Callable[[list[float], list[float]], float | None],
    apply_direction_fn: Callable[[float, str], float],
) -> tuple[dict[str, float], float]:
    """Estimate internal importances from correlation and variance signals."""
    if not completed_trials or not search_space:
        return {}, 0.0

    adjusted_scores = [
        apply_direction_fn(float(trial.value), direction) for trial in completed_trials
    ]
    if len(adjusted_scores) < 5:
        return {}, 0.0

    param_strength: dict[str, float] = {}
    for param_name in search_space:
        pairs = [
            (trial.params.get(param_name), score)
            for trial, score in zip(completed_trials, adjusted_scores, strict=False)
            if param_name in trial.params
        ]
        if len(pairs) < 5:
            continue
        numeric_values: list[float] = []
        numeric_scores: list[float] = []
        is_numeric = True
        for value, score in pairs:
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
                numeric_scores.append(float(score))
            else:
                is_numeric = False
                break
        if is_numeric:
            rho = spearman_rho_fn(numeric_values, numeric_scores)
            if rho is not None:
                param_strength[param_name] = abs(float(rho))
            continue

        groups: dict[str, list[float]] = {}
        for trial, score in zip(completed_trials, adjusted_scores, strict=False):
            if param_name not in trial.params:
                continue
            key = str(trial.params.get(param_name))
            groups.setdefault(key, []).append(score)
        if len(groups) < 2:
            continue
        global_mean = float(sum(adjusted_scores)) / float(len(adjusted_scores))
        total_var = sum((score - global_mean) ** 2 for score in adjusted_scores)
        if total_var <= 1e-12:
            continue
        between_var = 0.0
        for group_scores in groups.values():
            if not group_scores:
                continue
            group_mean = float(sum(group_scores)) / float(len(group_scores))
            between_var += float(len(group_scores)) * ((group_mean - global_mean) ** 2)
        eta2 = max(0.0, min(1.0, between_var / total_var))
        param_strength[param_name] = float(eta2)

    normalized = normalize_importances(param_strength, search_space=search_space)
    quality = float(len(normalized)) / float(max(1, len(search_space)))
    return normalized, quality


def resolve_importances(
    *,
    search_space: dict[str, Any],
    external_importances: dict[str, float],
    completed_trials: list[Any],
    direction: str,
    use_internal: bool,
    compute_internal_importances_fn: Callable[..., tuple[dict[str, float], float]],
) -> tuple[dict[str, float], str, float]:
    """Blend external/internal importances and return source metadata."""
    external = normalize_importances(external_importances, search_space=search_space)
    external_quality = min(1.0, float(len(external)) / float(max(1, len(search_space))))

    internal: dict[str, float] = {}
    internal_quality = 0.0
    if use_internal:
        internal, internal_quality = compute_internal_importances_fn(
            completed_trials=completed_trials,
            direction=direction,
            search_space=search_space,
        )

    if external and internal:
        alpha = max(0.2, min(0.8, external_quality))
        blended: dict[str, float] = {}
        for param_name in search_space:
            score = (alpha * external.get(param_name, 0.0)) + (
                (1.0 - alpha) * internal.get(param_name, 0.0)
            )
            if score > 0:
                blended[param_name] = score
        normalized = normalize_importances(blended, search_space=search_space)
        quality = (external_quality + internal_quality) / 2.0
        return normalized, "blended", quality
    if external:
        return external, "external", external_quality
    if internal:
        return internal, "internal", internal_quality
    return {}, "none", 0.0


__all__ = [
    "compute_internal_importances",
    "normalize_importances",
    "resolve_importances",
]

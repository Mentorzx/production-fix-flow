"""Pure helpers for HPO bounds and scoring utilities."""

from __future__ import annotations

import math
from typing import Any
from collections.abc import Iterable


def normalize_metric(
    value: float | None, *, low: float, high: float, cap: bool = True
) -> float:
    """Normalize a metric into [0, 1] with optional capping."""
    if value is None:
        return 0.0
    if isinstance(value, float) and math.isnan(value):
        return 0.0
    if high < low:
        low, high = high, low
    if high == low:
        if value < low:
            return 0.0
        if value > high:
            return 1.0
        return 0.5
    span = high - low
    normalized = (value - low) / span
    normalized = max(normalized, 0.0)
    if cap:
        normalized = min(1.0, normalized)
    return float(normalized)


def blend_scores(scores: Iterable[tuple[float, float]]) -> float:
    """Blend (value, weight) pairs into a weighted mean."""
    total_weight = 0.0
    total = 0.0
    for value, weight in scores:
        if weight <= 0.0:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        total_weight += float(weight)
        total += float(value) * float(weight)
    if total_weight <= 0.0:
        return 0.0
    return total / total_weight


def get_range(
    bounds: dict[str, Any],
    path: list[str],
    default_low: float,
    default_high: float,
) -> tuple[float, float]:
    """Fetch a numeric low/high bound from nested dicts with defaults."""
    current: Any = bounds
    for key in path:
        if not isinstance(current, dict):
            return float(default_low), float(default_high)
        current = current.get(key, {})
    if not isinstance(current, dict):
        return float(default_low), float(default_high)
    low = float(current.get("low", default_low))
    high = float(current.get("high", default_high))
    if low > high:
        return float(default_low), float(default_high)
    return float(low), float(high)

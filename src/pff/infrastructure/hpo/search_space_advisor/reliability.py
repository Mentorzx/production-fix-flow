"""Reliability scoring helpers for Search Space Advisor."""

from __future__ import annotations

import math
from typing import Any


def confidence_from_support(base: str, support: float | None) -> str:
    """Adjust confidence level using bootstrap support when available."""
    if support is None:
        return base
    if support >= 0.75:
        return "high"
    if support >= 0.5:
        return "medium"
    return "low"


def wilson_lower_bound(*, successes: int, total: int, z: float = 1.96) -> float:
    """Wilson score lower bound for Bernoulli confidence intervals."""
    if total <= 0:
        return 0.0
    p = float(successes) / float(total)
    z2 = z * z
    denom = 1.0 + z2 / float(total)
    centre = p + z2 / (2.0 * float(total))
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * float(total))) / float(total))
    return max(0.0, min(1.0, (centre - margin) / denom))


def compute_reliability_summary(recommendations: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate reliability metrics for recommendation payloads."""
    total = len(recommendations)
    if total <= 0:
        return {
            "total": 0,
            "actionable": 0,
            "blocked": 0,
            "validation_pass_rate": 1.0,
            "validation_pass_wilson_lb": 1.0,
            "mean_confidence_score": 0.0,
            "high_confidence_rate": 0.0,
            "high_confidence_wilson_lb": 0.0,
        }

    actionable = sum(1 for rec in recommendations if str(rec.get("action", "keep")) != "keep")
    blocked = sum(1 for rec in recommendations if rec.get("blocked_action") is not None)
    validations = [
        bool((rec.get("validation") or {}).get("passed", True))
        for rec in recommendations
        if isinstance(rec, dict)
    ]
    pass_rate = (
        float(sum(1 for passed in validations if passed)) / float(len(validations))
        if validations
        else 1.0
    )
    pass_successes = sum(1 for passed in validations if passed)
    pass_total = len(validations)
    pass_wilson_lb = (
        wilson_lower_bound(successes=pass_successes, total=pass_total) if pass_total > 0 else 1.0
    )

    confidence_scores = [
        float(score)
        for rec in recommendations
        for score in [rec.get("confidence_score")]
        if isinstance(score, (int, float))
    ]
    mean_confidence = (
        float(sum(confidence_scores)) / float(len(confidence_scores)) if confidence_scores else 0.0
    )

    high_confidence = sum(1 for rec in recommendations if str(rec.get("confidence")) == "high")
    high_confidence_rate = float(high_confidence) / float(total)
    high_confidence_wilson_lb = wilson_lower_bound(successes=high_confidence, total=total)

    return {
        "total": int(total),
        "actionable": int(actionable),
        "blocked": int(blocked),
        "validation_pass_rate": round(pass_rate, 4),
        "validation_pass_wilson_lb": round(pass_wilson_lb, 4),
        "mean_confidence_score": round(mean_confidence, 4),
        "high_confidence_rate": round(high_confidence_rate, 4),
        "high_confidence_wilson_lb": round(high_confidence_wilson_lb, 4),
    }


__all__ = [
    "compute_reliability_summary",
    "confidence_from_support",
    "wilson_lower_bound",
]

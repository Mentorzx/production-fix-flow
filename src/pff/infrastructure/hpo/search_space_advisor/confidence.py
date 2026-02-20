"""Confidence helpers for Search Space Advisor payloads."""

from __future__ import annotations

from typing import Any


def build_confidence_components(
    recommendation: dict[str, Any],
    *,
    prior_n: float = 20.0,
    prior_p: float = 0.5,
) -> dict[str, Any]:
    """Build confidence components with evidence-calibrated bootstrap support."""
    base_map = {"high": 0.85, "medium": 0.6, "low": 0.35}
    label = str(recommendation.get("confidence", "low"))
    attempts_summary = recommendation.get("attempts_summary") or {}
    evidence_count = attempts_summary.get("count")
    support = recommendation.get("bootstrap_support")
    calibrated_support = support
    if (
        isinstance(support, (int, float))
        and isinstance(evidence_count, int)
        and evidence_count >= 0
    ):
        calibrated_support = (
            (float(support) * float(evidence_count)) + (float(prior_p) * float(prior_n))
        ) / (float(evidence_count) + float(prior_n))
    return {
        "base_label": label,
        "base_score": base_map.get(label, 0.35),
        "bootstrap_support": support,
        "calibrated_support": calibrated_support,
        "evidence_count": evidence_count,
        "uncertainty": recommendation.get("uncertainty"),
    }


def compute_confidence_score(
    recommendation: dict[str, Any],
    *,
    prior_n: float = 20.0,
    prior_p: float = 0.5,
) -> float:
    """Compute confidence score in [0, 1] from recommendation components."""
    components = build_confidence_components(
        recommendation,
        prior_n=prior_n,
        prior_p=prior_p,
    )
    base = float(components["base_score"])
    support = components.get("calibrated_support")
    if isinstance(support, (int, float)):
        uncertainty = recommendation.get("uncertainty")
        uncertainty_penalty = 0.0
        if isinstance(uncertainty, (int, float)):
            uncertainty_penalty = 0.15 * float(uncertainty)
        return round(
            max(0.0, min(1.0, 0.45 * base + 0.55 * float(support) - uncertainty_penalty)),
            3,
        )
    uncertainty = recommendation.get("uncertainty")
    if isinstance(uncertainty, (int, float)):
        return round(max(0.0, min(1.0, base * (1.0 - 0.4 * float(uncertainty)))), 3)
    return round(base, 3)


__all__ = [
    "build_confidence_components",
    "compute_confidence_score",
]

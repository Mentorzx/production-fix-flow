"""Recommendation serialization and payload validation helpers."""

from __future__ import annotations

import math
from typing import Any

from .confidence import build_confidence_components, compute_confidence_score
from .models import ParamRecommendation


def validate_recommendation_payload(
    recommendation_payload: dict[str, Any],
    *,
    max_expansion_factor: float = 10.0,
) -> dict[str, Any]:
    """Validate advisor recommendation payload and return blocking metadata."""
    action = str(recommendation_payload.get("action", "keep"))
    recommendation = recommendation_payload.get("recommendation", {}) or {}
    checks: list[str] = []
    passed = True
    blocked_reason: str | None = None

    def _invalidate(reason: str) -> None:
        nonlocal passed, blocked_reason
        passed = False
        blocked_reason = reason

    if action == "expand_upper":
        old_high = recommendation.get("old_high")
        new_high = recommendation.get("new_high")
        if not isinstance(old_high, (int, float)) or not isinstance(new_high, (int, float)):
            _invalidate("expand_upper_missing_bounds")
        elif not math.isfinite(float(new_high)):
            _invalidate("expand_upper_non_finite")
        elif float(new_high) <= float(old_high):
            _invalidate("expand_upper_non_improving")
        elif float(old_high) > 0 and float(new_high) / float(old_high) > float(max_expansion_factor):
            _invalidate("expand_upper_excessive_factor")
        else:
            checks.append("expand_upper_valid")
    elif action == "expand_lower":
        old_low = recommendation.get("old_low")
        new_low = recommendation.get("new_low")
        if not isinstance(old_low, (int, float)) or not isinstance(new_low, (int, float)):
            _invalidate("expand_lower_missing_bounds")
        elif not math.isfinite(float(new_low)):
            _invalidate("expand_lower_non_finite")
        elif float(new_low) >= float(old_low):
            _invalidate("expand_lower_non_improving")
        elif float(new_low) < 0 <= float(old_low):
            _invalidate("expand_lower_negative_bound")
        elif (
            float(new_low) > 0
            and float(old_low) > 0
            and float(old_low) / float(new_low) > float(max_expansion_factor)
        ):
            _invalidate("expand_lower_excessive_factor")
        else:
            checks.append("expand_lower_valid")
    elif action == "narrow":
        new_low = recommendation.get("new_low")
        new_high = recommendation.get("new_high")
        if not isinstance(new_low, (int, float)) or not isinstance(new_high, (int, float)):
            _invalidate("narrow_missing_bounds")
        elif float(new_low) >= float(new_high):
            _invalidate("narrow_invalid_interval")
        else:
            checks.append("narrow_interval_valid")
    elif action == "reduce_categories":
        keep = recommendation.get("keep", [])
        remove = recommendation.get("remove", [])
        keep_tokens = {str(v) for v in keep}
        remove_tokens = {str(v) for v in remove}
        if not keep_tokens:
            _invalidate("reduce_categories_empty_keep")
        elif keep_tokens & remove_tokens:
            _invalidate("reduce_categories_overlap")
        else:
            checks.append("reduce_categories_partition_valid")
    elif action == "fix":
        if recommendation.get("fix_value") is None:
            _invalidate("fix_missing_value")
        else:
            checks.append("fix_value_present")
    elif action == "change_distribution":
        distribution = str(recommendation.get("distribution", ""))
        low = recommendation.get("low")
        high = recommendation.get("high")
        if distribution == "log_uniform":
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                _invalidate("change_distribution_missing_bounds")
            elif float(low) <= 0 or float(high) <= 0:
                _invalidate("change_distribution_non_positive_log_bounds")
            elif float(low) >= float(high):
                _invalidate("change_distribution_invalid_interval")
            else:
                checks.append("change_distribution_log_uniform_valid")
        else:
            checks.append("change_distribution_unchecked_distribution")
    else:
        checks.append("no_validation_required")

    return {"passed": passed, "checks": checks, "blocked_reason": blocked_reason}


def recommendation_to_dict(
    recommendation: ParamRecommendation,
    *,
    confidence_prior_n: float = 20.0,
    confidence_prior_p: float = 0.5,
    max_expansion_factor: float = 10.0,
) -> dict[str, Any]:
    """Serialize recommendation dataclass into API payload."""
    payload = {
        "param_name": recommendation.param_name,
        "current_space": recommendation.current_space,
        "attempts_summary": recommendation.attempts_summary,
        "best_region": recommendation.best_region,
        "importance": recommendation.importance,
        "action": recommendation.action,
        "recommendation": recommendation.recommendation,
        "rationale": recommendation.rationale,
        "scope": "global",
        "confidence": recommendation.confidence,
        "uncertainty": recommendation.uncertainty,
        "bootstrap_support": recommendation.bootstrap_support,
        "interaction_strength": recommendation.interaction_strength,
        "surrogate_bounds": recommendation.surrogate_bounds,
    }
    payload["validation"] = validate_recommendation_payload(
        payload,
        max_expansion_factor=max_expansion_factor,
    )
    payload["confidence_score"] = compute_confidence_score(
        payload,
        prior_n=confidence_prior_n,
        prior_p=confidence_prior_p,
    )
    payload["confidence_components"] = build_confidence_components(
        payload,
        prior_n=confidence_prior_n,
        prior_p=confidence_prior_p,
    )
    return payload


__all__ = [
    "recommendation_to_dict",
    "validate_recommendation_payload",
]

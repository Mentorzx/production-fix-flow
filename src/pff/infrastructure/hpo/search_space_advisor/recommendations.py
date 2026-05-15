"""Recommendation assembly and post-processing helpers."""

from __future__ import annotations

import math
from typing import Any, Callable

from .models import ParamRecommendation


def resolve_recommendation_scope(
    *, evidence_ratio: float, conditional_threshold: float = 0.9
) -> str:
    """Classify recommendation scope as global or conditional."""
    return "conditional" if evidence_ratio < conditional_threshold else "global"


def compute_search_space_coverage(
    *,
    search_space: dict[str, Any],
    observed_param_counts: dict[str, int],
    total_trials: int,
) -> tuple[float, list[str]]:
    """Compute parameter coverage ratio and missing-parameter list."""
    if not search_space:
        return 1.0, []
    covered = 0
    missing: list[str] = []
    for param_name in search_space:
        count = int(observed_param_counts.get(param_name, 0))
        if count > 0 and total_trials > 0:
            covered += 1
        else:
            missing.append(param_name)
    ratio = float(covered) / float(len(search_space))
    return ratio, missing


def make_recommendation(
    *,
    param_name: str,
    parsed: dict[str, Any],
    all_values: list[Any],
    top_k_values: list[Any],
    importance: float,
    action: str,
    recommendation: dict[str, Any],
    rationale: str,
    confidence: str,
    estimate_uncertainty_fn: Callable[[int, int], float],
    numeric_stats_fn: Callable[[list[float]], dict[str, float]],
    reservoir_sample_fn: Callable[[list[float], int, int], list[float]],
    categorical_counts_fn: Callable[[list[Any]], dict[str, int]],
    reservoir_size: int,
    bootstrap_support: float | None = None,
    interaction_strength: float | None = None,
    surrogate_bounds: dict[str, float] | None = None,
) -> ParamRecommendation:
    """Build typed recommendation payload from raw analysis artifacts."""
    n_trials = len(all_values)
    top_k_count = len(top_k_values)
    uncertainty = estimate_uncertainty_fn(n_trials, top_k_count)
    if parsed.get("type") in ("float", "int"):
        numeric_all = [float(v) for v in all_values if v is not None]
        numeric_top = [float(v) for v in top_k_values if v is not None]
        attempts_summary = {
            "count": len(numeric_all),
            "stats": numeric_stats_fn(numeric_all),
            "samples": reservoir_sample_fn(numeric_all, reservoir_size, 42),
        }
        best_region = {
            "top_k_count": len(numeric_top),
            "stats": numeric_stats_fn(numeric_top),
        }
    else:
        attempts_summary = {
            "count": len(all_values),
            "distribution": categorical_counts_fn(all_values),
        }
        best_region = {
            "top_k_count": len(top_k_values),
            "distribution": categorical_counts_fn(top_k_values),
        }

    return ParamRecommendation(
        param_name=param_name,
        current_space=parsed,
        attempts_summary=attempts_summary,
        best_region=best_region,
        importance=importance,
        action=action,
        recommendation=recommendation,
        rationale=rationale,
        confidence=confidence,
        uncertainty=uncertainty,
        bootstrap_support=bootstrap_support,
        interaction_strength=interaction_strength,
        surrogate_bounds=surrogate_bounds,
    )


def build_fixed_parameter_recommendation(
    *,
    param_name: str,
    parsed: dict[str, Any],
    all_values: list[Any],
    top_k_values: list[Any],
    importance: float,
    n_trials: int,
    min_trials: int,
    high_importance_threshold: float,
    low_importance_threshold: float,
) -> ParamRecommendation:
    """Build a non-mutating diagnostic for parameters outside the search space."""
    fixed_value = parsed.get("value")
    observed_count = len(all_values)
    top_k_count = len(top_k_values)
    name_lower = param_name.lower()
    is_capacity_param = any(token in name_lower for token in ("embedding", "hidden", "dim"))
    is_regularization_param = any(
        token in name_lower for token in ("lambda", "dropout", "weight_decay")
    )
    enough_evidence = observed_count >= min_trials and top_k_count >= 3

    if importance >= high_importance_threshold:
        diagnostic = "needs_exploration"
        severity = "high" if importance >= high_importance_threshold * 2.0 else "medium"
        suggested_action = "convert_fixed_to_bounded_search"
        confidence = "medium" if enough_evidence else "low"
        rationale = (
            f"{param_name} is fixed at {fixed_value!r}, but its resolved importance "
            f"is {importance:.3f}. The Advisor cannot estimate sensitivity from a "
            "single value; run a local sweep or bounded distribution before treating "
            "this value as optimal."
        )
    elif enough_evidence and importance <= low_importance_threshold:
        diagnostic = "stable_fixed_value"
        severity = "low"
        suggested_action = "keep_fixed"
        confidence = "medium"
        rationale = (
            f"{param_name} is fixed at {fixed_value!r} and has low resolved importance "
            f"({importance:.3f}) across observed trials. Keep it fixed unless a new "
            "dataset, budget, or architecture change makes it relevant again."
        )
    else:
        diagnostic = "watch_fixed_value"
        severity = "medium" if is_capacity_param or is_regularization_param else "low"
        suggested_action = "collect_more_evidence"
        confidence = "low"
        rationale = (
            f"{param_name} is fixed at {fixed_value!r}. Current evidence is not strong "
            "enough to classify it as safe or critical; keep the value for now and "
            "revisit after more completed trials."
        )

    return ParamRecommendation(
        param_name=param_name,
        current_space=parsed,
        attempts_summary={
            "count": observed_count,
            "fixed_value": fixed_value,
            "unique_values": sorted({str(value) for value in all_values}),
        },
        best_region={
            "top_k_count": top_k_count,
            "fixed_value": fixed_value,
            "unique_values": sorted({str(value) for value in top_k_values}),
        },
        importance=importance,
        action="keep",
        recommendation={
            "fixed_value": fixed_value,
            "diagnostic": diagnostic,
            "severity": severity,
            "suggested_action": suggested_action,
            "observed_trials": observed_count,
            "top_k_trials": top_k_count,
        },
        rationale=rationale,
        confidence=confidence,
        uncertainty=0.5 if enough_evidence else 0.8,
    )


def build_dataset_heuristic_recommendations(
    *,
    search_space: dict[str, Any],
    dataset_profile: dict[str, Any] | None,
    parse_search_space_entry_fn: Callable[[str, Any], dict[str, Any]],
    make_recommendation_fn: Callable[..., ParamRecommendation],
    recommendation_to_dict_fn: Callable[[ParamRecommendation], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build low-evidence heuristic recommendations from dataset scale/profile."""
    if not dataset_profile:
        return []

    n_entities = int(dataset_profile.get("n_entities", 0) or 0)
    n_relations = int(dataset_profile.get("n_relations", 0) or 0)
    n_triples = int(dataset_profile.get("n_triples", 0) or 0)
    density = float(dataset_profile.get("density", 0.0) or 0.0)
    if n_triples <= 0:
        return []

    recommendations: list[dict[str, Any]] = []
    target_embedding = int(
        min(
            1024,
            max(
                64,
                2 ** round(math.log2(max(64, int(math.sqrt(max(n_entities, 1)) * 2)))),
            ),
        )
    )
    target_neg_sampling = (
        64 if n_triples < 100_000 else (128 if n_triples < 1_000_000 else 256)
    )
    lambda_low, lambda_high = (0.0, 0.12) if density < 0.001 else (0.02, 0.4)

    for param_name, spec in search_space.items():
        parsed = parse_search_space_entry_fn(param_name, spec)
        param_lower = param_name.lower()
        rec: dict[str, Any] | None = None

        if "embedding" in param_lower and "dim" in param_lower:
            if parsed.get("type") == "categorical":
                choices = [
                    int(v)
                    for v in parsed.get("choices", [])
                    if isinstance(v, (int, float))
                ]
                if choices:
                    ordered = sorted(
                        set(choices), key=lambda v: abs(v - target_embedding)
                    )
                    keep = sorted(ordered[: min(3, len(ordered))])
                    rec = recommendation_to_dict_fn(
                        make_recommendation_fn(
                            param_name=param_name,
                            parsed=parsed,
                            all_values=[],
                            top_k_values=[],
                            importance=0.0,
                            action="reduce_categories",
                            recommendation={
                                "keep": keep,
                                "remove": sorted(
                                    [str(c) for c in choices if c not in set(keep)]
                                ),
                            },
                            rationale=(
                                "Dataset heuristics: low trial evidence. "
                                f"Suggested embedding_dim around {target_embedding} "
                                f"(n_entities={n_entities}, n_relations={n_relations})."
                            ),
                            confidence="low",
                        )
                    )
            elif (
                parsed.get("type") in ("int", "float")
                and "low" in parsed
                and "high" in parsed
            ):
                current_low = int(parsed["low"])
                current_high = int(parsed["high"])
                suggested_low = max(current_low, target_embedding // 2)
                suggested_high = min(current_high, target_embedding * 2)
                if suggested_low <= suggested_high:
                    rec = recommendation_to_dict_fn(
                        make_recommendation_fn(
                            param_name=param_name,
                            parsed=parsed,
                            all_values=[],
                            top_k_values=[],
                            importance=0.0,
                            action="narrow",
                            recommendation={
                                "new_low": int(suggested_low),
                                "new_high": int(suggested_high),
                                "old_low": current_low,
                                "old_high": current_high,
                            },
                            rationale=(
                                "Dataset heuristics: low trial evidence. "
                                f"Constraining embedding_dim around {target_embedding} "
                                f"(n_entities={n_entities}, n_relations={n_relations})."
                            ),
                            confidence="low",
                        )
                    )

        elif "neg" in param_lower and (
            "sample" in param_lower or "sampling" in param_lower
        ):
            if (
                parsed.get("type") in ("int", "float")
                and "low" in parsed
                and "high" in parsed
            ):
                current_low = int(parsed["low"])
                current_high = int(parsed["high"])
                suggested_low = max(current_low, target_neg_sampling // 2)
                suggested_high = min(current_high, target_neg_sampling * 2)
                if suggested_low <= suggested_high:
                    rec = recommendation_to_dict_fn(
                        make_recommendation_fn(
                            param_name=param_name,
                            parsed=parsed,
                            all_values=[],
                            top_k_values=[],
                            importance=0.0,
                            action="narrow",
                            recommendation={
                                "new_low": int(suggested_low),
                                "new_high": int(suggested_high),
                                "old_low": current_low,
                                "old_high": current_high,
                            },
                            rationale=(
                                "Dataset heuristics: low trial evidence. "
                                f"Aligning negative sampling with dataset scale (n_triples={n_triples})."
                            ),
                            confidence="low",
                        )
                    )

        elif any(
            token in param_lower for token in ("lambda", "weight_decay", "dropout")
        ):
            if (
                parsed.get("type") in ("int", "float")
                and "low" in parsed
                and "high" in parsed
            ):
                current_low = float(parsed["low"])
                current_high = float(parsed["high"])
                suggested_low = max(current_low, lambda_low)
                suggested_high = min(current_high, lambda_high)
                if suggested_low <= suggested_high:
                    rec = recommendation_to_dict_fn(
                        make_recommendation_fn(
                            param_name=param_name,
                            parsed=parsed,
                            all_values=[],
                            top_k_values=[],
                            importance=0.0,
                            action="narrow",
                            recommendation={
                                "new_low": round(float(suggested_low), 6),
                                "new_high": round(float(suggested_high), 6),
                                "old_low": current_low,
                                "old_high": current_high,
                            },
                            rationale=(
                                "Dataset heuristics: low trial evidence. "
                                f"Regularization guided by graph density={density:.4g}."
                            ),
                            confidence="low",
                        )
                    )

        if rec is not None:
            recommendations.append(rec)

    return recommendations


__all__ = [
    "build_fixed_parameter_recommendation",
    "build_dataset_heuristic_recommendations",
    "compute_search_space_coverage",
    "make_recommendation",
    "resolve_recommendation_scope",
]

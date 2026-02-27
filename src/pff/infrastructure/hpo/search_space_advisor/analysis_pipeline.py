"""Parameter-level analysis pipeline for Search Space Advisor."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from .analysis_categorical import categorical_counts, decide_categorical_action
from .analysis_numeric import (
    ballet_safe_shrink,
    decide_numeric_action,
    surrogate_grid_bounds,
)
from .bootstrap import bootstrap_action_support
from .models import (
    ParamMeta,
    ParamRecommendation,
    SurrogateModel,
    TrialSummary,
    TrustState,
)
from .parsing import is_cost_sensitive_param, normalize_direction
from .recommendations import make_recommendation
from .statistics import (
    estimate_uncertainty,
    numeric_stats,
    reservoir_sample,
    spearman_rho,
)
from .surrogate import (
    denormalize_log_value,
    encode_params,
    normalize_log_value,
    predict_surrogate,
)


def apply_direction(value: float, direction: str) -> float:
    """Apply objective direction to score."""
    return value if normalize_direction(direction) == "maximize" else -value


def compute_spearman(
    values: list[float],
    scores: list[float],
    *,
    min_points: int,
    rust_fast_fn: Any,
    rust_min_len: int,
    np_module: Any,
) -> float | None:
    """Compute Spearman correlation using rust path when available."""
    return spearman_rho(
        values,
        scores,
        min_points=min_points,
        rust_fast_fn=rust_fast_fn,
        rust_min_len=rust_min_len,
        np_module=np_module,
    )


def analyze_numeric_param(
    *,
    param_name: str,
    parsed: dict[str, Any],
    all_values: list[float],
    all_scores: list[float],
    top_k_values: list[float],
    importance: float,
    n_trials: int,
    param_meta: ParamMeta,
    param_meta_map: dict[str, ParamMeta],
    trust_state: TrustState | None,
    surrogate: SurrogateModel | None,
    anchor_params: dict[str, Any],
    interaction_strength: float,
    interaction_threshold: float,
    confidence: str,
    bootstrap_support: float | None,
    edge_threshold: float,
    concentration_cv_threshold: float,
    low_importance_threshold: float,
    min_trials_aggressive: int,
    trust_success: int,
    correlation_gate_abs: float,
    cost_sensitive_upper_rho: float,
    ucb_std_mult: float,
    reservoir_size: int,
    rust_fast_spearman_fn: Any,
    rust_spearman_min_len: int,
    np_module: Any,
) -> ParamRecommendation:
    """Analyze one numeric parameter and produce an actionable recommendation."""

    def _surrogate_grid_bounds(
        surrogate_model: SurrogateModel,
        meta_map: dict[str, ParamMeta],
        anchor: dict[str, Any],
        *,
        param_name: str,
        low: float,
        high: float,
    ) -> tuple[list[float], list[float], list[float]]:
        return surrogate_grid_bounds(
            surrogate=surrogate_model,
            param_meta=meta_map,
            anchor_params=anchor,
            param_name=param_name,
            low=low,
            high=high,
            denormalize_log_value=lambda value: denormalize_log_value(
                value,
                is_log=bool(getattr(meta_map.get(param_name), "is_log", False)),
            ),
            encode_params=lambda params, inner_meta: encode_params(
                params,
                inner_meta,
                normalize_log_value_fn=lambda raw, is_log: normalize_log_value(
                    raw,
                    is_log=is_log,
                ),
            ),
            predict_surrogate=predict_surrogate,
            ucb_std_mult=ucb_std_mult,
        )

    action, rec, rationale_parts, surrogate_bounds = decide_numeric_action(
        param_name=param_name,
        parsed=parsed,
        all_values=all_values,
        all_scores=all_scores,
        top_k_values=top_k_values,
        importance=importance,
        n_trials=n_trials,
        param_meta=param_meta,
        param_meta_map=param_meta_map,
        trust_state=trust_state,
        surrogate=surrogate,
        anchor_params=anchor_params,
        normalize_log_value=lambda value, is_log: normalize_log_value(
            value, is_log=is_log
        ),
        denormalize_log_value=lambda value, is_log: denormalize_log_value(
            value, is_log=is_log
        ),
        numeric_stats=numeric_stats,
        spearman_rho=lambda values, scores: compute_spearman(
            values,
            scores,
            min_points=8,
            rust_fast_fn=rust_fast_spearman_fn,
            rust_min_len=rust_spearman_min_len,
            np_module=np_module,
        ),
        is_cost_sensitive_param=is_cost_sensitive_param,
        surrogate_grid_bounds_fn=_surrogate_grid_bounds,
        ballet_safe_shrink_fn=ballet_safe_shrink,
        edge_threshold=edge_threshold,
        concentration_cv_threshold=concentration_cv_threshold,
        low_importance_threshold=low_importance_threshold,
        min_trials_aggressive=min_trials_aggressive,
        trust_success=trust_success,
        correlation_gate_abs=correlation_gate_abs,
        cost_sensitive_upper_rho=cost_sensitive_upper_rho,
    )

    if interaction_strength > interaction_threshold and action in {"fix", "narrow"}:
        action = "keep"
        rec = {"delta": "none"}
        rationale_parts.append(
            "Strong interactions detected; avoiding fixing or narrowing this parameter."
        )

    return make_recommendation(
        param_name=param_name,
        parsed=parsed,
        all_values=all_values,
        top_k_values=top_k_values,
        importance=importance,
        action=action,
        recommendation=rec,
        rationale=" ".join(rationale_parts),
        confidence=confidence,
        estimate_uncertainty_fn=lambda n_param_trials, top_k_count: estimate_uncertainty(
            n_trials=n_param_trials,
            top_k_count=top_k_count,
        ),
        numeric_stats_fn=numeric_stats,
        reservoir_sample_fn=lambda values, k, seed: reservoir_sample(
            values, k=k, seed=seed
        ),
        categorical_counts_fn=categorical_counts,
        reservoir_size=reservoir_size,
        bootstrap_support=bootstrap_support,
        interaction_strength=interaction_strength,
        surrogate_bounds=surrogate_bounds,
    )


def analyze_categorical_param(
    *,
    param_name: str,
    parsed: dict[str, Any],
    all_values: list[Any],
    top_k_values: list[Any],
    importance: float,
    n_trials: int,
    param_meta: ParamMeta,
    param_meta_map: dict[str, ParamMeta],
    surrogate: SurrogateModel | None,
    anchor_params: dict[str, Any],
    interaction_strength: float,
    interaction_threshold: float,
    confidence: str,
    bootstrap_support: float | None,
    categorical_dominance_threshold: float,
    low_importance_threshold: float,
    ucb_std_mult: float,
    reservoir_size: int,
    categorical_min_topk_samples: int,
    categorical_min_topk_unique: int,
    categorical_min_effective_categories: float,
) -> ParamRecommendation:
    """Analyze one categorical parameter and produce an actionable recommendation."""
    _ = n_trials
    _ = param_meta
    action, rec, rationale_parts = decide_categorical_action(
        parsed=parsed,
        top_k_values=top_k_values,
        importance=importance,
        surrogate=surrogate,
        anchor_params=anchor_params,
        param_name=param_name,
        param_meta_map=param_meta_map,
        encode_params_fn=lambda params, meta: encode_params(
            params,
            meta,
            normalize_log_value_fn=lambda raw, is_log: normalize_log_value(
                raw, is_log=is_log
            ),
        ),
        predict_surrogate_fn=predict_surrogate,
        interaction_strength=interaction_strength,
        interaction_threshold=interaction_threshold,
        categorical_dominance_threshold=categorical_dominance_threshold,
        low_importance_threshold=low_importance_threshold,
        ucb_std_mult=ucb_std_mult,
        min_topk_samples=categorical_min_topk_samples,
        min_topk_unique=categorical_min_topk_unique,
        min_effective_categories=categorical_min_effective_categories,
    )

    return make_recommendation(
        param_name=param_name,
        parsed=parsed,
        all_values=all_values,
        top_k_values=top_k_values,
        importance=importance,
        action=action,
        recommendation=rec,
        rationale=" ".join(rationale_parts),
        confidence=confidence,
        estimate_uncertainty_fn=lambda n_param_trials, top_k_count: estimate_uncertainty(
            n_trials=n_param_trials,
            top_k_count=top_k_count,
        ),
        numeric_stats_fn=numeric_stats,
        reservoir_sample_fn=lambda values, k, seed: reservoir_sample(
            values, k=k, seed=seed
        ),
        categorical_counts_fn=categorical_counts,
        reservoir_size=reservoir_size,
        bootstrap_support=bootstrap_support,
        interaction_strength=interaction_strength,
    )


def bootstrap_support_for_param(
    *,
    trials: list[TrialSummary],
    direction: str,
    top_k_fraction: float,
    top_k_min: int,
    param_name: str,
    parsed: dict[str, Any],
    importance: float,
    param_meta: ParamMeta,
    param_meta_map: dict[str, ParamMeta],
    trust_state: TrustState | None,
    surrogate: SurrogateModel | None,
    anchor_params: dict[str, Any],
    interaction_strength: float,
    interaction_threshold: float,
    seed: int,
    final_action: str,
    min_trials_aggressive: int,
    bootstrap_samples: int,
    edge_threshold: float,
    concentration_cv_threshold: float,
    low_importance_threshold: float,
    trust_success: int,
    correlation_gate_abs: float,
    cost_sensitive_upper_rho: float,
    categorical_dominance_threshold: float,
    categorical_min_topk_samples: int,
    categorical_min_topk_unique: int,
    categorical_min_effective_categories: float,
    ucb_std_mult: float,
    reservoir_size: int,
    rust_fast_spearman_fn: Any,
    rust_spearman_min_len: int,
    np_module: Any,
    select_top_k_fn: Callable[..., list[TrialSummary]],
) -> float | None:
    """Estimate recommendation action support via bootstrap re-sampling."""

    def _evaluate_sample(sample: list[TrialSummary]) -> bool | None:
        top_k = select_top_k_fn(
            sample,
            direction=direction,
            fraction=top_k_fraction,
            min_k=top_k_min,
        )
        all_pairs = [
            (value, apply_direction(float(trial.value), direction))
            for trial in sample
            if param_name in trial.params
            for value in [trial.params.get(param_name)]
            if value is not None
        ]
        top_values = [t.params.get(param_name) for t in top_k if param_name in t.params]
        all_values = [value for value, _ in all_pairs]
        all_scores = [score for _, score in all_pairs]
        top_values = [v for v in top_values if v is not None]
        if not all_values:
            return None
        if parsed.get("type") in ("float", "int"):
            rec = analyze_numeric_param(
                param_name=param_name,
                parsed=parsed,
                all_values=[float(v) for v in all_values],
                all_scores=all_scores,
                top_k_values=[float(v) for v in top_values],
                importance=importance,
                n_trials=len(sample),
                param_meta=param_meta,
                param_meta_map=param_meta_map,
                trust_state=trust_state,
                surrogate=surrogate,
                anchor_params=anchor_params,
                interaction_strength=interaction_strength,
                interaction_threshold=interaction_threshold,
                confidence="low",
                bootstrap_support=None,
                edge_threshold=edge_threshold,
                concentration_cv_threshold=concentration_cv_threshold,
                low_importance_threshold=low_importance_threshold,
                min_trials_aggressive=min_trials_aggressive,
                trust_success=trust_success,
                correlation_gate_abs=correlation_gate_abs,
                cost_sensitive_upper_rho=cost_sensitive_upper_rho,
                ucb_std_mult=ucb_std_mult,
                reservoir_size=reservoir_size,
                rust_fast_spearman_fn=rust_fast_spearman_fn,
                rust_spearman_min_len=rust_spearman_min_len,
                np_module=np_module,
            )
        else:
            rec = analyze_categorical_param(
                param_name=param_name,
                parsed=parsed,
                all_values=all_values,
                top_k_values=top_values,
                importance=importance,
                n_trials=len(sample),
                param_meta=param_meta,
                param_meta_map=param_meta_map,
                surrogate=surrogate,
                anchor_params=anchor_params,
                interaction_strength=interaction_strength,
                interaction_threshold=interaction_threshold,
                confidence="low",
                bootstrap_support=None,
                categorical_dominance_threshold=categorical_dominance_threshold,
                categorical_min_topk_samples=categorical_min_topk_samples,
                categorical_min_topk_unique=categorical_min_topk_unique,
                categorical_min_effective_categories=categorical_min_effective_categories,
                low_importance_threshold=low_importance_threshold,
                ucb_std_mult=ucb_std_mult,
                reservoir_size=reservoir_size,
            )
        return rec.action == final_action

    return bootstrap_action_support(
        trials=trials,
        min_trials=min_trials_aggressive,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        evaluate_sample=_evaluate_sample,
    )


def with_bootstrap_confidence(
    recommendation: ParamRecommendation,
    *,
    base_confidence: str,
    bootstrap_support: float | None,
    interaction_strength: float,
    confidence_from_support_fn: Callable[[str, float | None], str],
) -> ParamRecommendation:
    """Attach bootstrap confidence metadata to recommendation."""
    return replace(
        recommendation,
        bootstrap_support=bootstrap_support,
        confidence=confidence_from_support_fn(base_confidence, bootstrap_support),
        interaction_strength=interaction_strength,
    )


__all__ = [
    "analyze_categorical_param",
    "analyze_numeric_param",
    "apply_direction",
    "bootstrap_support_for_param",
    "compute_spearman",
    "with_bootstrap_confidence",
]

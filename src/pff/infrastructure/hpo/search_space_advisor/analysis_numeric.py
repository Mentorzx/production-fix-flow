"""Numeric-analysis helpers for Search Space Advisor."""

from __future__ import annotations

from typing import Any, Callable


def surrogate_grid_bounds(
    *,
    surrogate: Any,
    param_meta: dict[str, Any],
    anchor_params: dict[str, Any],
    param_name: str,
    low: float,
    high: float,
    denormalize_log_value: Callable[[float], float],
    encode_params: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    predict_surrogate: Callable[
        [Any, list[dict[str, Any]]], tuple[list[float], list[float]]
    ],
    ucb_std_mult: float = 1.96,
) -> tuple[list[float], list[float], list[float]]:
    """Evaluate surrogate over a fixed grid and return (grid, ucb, lcb)."""
    grid = [low + (high - low) * i / 24 for i in range(25)]
    rows: list[dict[str, Any]] = []
    for val in grid:
        params = dict(anchor_params)
        meta = param_meta[param_name]
        raw_val = (
            denormalize_log_value(val) if bool(getattr(meta, "is_log", False)) else val
        )
        params[param_name] = raw_val
        rows.append(encode_params(params, param_meta))
    means, stds = predict_surrogate(surrogate, rows)
    ucb = [
        mean + float(ucb_std_mult) * std for mean, std in zip(means, stds, strict=False)
    ]
    lcb = [
        mean - float(ucb_std_mult) * std for mean, std in zip(means, stds, strict=False)
    ]
    return grid, ucb, lcb


def ballet_safe_shrink(
    *,
    grid: list[float],
    ucb: list[float],
    lcb: list[float],
    new_low: float,
    new_high: float,
) -> tuple[bool, dict[str, float]]:
    """BALLET-style safety check for narrowing a numeric interval."""
    inside = [
        idx for idx, val in enumerate(grid) if float(new_low) <= val <= float(new_high)
    ]
    outside = [idx for idx in range(len(grid)) if idx not in inside]
    if not inside or not outside:
        return True, {
            "lcb_inside": max(lcb) if lcb else 0.0,
            "ucb_outside": max(ucb) if ucb else 0.0,
        }
    max_inside_lcb = max(lcb[idx] for idx in inside)
    max_outside_ucb = max(ucb[idx] for idx in outside)
    return max_outside_ucb < max_inside_lcb, {
        "lcb_inside": max_inside_lcb,
        "ucb_outside": max_outside_ucb,
    }


def decide_numeric_action(
    *,
    param_name: str,
    parsed: dict[str, Any],
    all_values: list[float],
    all_scores: list[float],
    top_k_values: list[float],
    importance: float,
    n_trials: int,
    param_meta: Any,
    param_meta_map: dict[str, Any],
    trust_state: Any | None,
    surrogate: Any | None,
    anchor_params: dict[str, Any],
    normalize_log_value: Callable[[float, bool], float],
    denormalize_log_value: Callable[[float, bool], float],
    numeric_stats: Callable[[list[float]], dict[str, float]],
    spearman_rho: Callable[[list[float], list[float]], float | None],
    is_cost_sensitive_param: Callable[[str], bool],
    surrogate_grid_bounds_fn: Callable[
        ..., tuple[list[float], list[float], list[float]]
    ],
    ballet_safe_shrink_fn: Callable[..., tuple[bool, dict[str, float]]],
    edge_threshold: float,
    concentration_cv_threshold: float,
    low_importance_threshold: float,
    min_trials_aggressive: int,
    trust_success: int,
    correlation_gate_abs: float,
    cost_sensitive_upper_rho: float,
) -> tuple[str, dict[str, Any], list[str], dict[str, float] | None]:
    """Decide numeric recommendation action with rationale and optional surrogate bounds."""
    low = float(parsed["low"])
    high = float(parsed["high"])
    span = high - low
    if span <= 0:
        return (
            "keep",
            {"delta": "none"},
            ["Degenerate range (low == high); parameter is effectively fixed."],
            None,
        )

    use_log_scale = bool(param_meta.is_log and low > 0 and high > 0)
    low_t = normalize_log_value(low, use_log_scale)
    high_t = normalize_log_value(high, use_log_scale)
    span_t = high_t - low_t

    all_values_t = [
        normalize_log_value(float(value), use_log_scale) for value in all_values
    ]
    all_stats = numeric_stats(all_values_t)
    top_values_t = [
        normalize_log_value(float(value), use_log_scale) for value in top_k_values
    ]
    top_stats = numeric_stats(top_values_t)

    if not top_stats or not all_stats:
        return ("keep", {"delta": "none"}, ["No top-k data available."], None)

    action = "keep"
    recommendation: dict[str, Any] = {"delta": "none"}
    rationale_parts: list[str] = []
    surrogate_bounds: dict[str, float] | None = None

    upper_proximity = (top_stats["q90"] - low_t) / span_t
    lower_proximity = (top_stats["q10"] - low_t) / span_t
    top_cv = top_stats["std"] / max(abs(top_stats["mean"]), 1e-12)

    trust_upper = trust_state is not None and trust_state.upper_success >= int(
        trust_success
    )
    trust_lower = trust_state is not None and trust_state.lower_success >= int(
        trust_success
    )

    upper_alignment = (
        top_stats["q90"] >= all_stats["q90"] - 1e-12
        and top_stats["mean"] >= all_stats["mean"] - 1e-12
    )
    lower_alignment = (
        top_stats["q10"] <= all_stats["q10"] + 1e-12
        and top_stats["mean"] <= all_stats["mean"] + 1e-12
    )
    monotonic = spearman_rho(all_values_t, all_scores)
    unique_count = len(set(all_values_t))
    weak_monotonic_evidence = (
        monotonic is None and unique_count < 4 and not trust_upper and not trust_lower
    )
    upper_cost_sensitive = is_cost_sensitive_param(param_name)
    upper_edge_signal = trust_upper or (
        upper_proximity > (1 - float(edge_threshold)) and upper_alignment
    )
    lower_edge_signal = trust_lower or (
        lower_proximity < float(edge_threshold) and lower_alignment
    )
    upper_correlation_ok = (monotonic is None and not weak_monotonic_evidence) or (
        monotonic is not None and monotonic >= float(correlation_gate_abs)
    )
    if upper_cost_sensitive:
        upper_correlation_ok = upper_correlation_ok and (
            monotonic is not None
            and monotonic >= float(cost_sensitive_upper_rho)
            and importance >= 0.1
        )
    lower_correlation_ok = (monotonic is None and not weak_monotonic_evidence) or (
        monotonic is not None and monotonic <= -float(correlation_gate_abs)
    )

    if upper_edge_signal and upper_correlation_ok:
        new_high_t = high_t + span_t * 0.5
        new_high = denormalize_log_value(new_high_t, use_log_scale)
        action = "expand_upper"
        recommendation = {"new_high": round(new_high, 6), "old_high": high}
        rationale_parts.append(
            f"Top trials concentrate near upper bound (q90={denormalize_log_value(top_stats['q90'], use_log_scale):.4g}, "
            f"upper={high}). Expanding upper bound."
        )
    elif lower_edge_signal and lower_correlation_ok:
        new_low_t = low_t - span_t * 0.5
        new_low = denormalize_log_value(new_low_t, use_log_scale)
        if low >= 0:
            new_low = max(0.0, new_low)
        action = "expand_lower"
        recommendation = {"new_low": round(new_low, 6), "old_low": low}
        rationale_parts.append(
            f"Top trials concentrate near lower bound (q10={denormalize_log_value(top_stats['q10'], use_log_scale):.4g}, "
            f"lower={low}). Expanding lower bound."
        )
    elif top_cv < float(concentration_cv_threshold) and n_trials >= int(
        min_trials_aggressive
    ):
        new_low_t = top_stats["q10"]
        new_high_t = top_stats["q90"]
        if new_high_t - new_low_t < span_t * 0.1:
            margin = span_t * 0.05
            new_low_t = max(low_t, top_stats["q50"] - margin)
            new_high_t = min(high_t, top_stats["q50"] + margin)
        if surrogate is not None:
            grid, ucb, lcb = surrogate_grid_bounds_fn(
                surrogate,
                param_meta_map,
                anchor_params,
                param_name=param_meta.name,
                low=low_t,
                high=high_t,
            )
            ballet_ok, bounds = ballet_safe_shrink_fn(
                grid,
                ucb,
                lcb,
                new_low=new_low_t,
                new_high=new_high_t,
            )
            surrogate_bounds = bounds
            if not ballet_ok:
                rationale_parts.append(
                    "Surrogate uncertainty suggests keeping the wider region (BALLET safety failed)."
                )
            else:
                new_low_val = denormalize_log_value(new_low_t, use_log_scale)
                new_high_val = denormalize_log_value(new_high_t, use_log_scale)
                action = "narrow"
                recommendation = {
                    "new_low": round(new_low_val, 6),
                    "new_high": round(new_high_val, 6),
                    "old_low": low,
                    "old_high": high,
                }
                rationale_parts.append(
                    f"Top trials tightly concentrated (CV={top_cv:.3f}). Narrowing to "
                    f"[{recommendation['new_low']}, {recommendation['new_high']}]."
                )
        else:
            new_low_val = denormalize_log_value(new_low_t, use_log_scale)
            new_high_val = denormalize_log_value(new_high_t, use_log_scale)
            action = "narrow"
            recommendation = {
                "new_low": round(new_low_val, 6),
                "new_high": round(new_high_val, 6),
                "old_low": low,
                "old_high": high,
            }
            rationale_parts.append(
                f"Top trials tightly concentrated (CV={top_cv:.3f}). Narrowing to "
                f"[{recommendation['new_low']}, {recommendation['new_high']}]."
            )

    if action == "keep" and monotonic is not None:
        if upper_edge_signal and not upper_correlation_ok:
            if upper_cost_sensitive and (
                monotonic < float(cost_sensitive_upper_rho) or importance < 0.1
            ):
                rationale_parts.append(
                    "Directional expansion blocked: cost-sensitive parameter requires strong "
                    f"monotonic gain evidence (Spearman={monotonic:.3f})."
                )
            else:
                rationale_parts.append(
                    "Directional expansion blocked: upper-edge evidence conflicts with "
                    f"monotonic trend (Spearman={monotonic:.3f})."
                )
        elif lower_edge_signal and not lower_correlation_ok:
            rationale_parts.append(
                "Directional expansion blocked: lower-edge evidence conflicts with "
                f"monotonic trend (Spearman={monotonic:.3f})."
            )
    elif (
        action == "keep"
        and weak_monotonic_evidence
        and (upper_edge_signal or lower_edge_signal)
    ):
        rationale_parts.append(
            "Directional expansion blocked: weak monotonic evidence and low parameter cardinality."
        )

    if importance < float(low_importance_threshold) and action == "keep":
        mid_t = top_stats["q50"]
        mid = denormalize_log_value(mid_t, use_log_scale)
        action = "fix"
        recommendation = {"fix_value": round(mid, 6)}
        rationale_parts.append(
            f"Low importance ({importance:.3f}). Consider fixing at median of top-k ({mid:.4g})."
        )

    has_explicit_log = bool(parsed.get("log_specified"))
    if param_meta.is_log and not has_explicit_log:
        rationale_parts.append(
            "Parameter name/range suggests log-uniform distribution."
        )
        if action == "keep":
            action = "change_distribution"
            recommendation = {"distribution": "log_uniform", "low": low, "high": high}
    elif bool(parsed.get("log")) and not use_log_scale:
        rationale_parts.append(
            "Log-scale disabled for this parameter because bounds include non-positive values."
        )

    if not rationale_parts:
        rationale_parts.append(
            "Current search space appears well-calibrated for this parameter."
        )

    return action, recommendation, rationale_parts, surrogate_bounds


__all__ = [
    "decide_numeric_action",
    "ballet_safe_shrink",
    "surrogate_grid_bounds",
]

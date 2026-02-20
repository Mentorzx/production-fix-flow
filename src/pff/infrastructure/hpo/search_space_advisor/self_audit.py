"""Self-audit helpers for directional recommendation reliability."""

from __future__ import annotations

from typing import Any, Callable


def is_directional_action(action: str) -> bool:
    """Return whether an action carries directional expectation."""
    return action in {"expand_upper", "expand_lower"}


def audit_prefix_sizes(
    n_completed_trials: int,
    *,
    min_prefix: int,
    min_suffix: int,
    period_trials: int,
    max_prefixes: int,
) -> list[int]:
    """Build periodic prefix checkpoints used by self-audit."""
    min_trials = min_prefix + min_suffix
    if n_completed_trials < min_trials:
        return []
    all_prefixes = list(range(min_prefix, n_completed_trials - min_suffix + 1, period_trials))
    if not all_prefixes:
        return []
    if len(all_prefixes) <= max_prefixes:
        return all_prefixes

    step = float(len(all_prefixes) - 1) / float(max(1, max_prefixes - 1))
    selected_idx: list[int] = []
    seen: set[int] = set()
    for pos in range(max_prefixes):
        idx = int(round(pos * step))
        idx = max(0, min(len(all_prefixes) - 1, idx))
        if idx not in seen:
            selected_idx.append(idx)
            seen.add(idx)

    if (len(all_prefixes) - 1) not in seen:
        selected_idx.append(len(all_prefixes) - 1)
    selected_idx = sorted(set(selected_idx))
    return [all_prefixes[idx] for idx in selected_idx]


def match_directional_suffix_trend(
    *,
    action: str,
    param_name: str,
    suffix_trials: list[Any],
    direction: str,
    min_points: int,
    apply_direction: Callable[[float, str], float],
    spearman_rho: Callable[[list[float], list[float]], float | None],
) -> tuple[bool | None, float | None]:
    """Validate directional action against suffix-trial monotonic trend."""
    values: list[float] = []
    adjusted_scores: list[float] = []
    for trial in suffix_trials:
        params = getattr(trial, "params", {})
        value = params.get(param_name) if isinstance(params, dict) else None
        if not isinstance(value, (int, float)):
            continue
        score = getattr(trial, "value", None)
        if not isinstance(score, (int, float)):
            continue
        values.append(float(value))
        adjusted_scores.append(apply_direction(float(score), direction))

    if len(values) < max(2, int(min_points)):
        return None, None
    rho = spearman_rho(values, adjusted_scores)
    if rho is None:
        return None, None
    if action == "expand_upper":
        return bool(rho > 0), float(rho)
    if action == "expand_lower":
        return bool(rho < 0), float(rho)
    return None, float(rho)


def apply_self_audit_blocks(
    *,
    recommendations: list[dict[str, Any]],
    self_audit: dict[str, Any],
    wilson_block_threshold: float,
) -> int:
    """Apply self-audit villain gates to directional recommendations."""
    villains = self_audit.get("villains", [])
    villain_map = {
        (str(item.get("param_name", "")), str(item.get("action", ""))): item
        for item in villains
        if isinstance(item, dict)
    }
    blocked_by_self_audit = 0
    if villain_map:
        for recommendation in recommendations:
            action = str(recommendation.get("action", "keep"))
            if not is_directional_action(action):
                continue
            param_name = str(recommendation.get("param_name", ""))
            villain = villain_map.get((param_name, action))
            if villain is None:
                continue
            recommendation["blocked_action"] = action
            recommendation["blocked_by"] = "self_audit"
            recommendation["action"] = "keep"
            recommendation["recommendation"] = {
                "delta": f"blocked:self_audit_wilson_lb<{wilson_block_threshold:.2f}"
            }
            current_lb = villain.get("hit_rate_wilson_lb")
            rationale = str(recommendation.get("rationale", "")).strip()
            block_msg = (
                "Recommendation blocked by periodic self-audit due to weak directional "
                f"reliability (Wilson LB={current_lb})."
            )
            recommendation["rationale"] = (
                f"{rationale} {block_msg}".strip() if rationale else block_msg
            )
            blocked_by_self_audit += 1
    self_audit["blocked_actions_current"] = blocked_by_self_audit
    return blocked_by_self_audit


__all__ = [
    "apply_self_audit_blocks",
    "audit_prefix_sizes",
    "is_directional_action",
    "match_directional_suffix_trend",
]

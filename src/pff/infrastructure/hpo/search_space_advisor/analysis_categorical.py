"""Categorical-analysis helpers for Search Space Advisor."""

from __future__ import annotations

import math
from typing import Any


def categorical_counts(values: list[Any]) -> dict[str, int]:
    """Count categorical occurrences after deterministic string coercion."""
    counts: dict[str, int] = {}
    for value in values:
        token = str(value)
        counts[token] = counts.get(token, 0) + 1
    return counts


def canonical_category(value: Any) -> str:
    """Canonical category token representation."""
    return str(value)


def materialize_category_sets(
    choices: list[Any],
    keep_tokens: set[str],
) -> tuple[list[Any], list[Any]]:
    """Materialize keep/remove lists preserving original choice values."""
    token_to_choice: dict[str, Any] = {}
    for choice in choices:
        token = canonical_category(choice)
        if token not in token_to_choice:
            token_to_choice[token] = choice
    keep_values = [
        token_to_choice[token]
        for token in sorted(keep_tokens)
        if token in token_to_choice
    ]
    remove_values = [
        choice for choice in choices if canonical_category(choice) not in keep_tokens
    ]
    return keep_values, remove_values


def decide_categorical_action(
    *,
    parsed: dict[str, Any],
    top_k_values: list[Any],
    importance: float,
    surrogate: Any | None,
    anchor_params: dict[str, Any],
    param_name: str,
    param_meta_map: dict[str, Any],
    encode_params_fn: Any,
    predict_surrogate_fn: Any,
    interaction_strength: float,
    interaction_threshold: float,
    categorical_dominance_threshold: float,
    low_importance_threshold: float,
    ucb_std_mult: float,
    min_topk_samples: int,
    min_topk_unique: int,
    min_effective_categories: float,
) -> tuple[str, dict[str, Any], list[str]]:
    """Decide categorical action and rationale."""
    choices = list(parsed.get("choices", []))
    top_counts = categorical_counts(top_k_values)
    total_top = len(top_k_values) if top_k_values else 1

    action = "keep"
    recommendation: dict[str, Any] = {"delta": "none"}
    rationale_parts: list[str] = []
    reduction_block_reason: str | None = None

    dominant_categories: list[str] = []
    for category, count in top_counts.items():
        share = count / total_top
        if share >= float(categorical_dominance_threshold):
            dominant_categories.append(category)

    observed_top_unique = len(top_counts)
    entropy = 0.0
    if total_top > 0:
        for count in top_counts.values():
            if count <= 0:
                continue
            probability = float(count) / float(total_top)
            entropy -= probability * math.log(max(probability, 1e-12))
    effective_categories = math.exp(entropy) if observed_top_unique > 0 else 0.0

    def _allow_reduction() -> bool:
        nonlocal reduction_block_reason
        if int(total_top) < int(max(1, min_topk_samples)):
            reduction_block_reason = f"insufficient_topk_samples(total_top={int(total_top)}, min={int(min_topk_samples)})"
            return False
        if int(observed_top_unique) < int(max(1, min_topk_unique)):
            reduction_block_reason = (
                "insufficient_topk_unique_categories("
                f"observed={int(observed_top_unique)}, min={int(min_topk_unique)})"
            )
            return False
        if float(effective_categories) < float(min_effective_categories) and int(
            total_top
        ) < int(max(1, min_topk_samples) * 2):
            reduction_block_reason = (
                "insufficient_effective_category_evidence("
                f"effective={effective_categories:.3f}, min={float(min_effective_categories):.3f})"
            )
            return False
        return True

    if surrogate is not None and len(choices) > 2:
        rows = []
        for choice in choices:
            params = dict(anchor_params)
            params[param_name] = choice
            rows.append(encode_params_fn(params, param_meta_map))
        means, stds = predict_surrogate_fn(surrogate, rows)
        ucb = [
            mean + float(ucb_std_mult) * std
            for mean, std in zip(means, stds, strict=False)
        ]
        lcb = [
            mean - float(ucb_std_mult) * std
            for mean, std in zip(means, stds, strict=False)
        ]
        best_idx = max(range(len(choices)), key=lambda idx: lcb[idx], default=0)
        best_lcb = lcb[best_idx]
        keep_tokens = {canonical_category(choices[best_idx])}
        for idx, choice in enumerate(choices):
            if ucb[idx] >= best_lcb:
                keep_tokens.add(canonical_category(choice))
        keep_values, remove_values = materialize_category_sets(choices, keep_tokens)
        if remove_values and len(keep_values) >= 2:
            if _allow_reduction():
                action = "reduce_categories"
                recommendation = {"keep": keep_values, "remove": remove_values}
                rationale_parts.append(
                    "Surrogate UCB/LCB suggests some categories are consistently worse than the best ones."
                )
    elif dominant_categories and len(choices) > 2:
        keep_tokens = set(dominant_categories)
        if len(keep_tokens) < 2:
            runner_up = max(
                (
                    (
                        canonical_category(choice),
                        top_counts.get(canonical_category(choice), 0),
                    )
                    for choice in choices
                    if canonical_category(choice) not in keep_tokens
                ),
                key=lambda item: item[1],
                default=(None, 0),
            )
            if runner_up[0] is not None:
                keep_tokens.add(runner_up[0])
        keep_values, remove_values = materialize_category_sets(choices, keep_tokens)
        if _allow_reduction():
            action = "reduce_categories"
            recommendation = {"keep": keep_values, "remove": remove_values}
            keep_list = sorted(keep_values, key=lambda value: str(value))
            rationale_parts.append(
                f"Categories {dominant_categories} dominate top-k "
                f"({', '.join(f'{c}={top_counts.get(c, 0) / total_top:.0%}' for c in dominant_categories)}). "
                f"Consider reducing to {keep_list}."
            )

    if importance < float(low_importance_threshold) and action == "keep":
        most_common = max(
            top_counts, key=lambda category: top_counts[category], default=None
        )
        if most_common is not None:
            action = "fix"
            recommendation = {"fix_value": most_common}
            rationale_parts.append(
                f"Low importance ({importance:.3f}). Consider fixing at most common top-k value: {most_common}."
            )

    if interaction_strength > float(interaction_threshold) and action in {
        "fix",
        "reduce_categories",
    }:
        action = "keep"
        recommendation = {"delta": "none"}
        rationale_parts.append(
            "Strong interactions detected; avoiding category reduction/fix."
        )
    elif action == "keep" and reduction_block_reason:
        rationale_parts.append(f"Category reduction blocked: {reduction_block_reason}.")

    if not rationale_parts:
        rationale_parts.append(
            "Category distribution appears balanced in top-k trials."
        )

    return action, recommendation, rationale_parts


__all__ = [
    "canonical_category",
    "categorical_counts",
    "decide_categorical_action",
    "materialize_category_sets",
]

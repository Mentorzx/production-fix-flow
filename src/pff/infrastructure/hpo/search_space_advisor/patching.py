"""Search-space patch generation helpers."""

from __future__ import annotations

from typing import Any


def generate_search_space_patch(
    recommendations: list[dict[str, Any]],
    current_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate preview patch from recommendation payloads."""
    _ = current_config
    patch: dict[str, Any] = {}
    for recommendation in recommendations:
        action = recommendation.get("action", "keep")
        param = recommendation["param_name"]
        delta = recommendation.get("recommendation", {})
        validation = recommendation.get("validation", {})
        if isinstance(validation, dict) and validation.get("passed") is False:
            continue

        if action == "keep":
            continue
        if action == "expand_upper":
            new_high = delta.get("new_high")
            current = recommendation.get("current_space", {})
            patch[param] = {**current, "high": new_high}
        elif action == "expand_lower":
            new_low = delta.get("new_low")
            current = recommendation.get("current_space", {})
            patch[param] = {**current, "low": new_low}
        elif action == "narrow":
            current = recommendation.get("current_space", {})
            patch[param] = {
                **current,
                "low": delta.get("new_low"),
                "high": delta.get("new_high"),
            }
        elif action == "fix":
            patch[param] = {"type": "fixed", "value": delta.get("fix_value")}
        elif action == "reduce_categories":
            patch[param] = {"type": "categorical", "choices": delta.get("keep", [])}
        elif action == "change_distribution":
            patch[param] = {
                "type": delta.get("distribution", "log_uniform"),
                "low": delta.get("low"),
                "high": delta.get("high"),
                "log": True,
            }
    return patch


__all__ = ["generate_search_space_patch"]


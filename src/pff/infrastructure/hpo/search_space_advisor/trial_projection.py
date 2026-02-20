"""Trial projection helpers for Search Space Advisor."""

from __future__ import annotations

from typing import Any, Callable

from .models import TrialSummary


def extract_pruned_value(intermediate_values: dict[int, float] | None) -> float | None:
    """Extract last numeric intermediate value from a pruned trial."""
    if not intermediate_values:
        return None
    try:
        last_step = max(intermediate_values)
    except ValueError:
        return None
    value = intermediate_values.get(last_step)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_trial_summaries(
    trials_data: list[dict[str, Any]],
    projected_scores: list[float | None],
    *,
    normalize_trial_state_fn: Callable[[Any], str],
) -> tuple[list[TrialSummary], list[TrialSummary]]:
    """Build all/completed trial summaries from raw trial payloads."""
    completed: list[TrialSummary] = []
    all_trials: list[TrialSummary] = []
    for idx, trial in enumerate(trials_data):
        params = trial.get("params")
        if not isinstance(params, dict):
            continue
        state = normalize_trial_state_fn(trial.get("state", "COMPLETE"))
        raw_value = trial.get("value")
        if raw_value is None:
            values_list = trial.get("values")
            if isinstance(values_list, (list, tuple)) and values_list:
                raw_value = values_list[0]
        projected_value = projected_scores[idx] if idx < len(projected_scores) else None
        if projected_value is None and raw_value is None and state == "PRUNED":
            projected_value = extract_pruned_value(trial.get("intermediate_values"))
        if projected_value is None and raw_value is None:
            continue

        value_for_ranking = (
            float(projected_value)
            if isinstance(projected_value, (int, float))
            else float(raw_value)
        )
        raw_value_for_report = (
            float(raw_value)
            if isinstance(raw_value, (int, float))
            else float(value_for_ranking)
        )
        summary = TrialSummary(
            number=int(trial.get("id", trial.get("number", 0))),
            value=float(value_for_ranking),
            raw_value=float(raw_value_for_report),
            params=params,
            state=state,
            intermediate_values=trial.get("intermediate_values"),
        )
        all_trials.append(summary)
        if state == "COMPLETE":
            completed.append(summary)
    return all_trials, completed


__all__ = [
    "build_trial_summaries",
    "extract_pruned_value",
]

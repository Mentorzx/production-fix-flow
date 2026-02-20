"""Trust-state update helpers for Search Space Advisor."""

from __future__ import annotations

from typing import Callable

from .models import ParamMeta, TrialSummary, TrustState


def select_best_trial(
    completed_trials: list[TrialSummary],
    *,
    direction: str,
) -> TrialSummary | None:
    """Select best completed trial according to normalized direction."""
    if not completed_trials:
        return None
    if direction == "maximize":
        return max(completed_trials, key=lambda trial: trial.value)
    return min(completed_trials, key=lambda trial: trial.value)


def update_trust_bucket(
    *,
    trust_bucket: dict[str, TrustState],
    param_meta_map: dict[str, ParamMeta],
    completed_trials: list[TrialSummary],
    direction: str,
    edge_threshold: float,
    trust_failure_threshold: int,
    normalize_log_value_fn: Callable[[float, bool], float],
) -> TrialSummary | None:
    """Update per-parameter trust state from best-trial boundary proximity."""
    best_trial = select_best_trial(completed_trials, direction=direction)
    if best_trial is None:
        return None

    for param_name, meta in param_meta_map.items():
        trust_state = trust_bucket.setdefault(param_name, TrustState())
        if trust_state.last_trial == best_trial.number:
            continue

        best_value = best_trial.value
        improved = trust_state.best_value is None or best_value > trust_state.best_value + 1e-12
        if improved:
            trust_state.best_value = best_value
            trust_state.best_params = dict(best_trial.params)

        if (
            meta.param_type in {"float", "int"}
            and meta.low is not None
            and meta.high is not None
        ):
            low_t = normalize_log_value_fn(float(meta.low), bool(meta.is_log))
            high_t = normalize_log_value_fn(float(meta.high), bool(meta.is_log))
            span_t = high_t - low_t
            value = best_trial.params.get(param_name)
            if value is not None and span_t > 0:
                val_t = normalize_log_value_fn(float(value), bool(meta.is_log))
                proximity = (val_t - low_t) / span_t
                if improved and proximity > (1 - edge_threshold):
                    trust_state.upper_success += 1
                    trust_state.failure = 0
                elif improved and proximity < edge_threshold:
                    trust_state.lower_success += 1
                    trust_state.failure = 0
                else:
                    trust_state.failure += 1
        else:
            trust_state.failure += 1

        if trust_state.failure >= trust_failure_threshold:
            trust_state.upper_success = 0
            trust_state.lower_success = 0
        trust_state.last_trial = best_trial.number

    return best_trial


__all__ = [
    "select_best_trial",
    "update_trust_bucket",
]

"""Shared diagnostics for HPO stagnation and local-optima heuristics."""

from __future__ import annotations

import math
from statistics import median
from typing import Any

from pff.infrastructure.hpo.search_space_advisor.parsing import (
    normalize_direction,
    normalize_trial_state,
    parse_search_space_entry,
)

DEFAULT_STAGNATION_WINDOW_SIZE = 7
DEFAULT_STAGNATION_MIN_TRIALS = 10
DEFAULT_STAGNATION_IMPROVEMENT_THRESHOLD = 0.02
DEFAULT_MULTI_REGION_MIN_TRIALS = 12
DEFAULT_ELITE_FRACTION = 0.20
DEFAULT_ELITE_MIN = 6
DEFAULT_ELITE_MAX = 12
DEFAULT_SIGNATURE_PARAM_LIMIT = 4
COMPETITIVE_REGION_TOLERANCE = 0.01


def _coerce_finite_number(value: Any) -> float | None:
    """Convert numeric-like values into finite floats."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _is_better(candidate: float, incumbent: float, direction: str) -> bool:
    """Return whether a candidate beats the incumbent for the objective direction."""
    if direction == "minimize":
        return candidate < incumbent
    return candidate > incumbent


def _compute_recent_range(values: list[float], direction: str) -> float | None:
    """Compute the recent relative spread using the observer-compatible reference."""
    if not values:
        return None
    recent_best = max(values)
    recent_worst = min(values)
    reference = recent_worst if direction == "maximize" else recent_best
    if reference == 0:
        return 0.0
    return (recent_best - recent_worst) / abs(reference)


def analyze_stagnation(
    scores: list[float] | tuple[float, ...],
    *,
    direction: str = "maximize",
    trial_numbers: list[int] | tuple[int, ...] | None = None,
    window_size: int = DEFAULT_STAGNATION_WINDOW_SIZE,
    min_trials: int = DEFAULT_STAGNATION_MIN_TRIALS,
    improvement_threshold: float = DEFAULT_STAGNATION_IMPROVEMENT_THRESHOLD,
) -> dict[str, Any]:
    """Analyze score history for HPO stagnation."""
    normalized_direction = normalize_direction(direction)
    values_with_ids: list[tuple[float, int]] = []
    for index, score in enumerate(scores):
        numeric = _coerce_finite_number(score)
        if numeric is None:
            continue
        trial_number = index
        if trial_numbers is not None and index < len(trial_numbers):
            raw_trial_number = _coerce_finite_number(trial_numbers[index])
            if raw_trial_number is not None:
                trial_number = int(raw_trial_number)
        values_with_ids.append((numeric, trial_number))
    if not values_with_ids:
        return {
            "status": "insufficient_evidence",
            "stagnant": False,
            "recent_range": None,
            "trials_since_improvement": 0,
            "best_score": None,
            "best_trial_number": None,
        }
    values = [value for value, _ in values_with_ids]

    best_score, best_trial_number = values_with_ids[0]
    trials_since_improvement = 0
    for value, trial_number in values_with_ids[1:]:
        if _is_better(value, best_score, normalized_direction):
            best_score = value
            best_trial_number = trial_number
            trials_since_improvement = 0
        else:
            trials_since_improvement += 1

    recent_range = None
    if len(values) >= window_size:
        recent_range = _compute_recent_range(values[-window_size:], normalized_direction)

    stagnant = bool(
        len(values) >= min_trials
        and recent_range is not None
        and recent_range <= improvement_threshold
        and trials_since_improvement >= window_size
    )
    status = (
        "insufficient_evidence"
        if len(values) < min_trials
        else "stagnant"
        if stagnant
        else "exploring"
    )
    return {
        "status": status,
        "stagnant": stagnant,
        "recent_range": recent_range,
        "trials_since_improvement": trials_since_improvement,
        "best_score": best_score,
        "best_trial_number": best_trial_number,
    }


def _coerce_trial_record(trial: Any, order: int) -> dict[str, Any] | None:
    """Normalize a trial-like object or payload into a comparable dictionary."""
    if isinstance(trial, dict):
        params = trial.get("params") if isinstance(trial.get("params"), dict) else {}
        record_id = _coerce_finite_number(trial.get("id"))
        if record_id is None:
            raw_number = _coerce_finite_number(trial.get("number"))
            record_id = raw_number + 1 if raw_number is not None else float(order + 1)
        return {
            "id": int(record_id),
            "order": order,
            "state": normalize_trial_state(trial.get("state")),
            "value": _coerce_finite_number(trial.get("value")),
            "params": params,
            "warmstart": bool(trial.get("warmstart")),
        }

    params = getattr(trial, "params", None)
    if not isinstance(params, dict):
        params = {}
    record_id = _coerce_finite_number(getattr(trial, "id", None))
    if record_id is None:
        raw_number = _coerce_finite_number(getattr(trial, "number", None))
        record_id = raw_number + 1 if raw_number is not None else float(order + 1)
    return {
        "id": int(record_id),
        "order": order,
        "state": normalize_trial_state(getattr(trial, "state", None)),
        "value": _coerce_finite_number(getattr(trial, "value", None)),
        "params": params,
        "warmstart": bool(getattr(trial, "warmstart", False)),
    }


def _completed_trials(
    trials: list[Any] | tuple[Any, ...],
) -> list[dict[str, Any]]:
    """Filter COMPLETE trials with numeric objective values."""
    normalized: list[dict[str, Any]] = []
    for order, trial in enumerate(trials):
        record = _coerce_trial_record(trial, order)
        if not record:
            continue
        if record["state"] != "COMPLETE":
            continue
        if record["value"] is None:
            continue
        normalized.append(record)
    return sorted(normalized, key=lambda item: (item["id"], item["order"]))


def _median_value(values: list[float]) -> float:
    """Return a deterministic median float."""
    return float(median(values))


def _observed_numeric_ranges(trials: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Collect numeric min/max ranges per parameter across eligible trials."""
    ranges: dict[str, list[float]] = {}
    for trial in trials:
        params = trial.get("params", {})
        if not isinstance(params, dict):
            continue
        for param_name, raw_value in params.items():
            numeric = _coerce_finite_number(raw_value)
            if numeric is None:
                continue
            ranges.setdefault(str(param_name), []).append(numeric)
    observed: dict[str, tuple[float, float]] = {}
    for param_name, values in ranges.items():
        if not values:
            continue
        observed[param_name] = (min(values), max(values))
    return observed


def _bucket_numeric_value(
    param_name: str,
    value: float,
    *,
    search_space: dict[str, Any],
    observed_ranges: dict[str, tuple[float, float]],
) -> str:
    """Bucket numeric values into low/mid/high using configured or observed bounds."""
    parsed = parse_search_space_entry(param_name, search_space.get(param_name))
    low = _coerce_finite_number(parsed.get("low"))
    high = _coerce_finite_number(parsed.get("high"))
    if low is None or high is None or high <= low:
        low, high = observed_ranges.get(param_name, (low, high))
    if low is None or high is None or high <= low:
        return "mid"

    use_log = bool(parsed.get("log")) and low > 0 and high > 0 and value > 0
    if use_log:
        denominator = math.log(high) - math.log(low)
        if denominator == 0:
            return "mid"
        position = (math.log(value) - math.log(low)) / denominator
    else:
        denominator = high - low
        if denominator == 0:
            return "mid"
        position = (value - low) / denominator

    clamped = max(0.0, min(1.0, position))
    if clamped < 1.0 / 3.0:
        return "low"
    if clamped < 2.0 / 3.0:
        return "mid"
    return "high"


def _label_param_value(
    param_name: str,
    value: Any,
    *,
    search_space: dict[str, Any],
    observed_ranges: dict[str, tuple[float, float]],
) -> str:
    """Convert a raw parameter value into a signature label."""
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return str(value).lower()

    numeric = _coerce_finite_number(value)
    if numeric is not None:
        return _bucket_numeric_value(
            param_name,
            numeric,
            search_space=search_space,
            observed_ranges=observed_ranges,
        )
    return str(value)


def _select_signature_params(
    elite_trials: list[dict[str, Any]],
    *,
    search_space: dict[str, Any],
    observed_ranges: dict[str, tuple[float, float]],
) -> list[str]:
    """Pick the small set of varying parameters used to define region signatures."""
    candidate_names = sorted(
        {
            str(param_name)
            for trial in elite_trials
            for param_name in (trial.get("params") or {}).keys()
        }
    )
    varying: list[tuple[int, str]] = []
    for param_name in candidate_names:
        labels = {
            _label_param_value(
                param_name,
                (trial.get("params") or {}).get(param_name),
                search_space=search_space,
                observed_ranges=observed_ranges,
            )
            for trial in elite_trials
        }
        if len(labels) > 1:
            varying.append((len(labels), param_name))
    varying.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in varying[:DEFAULT_SIGNATURE_PARAM_LIMIT]]


def analyze_multi_region_evidence(
    trials: list[Any] | tuple[Any, ...],
    search_space: dict[str, Any] | None,
    *,
    direction: str = "maximize",
) -> dict[str, Any]:
    """Search for conservative evidence of multiple competitive parameter regions."""
    normalized_direction = normalize_direction(direction)
    completed = _completed_trials(trials)
    preferred = [trial for trial in completed if not trial.get("warmstart")]
    eligible = preferred if preferred else completed
    if len(eligible) < DEFAULT_MULTI_REGION_MIN_TRIALS:
        return {
            "detected": False,
            "status": "insufficient_evidence",
            "region_count": 0,
            "supporting_trials": 0,
            "summary_labels": [],
            "regions": [],
            "eligible_trials": len(eligible),
            "elite_trials": 0,
        }

    ordered = sorted(
        eligible,
        key=lambda trial: trial["value"],
        reverse=normalized_direction == "maximize",
    )
    elite_size = min(
        len(ordered),
        max(
            DEFAULT_ELITE_MIN,
            min(DEFAULT_ELITE_MAX, math.ceil(len(ordered) * DEFAULT_ELITE_FRACTION)),
        ),
    )
    elite_trials = ordered[:elite_size]
    observed_ranges = _observed_numeric_ranges(eligible)
    safe_search_space = search_space if isinstance(search_space, dict) else {}
    signature_params = _select_signature_params(
        elite_trials,
        search_space=safe_search_space,
        observed_ranges=observed_ranges,
    )
    if not signature_params:
        return {
            "detected": False,
            "status": "single_region",
            "region_count": 0,
            "supporting_trials": 0,
            "summary_labels": [],
            "regions": [],
            "eligible_trials": len(eligible),
            "elite_trials": elite_size,
        }

    grouped: dict[str, dict[str, Any]] = {}
    for trial in elite_trials:
        parts = []
        for param_name in signature_params:
            label = _label_param_value(
                param_name,
                (trial.get("params") or {}).get(param_name),
                search_space=safe_search_space,
                observed_ranges=observed_ranges,
            )
            parts.append(f"{param_name}={label}")
        signature = " | ".join(parts)
        bucket = grouped.setdefault(
            signature,
            {"label": signature, "scores": [], "members": []},
        )
        bucket["scores"].append(float(trial["value"]))
        bucket["members"].append(
            {"id": int(trial["id"]), "value": float(trial["value"])}
        )

    best_elite_score = float(elite_trials[0]["value"])
    tolerance = max(abs(best_elite_score), 1.0) * COMPETITIVE_REGION_TOLERANCE
    ranked_regions: list[dict[str, Any]] = []
    for region in grouped.values():
        region_scores = sorted(region["scores"])
        region_median = _median_value(region_scores)
        best_member = min(
            region["members"],
            key=lambda member: member["value"]
            if normalized_direction == "minimize"
            else -member["value"],
        )
        ranked_regions.append(
            {
                "label": region["label"],
                "support": len(region["scores"]),
                "median_score": region_median,
                "best_trial_id": best_member["id"],
                "competitive": abs(region_median - best_elite_score) <= tolerance,
            }
        )

    ranked_regions.sort(
        key=lambda region: (
            -int(region["support"]),
            region["median_score"]
            if normalized_direction == "minimize"
            else -region["median_score"],
            region["label"],
        )
    )
    competitive_regions = [
        region
        for region in ranked_regions
        if region["support"] >= 2 and region["competitive"]
    ]
    summary_labels = [region["label"] for region in competitive_regions[:2]]
    return {
        "detected": len(competitive_regions) >= 2,
        "status": "multiple_regions" if len(competitive_regions) >= 2 else "single_region",
        "region_count": len(competitive_regions),
        "supporting_trials": sum(region["support"] for region in competitive_regions),
        "summary_labels": summary_labels,
        "regions": competitive_regions[:3],
        "eligible_trials": len(eligible),
        "elite_trials": elite_size,
    }


def _recommended_action(status: str, multi_region_evidence: dict[str, Any]) -> str:
    """Choose the user-facing recommendation for the current diagnosis."""
    if status == "stagnant" and multi_region_evidence.get("detected"):
        return (
            "Retome a exploracao nas regioes competitivas e considere sampler.type='cmaes'."
        )
    if status == "stagnant":
        return (
            "Considere reiniciar com sampler.type='cmaes' e revisar bounds que ficaram flat."
        )
    if status == "multiple_regions":
        return "Explore separadamente as regioes competitivas antes de estreitar o search space."
    if status == "exploring":
        return "Mantenha a exploracao atual; ainda ha melhora recente ou variacao util."
    return "Aguarde mais trials completos antes de concluir sobre minimos locais."


def build_local_optima_diagnostics(
    trials: list[Any] | tuple[Any, ...],
    search_space: dict[str, Any] | None,
    *,
    direction: str = "maximize",
    current_sampler: str | None = None,
    window_size: int = DEFAULT_STAGNATION_WINDOW_SIZE,
    min_trials: int = DEFAULT_STAGNATION_MIN_TRIALS,
    improvement_threshold: float = DEFAULT_STAGNATION_IMPROVEMENT_THRESHOLD,
) -> dict[str, Any]:
    """Build the dashboard-ready local-optima diagnostics payload."""
    completed = _completed_trials(trials)
    stagnation = analyze_stagnation(
        [trial["value"] for trial in completed],
        direction=direction,
        window_size=window_size,
        min_trials=min_trials,
        improvement_threshold=improvement_threshold,
    )
    best_trial_id = None
    best_trial_number = stagnation.get("best_trial_number")
    if isinstance(best_trial_number, int) and 0 <= best_trial_number < len(completed):
        best_trial_id = completed[best_trial_number]["id"]

    multi_region_evidence = analyze_multi_region_evidence(
        completed,
        search_space,
        direction=direction,
    )
    if stagnation["status"] == "insufficient_evidence":
        status = "insufficient_evidence"
    elif stagnation["stagnant"]:
        status = "stagnant"
    elif multi_region_evidence.get("detected"):
        status = "multiple_regions"
    else:
        status = "exploring"

    return {
        "status": status,
        "stagnant": bool(stagnation["stagnant"]),
        "trials_since_improvement": int(stagnation["trials_since_improvement"]),
        "recent_range": stagnation.get("recent_range"),
        "best_trial_id": best_trial_id,
        "best_score": stagnation.get("best_score"),
        "current_sampler": str(current_sampler or "Unknown"),
        "recommended_action": _recommended_action(status, multi_region_evidence),
        "multi_region_evidence": multi_region_evidence,
        "completed_trials": len(completed),
    }


__all__ = [
    "DEFAULT_STAGNATION_IMPROVEMENT_THRESHOLD",
    "DEFAULT_STAGNATION_MIN_TRIALS",
    "DEFAULT_STAGNATION_WINDOW_SIZE",
    "analyze_multi_region_evidence",
    "analyze_stagnation",
    "build_local_optima_diagnostics",
]

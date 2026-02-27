"""Parsing and normalization helpers for Search Space Advisor."""

from __future__ import annotations

from typing import Any

from pff.shared.core.file_manager import FileManager


def normalize_direction(direction: Any) -> str:
    """Normalize objective direction to ``maximize`` or ``minimize``."""
    raw = str(direction or "maximize").strip().lower()
    if "." in raw:
        raw = raw.split(".")[-1]
    if raw in {"maximize", "max"}:
        return "maximize"
    if raw in {"minimize", "min"}:
        return "minimize"
    return "maximize"


def normalize_trial_state(state: Any) -> str:
    """Normalize Optuna trial state names."""
    raw = str(state or "COMPLETE").strip().upper()
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw


def is_log_scale_candidate(param_name: str, low: float, high: float) -> bool:
    """Heuristic to infer log-scale recommendation when spec omits ``log``."""
    if low <= 0 or high <= 0:
        return False
    log_hints = (
        "lr",
        "learning_rate",
        "weight_decay",
        "lambda",
        "kl_weight",
        "min_delta",
    )
    name_lower = param_name.lower()
    if any(hint in name_lower for hint in log_hints):
        return True
    return bool(high / low > 100)


def is_cost_sensitive_param(param_name: str) -> bool:
    """Return whether a parameter is likely tied to runtime/computational cost."""
    key = param_name.lower()
    return any(
        token in key
        for token in (
            "negative_sample_size",
            "num_global_negatives",
            "epochs",
            "validate_every",
            "patience",
            "rebuild_every",
            "batch_size",
        )
    )


def parse_search_space_entry(param_name: str, spec: Any) -> dict[str, Any]:
    """Parse a search-space entry into a normalized dictionary."""
    if spec is None:
        return {"type": "unknown"}

    if isinstance(spec, str):
        raw = spec.strip()
        if raw.startswith("{") or raw.startswith("["):
            try:
                return parse_search_space_entry(param_name, FileManager.json_loads(raw))
            except Exception:
                return {"type": "fixed", "value": spec}
        return {"type": "fixed", "value": spec}

    if isinstance(spec, dict):
        dist_name = spec.get("name", spec.get("type", ""))
        attrs = spec.get("attributes", spec.get("params", spec))
        low = attrs.get("low", attrs.get("min"))
        high = attrs.get("high", attrs.get("max"))
        choices = attrs.get("choices")
        log_specified = "log" in attrs
        log = attrs.get("log", False)
        step = attrs.get("step")

        if choices is not None:
            return {"type": "categorical", "choices": list(choices)}

        if low is not None and high is not None:
            result: dict[str, Any] = {
                "type": (
                    "float"
                    if isinstance(low, float) or isinstance(high, float)
                    else "int"
                ),
                "low": low,
                "high": high,
            }
            if log_specified:
                result["log"] = bool(log)
                result["log_specified"] = True
            if step is not None:
                result["step"] = step
            if "Int" in str(dist_name):
                result["type"] = "int"
            if "Float" in str(dist_name):
                result["type"] = "float"
            return result

        return {"type": str(dist_name) or "unknown", "raw": spec}

    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and all(isinstance(x, (int, float)) for x in spec):
            return {
                "type": "float" if any(isinstance(x, float) for x in spec) else "int",
                "low": spec[0],
                "high": spec[1],
            }
        return {"type": "categorical", "choices": list(spec)}

    return {"type": "fixed", "value": spec}


__all__ = [
    "is_cost_sensitive_param",
    "is_log_scale_candidate",
    "normalize_direction",
    "normalize_trial_state",
    "parse_search_space_entry",
]

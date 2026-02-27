"""Surrogate-model and parameter-meta helpers for Search Space Advisor."""

from __future__ import annotations

import math
from typing import Any, Callable, cast

from .models import ParamMeta, SurrogateModel, TrialSummary


def normalize_log_value(value: float, *, is_log: bool) -> float:
    """Normalize numeric value to log10 domain when configured."""
    if not is_log:
        return value
    safe = max(value, 1e-12)
    return math.log10(safe)


def denormalize_log_value(value: float, *, is_log: bool) -> float:
    """Map normalized log-domain value back to raw domain."""
    if not is_log:
        return value
    return 10**value


def build_param_meta(
    search_space: dict[str, Any],
    *,
    parse_search_space_entry_fn: Callable[[str, Any], dict[str, Any]],
    is_log_scale_candidate_fn: Callable[[str, float, float], bool],
) -> dict[str, ParamMeta]:
    """Build normalized parameter metadata map from raw search-space specs."""
    meta: dict[str, ParamMeta] = {}
    for name, spec in search_space.items():
        parsed = parse_search_space_entry_fn(name, spec)
        param_type = parsed.get("type", "unknown")
        is_categorical = param_type == "categorical"
        is_log = False
        if param_type in {"float", "int"} and "low" in parsed and "high" in parsed:
            low = float(parsed["low"])
            high = float(parsed["high"])
            if parsed.get("log_specified"):
                is_log = bool(parsed.get("log"))
            else:
                is_log = is_log_scale_candidate_fn(name, low, high)
        meta[name] = ParamMeta(
            name=name,
            param_type=param_type,
            is_categorical=is_categorical,
            is_log=is_log,
            low=parsed.get("low"),
            high=parsed.get("high"),
            choices=parsed.get("choices"),
        )
    return meta


def extract_anchor_params(
    param_meta: dict[str, ParamMeta],
    trials: list[TrialSummary],
    *,
    categorical_counts_fn: Callable[[list[Any]], dict[str, int]],
) -> dict[str, Any]:
    """Build median/mode anchor params used for surrogate local probes."""
    anchors: dict[str, Any] = {}
    for name, meta in param_meta.items():
        values = [t.params.get(name) for t in trials if name in t.params]
        values = [v for v in values if v is not None]
        if not values:
            continue
        if meta.is_categorical:
            counts = categorical_counts_fn(values)
            anchors[name] = max(counts, key=lambda key: counts[key])
        else:
            numeric = [float(v) for v in values]
            numeric.sort()
            mid = numeric[len(numeric) // 2]
            anchors[name] = mid
    return anchors


def build_surrogate_data(
    trials: list[TrialSummary],
    param_meta: dict[str, ParamMeta],
    *,
    direction: str,
    normalize_log_value_fn: Callable[[float, bool], float],
    apply_direction_fn: Callable[[float, str], float],
) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    """Transform trials into surrogate training rows, targets, and sample weights."""
    rows: list[dict[str, Any]] = []
    y: list[float] = []
    weights: list[float] = []
    for trial in trials:
        if trial.value is None:
            continue
        row: dict[str, Any] = {}
        skip = False
        for name, meta in param_meta.items():
            if name not in trial.params:
                skip = True
                break
            value = trial.params.get(name)
            if value is None:
                skip = True
                break
            if meta.is_categorical:
                row[name] = str(value)
            else:
                row[name] = normalize_log_value_fn(float(value), bool(meta.is_log))
        if skip:
            continue
        rows.append(row)
        y.append(apply_direction_fn(float(trial.value), direction))
        weights.append(0.5 if trial.state == "PRUNED" else 1.0)
    return rows, y, weights


def encode_params(
    params: dict[str, Any],
    param_meta: dict[str, ParamMeta],
    *,
    normalize_log_value_fn: Callable[[float, bool], float],
) -> dict[str, Any]:
    """Encode params into the same feature layout expected by surrogate preprocessor."""
    row: dict[str, Any] = {}
    for name, meta in param_meta.items():
        value = params.get(name)
        if meta.is_categorical:
            row[name] = str(value) if value is not None else ""
        else:
            if value is None:
                row[name] = 0.0
            else:
                row[name] = normalize_log_value_fn(float(value), bool(meta.is_log))
    return row


def fit_surrogate(
    trials: list[TrialSummary],
    param_meta: dict[str, ParamMeta],
    *,
    direction: str,
    surrogate_min_trials: int,
    normalize_log_value_fn: Callable[[float, bool], float],
    apply_direction_fn: Callable[[float, str], float],
) -> SurrogateModel | None:
    """Fit RandomForest surrogate over complete+pruned trials when enough data exists."""
    if len(trials) < surrogate_min_trials:
        return None
    rows, y, weights = build_surrogate_data(
        trials,
        param_meta,
        direction=direction,
        normalize_log_value_fn=normalize_log_value_fn,
        apply_direction_fn=apply_direction_fn,
    )
    if len(rows) < surrogate_min_trials:
        return None
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except Exception:
        return None

    feature_names = [name for name in param_meta]
    categorical = [name for name in feature_names if param_meta[name].is_categorical]
    numeric = [name for name in feature_names if not param_meta[name].is_categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=64,
        max_depth=8,
        random_state=42,
    )
    pipeline = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("rf", model),
        ]
    )
    try:
        pipeline.fit(rows, y, rf__sample_weight=weights)
    except Exception:
        return None

    param_groups: dict[str, list[int]] = {}
    try:
        preprocessor.fit(rows)
        transformed = preprocessor.transform(rows)
        if transformed is None:
            raise ValueError("Surrogate preprocessing returned None")
        transformed_array = cast(Any, transformed)
        feature_indices = list(range(transformed_array.shape[1]))
        start = 0
        for name, transformer, cols in preprocessor.transformers_:
            if transformer == "drop":
                continue
            if name == "cat":
                encoder = transformer
                cats = encoder.categories_
                for col_name, cat_list in zip(cols, cats, strict=False):
                    count = len(cat_list)
                    param_groups[col_name] = list(range(start, start + count))
                    start += count
            elif name == "num":
                for col_name in cols:
                    param_groups[col_name] = [start]
                    start += 1
        if start != len(feature_indices):
            missing = [i for i in feature_indices if i >= start]
            for col_name in numeric:
                if col_name not in param_groups and missing:
                    param_groups[col_name] = [missing.pop(0)]
    except Exception:
        param_groups = {name: [] for name in feature_names}

    return SurrogateModel(
        pipeline=pipeline,
        preprocessor=preprocessor,
        model=model,
        param_groups=param_groups,
    )


def predict_surrogate(
    surrogate: SurrogateModel,
    rows: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    """Predict mean/std from fitted RF surrogate (tree-variance uncertainty)."""
    try:
        features = surrogate.preprocessor.transform(rows)
    except Exception:
        features = surrogate.preprocessor.fit_transform(rows)
    estimators = getattr(surrogate.model, "estimators_", [])
    if not estimators:
        preds = surrogate.model.predict(features)
        return preds.tolist(), [0.0 for _ in preds]
    tree_preds = [est.predict(features) for est in estimators]
    means = [float(sum(vals) / len(vals)) for vals in zip(*tree_preds, strict=False)]
    stds = []
    for idx, mean in enumerate(means):
        variance = sum((tree[idx] - mean) ** 2 for tree in tree_preds) / max(
            1, len(tree_preds) - 1
        )
        stds.append(math.sqrt(variance))
    return means, stds


def compute_interactions(
    surrogate: SurrogateModel,
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Estimate pairwise interaction strengths with SHAP interaction values when available."""
    if not rows:
        return {}
    try:
        import shap
    except Exception:
        return {}
    try:
        features = surrogate.preprocessor.transform(rows)
    except Exception:
        return {}
    max_samples = min(50, features.shape[0])
    sampled_features = features[:max_samples]
    try:
        explainer = shap.TreeExplainer(surrogate.model)
        interactions = explainer.shap_interaction_values(sampled_features)
    except Exception:
        return {}
    if interactions is None:
        return {}
    import numpy as np

    interaction_strength = np.abs(interactions).mean(axis=0)
    result: dict[str, float] = {}
    params = list(surrogate.param_groups.keys())
    for i, param_i in enumerate(params):
        for j in range(i + 1, len(params)):
            param_j = params[j]
            idx_i = surrogate.param_groups.get(param_i, [])
            idx_j = surrogate.param_groups.get(param_j, [])
            if not idx_i or not idx_j:
                continue
            block = interaction_strength[idx_i][:, idx_j]
            strength = float(block.mean())
            result[f"{param_i}|{param_j}"] = strength
    return result


def interaction_strength_for_param(interactions: dict[str, float], param: str) -> float:
    """Average pairwise interaction strength touching one parameter."""
    values = []
    for key, value in interactions.items():
        left, right = key.split("|", maxsplit=1)
        if param in (left, right):
            values.append(value)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def compute_interaction_threshold(interactions: dict[str, float]) -> float:
    """Compute adaptive interaction threshold from pairwise strengths."""
    if not interactions:
        return 0.0
    values = list(interactions.values())
    return max(0.05, (sum(values) / len(values)) * 1.5)


__all__ = [
    "build_param_meta",
    "build_surrogate_data",
    "compute_interactions",
    "compute_interaction_threshold",
    "denormalize_log_value",
    "encode_params",
    "extract_anchor_params",
    "fit_surrogate",
    "interaction_strength_for_param",
    "normalize_log_value",
    "predict_surrogate",
]

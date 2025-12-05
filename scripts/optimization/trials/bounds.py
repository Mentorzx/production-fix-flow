from __future__ import annotations

import math
from typing import Any, Iterable

from pff.utils import logger
from pff.utils.core.file_manager import FileManager

from pff.config import ENSEMBLE_CONFIG_PATH, ENSEMBLE_HPO_CONFIG_PATH
from .config_loader import get_cached_config


def normalize_metric(value: float | None, *, low: float, high: float) -> float:
    """Clamp and scale a metric into [0, 1] interval."""
    if value is None:
        return 0.0
    if math.isnan(value):
        return 0.0
    if high <= low:
        return max(0.0, min(1.0, value))
    normalized = (value - low) / (high - low)
    return max(0.0, min(1.0, normalized))


def blend_scores(scores: Iterable[tuple[float, float]]) -> float:
    """Compute a weighted average from (value, weight) pairs skipping NaN values."""
    total_weight = 0.0
    total = 0.0
    for value, weight in scores:
        if weight <= 0:
            continue
        if math.isnan(value):
            continue
        total += value * weight
        total_weight += weight
    if total_weight == 0.0:
        return 0.0
    return total / total_weight


def get_range(bounds: dict[str, Any], path: list[str], default_low: float, default_high: float) -> tuple[float, float]:
    """Safely read a low/high pair from nested bounds with defaults and inverted-bound guard."""
    node: Any = bounds
    try:
        for key in path:
            if not isinstance(node, dict):
                node = {}
                break
            node = node.get(key, {})
        low = float(node.get("low", default_low)) if isinstance(node, dict) else default_low
        high = float(node.get("high", default_high)) if isinstance(node, dict) else default_high
        if high < low:
            logger.debug(f"Inverted bounds detected for path {path}: low={low}, high={high}; using defaults")
            return default_low, default_high
        return low, high
    except Exception:  # noqa: BLE001
        return default_low, default_high


def load_metric_bounds(file_manager: FileManager | None = None) -> dict[str, Any]:
    """Load metric normalization bounds from config/hpo/ensemble_hpo.yaml."""
    fm = file_manager or FileManager()
    default_bounds = {
        "kge": {"mrr": {"low": 0.15, "high": 0.75}},
        "rules": {
            "confidence": {"low": 0.4, "high": 0.95},
            "recall": {"low": 0.05, "high": 0.5},
            "coverage": {"low": 0.05, "high": 0.5},
            "relation_coverage": {"low": 0.05, "high": 0.60},
            "rules_per_relation": {"low": 1.0, "high": 80.0},
        },
        "learner": {
            "lgbm_auc": {"low": 0.6, "high": 0.99},
            "hybrid_f1": {"low": 0.45, "high": 0.9},
            "xgb_f1": {"low": 0.45, "high": 0.9},
            "xgb_test_auc": {"low": 0.6, "high": 0.99},
            "base_learner_agreement": {"low": 0.4, "high": 0.95},
            "lightgbm_ece": {"low": 0.0, "high": 0.10, "invert": True},
            "lightgbm_entropy": {"low": 0.0, "high": 0.70, "invert": True},
        },
        "ensemble": {
            "ensemble_ece": {"low": 0.0, "high": 0.10, "invert": True},
            "ensemble_entropy": {"low": 0.0, "high": 0.70, "invert": True},
        },
    }
    try:
        ensemble_config = get_cached_config(ENSEMBLE_HPO_CONFIG_PATH, fm)
        return ensemble_config.get("metrics_bounds", default_bounds) or default_bounds
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Failed to load metrics_bounds: {exc}")
        try:
            legacy_config = get_cached_config(ENSEMBLE_CONFIG_PATH, fm)
            return legacy_config.get("metrics_bounds", default_bounds) or default_bounds
        except Exception as legacy_exc:  # noqa: BLE001
            logger.debug(f"Legacy ensemble.yaml load failed for metrics_bounds: {legacy_exc}")
        return default_bounds


def get_rules_coverage_weight(file_manager: FileManager | None = None) -> float:
    """Load the rules coverage weight from ensemble.yaml config, clamped to [0.15, 0.40]."""
    try:
        ensemble_config = get_cached_config(ENSEMBLE_CONFIG_PATH, file_manager)
        balancing = ensemble_config.get("balancing", {})
        rules_config = balancing.get("rules", {})
        raw_weight = float(rules_config.get("coverage_weight", 0.2))
        clamped = max(0.15, min(0.40, raw_weight))
        if clamped != raw_weight:
            logger.debug(f"coverage_weight clamped: {raw_weight} -> {clamped} (allowed: [0.15, 0.40])")
        return clamped
    except Exception as e:
        logger.debug(f"Failed to load coverage_weight from config, using default 0.2: {e}")
        return 0.2


def get_rule_component_weights(file_manager: FileManager | None = None) -> tuple[float, float, float]:
    """Load rule component weights (confidence, recall, coverage) from config."""
    coverage_weight = get_rules_coverage_weight(file_manager)
    try:
        ensemble_config = get_cached_config(ENSEMBLE_CONFIG_PATH, file_manager)
        rules_cfg = ensemble_config.get("balancing", {}).get("rules", {})
        conf_raw = max(0.0, float(rules_cfg.get("confidence_weight", 0.5)))
        recall_raw = max(0.0, float(rules_cfg.get("recall_weight", 0.3)))
    except Exception as e:
        logger.debug(f"Failed to load rule component weights, using defaults: {e}")
        conf_raw, recall_raw = 0.5, 0.3

    remaining = max(0.0, 1.0 - coverage_weight)
    base_sum = conf_raw + recall_raw
    if base_sum <= 0:
        conf_weight = recall_weight = remaining * 0.5
    else:
        scale = remaining / base_sum
        conf_weight = conf_raw * scale
        recall_weight = recall_raw * scale
    return conf_weight, recall_weight, coverage_weight

"""Domain-level utilities for HPO."""

from .bounds import blend_scores, get_range, normalize_metric
from .scoring import (
    ScoreComponents,
    ScoreWeights,
    TimeScaleConfig,
    build_weights_from_settings,
    compute_score,
    rename_metric_keys,
)
from .search_space import SearchSpaceFactory, TuningConfig, TuningConfigBuilder
from .models import KGE_MODEL_ALIASES, KGE_MODEL_DSLFM, resolve_kge_model
from .selection import TrialSelectionEntry, select_best_trials

__all__ = [
    "blend_scores",
    "get_range",
    "normalize_metric",
    "ScoreComponents",
    "ScoreWeights",
    "TimeScaleConfig",
    "build_weights_from_settings",
    "compute_score",
    "rename_metric_keys",
    "SearchSpaceFactory",
    "TuningConfig",
    "TuningConfigBuilder",
    "TrialSelectionEntry",
    "select_best_trials",
    "KGE_MODEL_ALIASES",
    "KGE_MODEL_DSLFM",
    "resolve_kge_model",
]

"""Domain-level utilities for HPO.

Exports are resolved lazily to avoid importing heavy training/search modules at
package import time. This keeps startup faster while preserving the same public API.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

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

_SYMBOL_TO_MODULE = {
    "blend_scores": "pff.domain.hpo.bounds",
    "get_range": "pff.domain.hpo.bounds",
    "normalize_metric": "pff.domain.hpo.bounds",
    "ScoreComponents": "pff.domain.hpo.scoring",
    "ScoreWeights": "pff.domain.hpo.scoring",
    "TimeScaleConfig": "pff.domain.hpo.scoring",
    "build_weights_from_settings": "pff.domain.hpo.scoring",
    "compute_score": "pff.domain.hpo.scoring",
    "rename_metric_keys": "pff.domain.hpo.scoring",
    "SearchSpaceFactory": "pff.domain.hpo.search_space",
    "TuningConfig": "pff.domain.hpo.search_space",
    "TuningConfigBuilder": "pff.domain.hpo.search_space",
    "TrialSelectionEntry": "pff.domain.hpo.selection",
    "select_best_trials": "pff.domain.hpo.selection",
    "KGE_MODEL_ALIASES": "pff.domain.hpo.models",
    "KGE_MODEL_DSLFM": "pff.domain.hpo.models",
    "resolve_kge_model": "pff.domain.hpo.models",
}

if TYPE_CHECKING:
    from pff.domain.hpo.bounds import blend_scores, get_range, normalize_metric
    from pff.domain.hpo.models import (
        KGE_MODEL_ALIASES,
        KGE_MODEL_DSLFM,
        resolve_kge_model,
    )
    from pff.domain.hpo.scoring import (
        ScoreComponents,
        ScoreWeights,
        TimeScaleConfig,
        build_weights_from_settings,
        compute_score,
        rename_metric_keys,
    )
    from pff.domain.hpo.search_space import (
        SearchSpaceFactory,
        TuningConfig,
        TuningConfigBuilder,
    )
    from pff.domain.hpo.selection import TrialSelectionEntry, select_best_trials


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

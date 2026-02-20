"""Core data models for the Search Space Advisor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParamRecommendation:
    """Single parameter recommendation emitted by the advisor."""

    param_name: str
    current_space: dict[str, Any]
    attempts_summary: dict[str, Any]
    best_region: dict[str, Any]
    importance: float
    action: str
    recommendation: dict[str, Any]
    rationale: str
    confidence: str
    uncertainty: float
    bootstrap_support: float | None = None
    interaction_strength: float | None = None
    surrogate_bounds: dict[str, float] | None = None


@dataclass
class AdvisorResult:
    """Container for advisor recommendations and metadata."""

    recommendations: list[ParamRecommendation]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialSummary:
    """Lightweight trial summary used by advisor internals."""

    number: int
    value: float
    params: dict[str, Any]
    raw_value: float | None = None
    state: str = "COMPLETE"
    intermediate_values: dict[int, float] | None = None


@dataclass(frozen=True)
class ParamMeta:
    """Normalized parameter metadata derived from search space specs."""

    name: str
    param_type: str
    is_categorical: bool
    is_log: bool
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None


@dataclass
class TrustState:
    """State tracked to adapt directional recommendations over time."""

    upper_success: int = 0
    lower_success: int = 0
    failure: int = 0
    best_value: float | None = None
    best_params: dict[str, Any] = field(default_factory=dict)
    last_trial: int | None = None


@dataclass
class SurrogateModel:
    """Fitted surrogate model bundle and feature-group map."""

    pipeline: Any
    preprocessor: Any
    model: Any
    param_groups: dict[str, list[int]]


__all__ = [
    "AdvisorResult",
    "ParamMeta",
    "ParamRecommendation",
    "SurrogateModel",
    "TrialSummary",
    "TrustState",
]

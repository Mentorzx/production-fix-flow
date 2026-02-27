"""Runtime configuration and adaptive-control helpers for Search Space Advisor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .parsing import normalize_direction


@dataclass(frozen=True)
class RuntimeSettings:
    """Resolved runtime knobs from base config and per-call overrides."""

    norm_direction: str
    study_key: str
    effective_cfg: dict[str, Any]
    effective_min_trials_any: int
    effective_min_trials_aggressive: int
    conditional_scope_threshold: float
    enable_surrogate: bool
    enable_interactions: bool
    disable_internal_importances: bool
    adaptive_perf_enabled: bool
    adaptive_perf_ms_threshold: float
    adaptive_perf_validation_lb_min: float
    adaptive_perf_cooldown_calls: int
    rust_spearman_min_len: int
    self_audit_period_trials: int
    self_audit_min_prefix: int
    self_audit_min_suffix: int
    self_audit_max_prefixes: int
    self_audit_min_group_total: int
    self_audit_wilson_block: float
    self_audit_min_match_points: int
    categorical_min_topk_samples: int
    categorical_min_topk_unique: int
    categorical_min_effective_categories: float
    configured_distribution_conflicts: list[Any]
    configured_coverage_ratio: Any
    explicit_surrogate: bool
    explicit_interactions: bool
    explicit_internal_importances: bool


@dataclass(frozen=True)
class AdaptiveControlDecision:
    """Adaptive control decision over costly advisor subsystems."""

    enable_surrogate: bool
    enable_interactions: bool
    disable_internal_importances: bool
    adaptive_decision: str


def build_adaptive_performance_metadata(
    *,
    enabled: bool,
    decision: str,
    threshold_ms: float,
    validation_lb_min: float,
    cooldown_calls: int,
    cooldown_before: int,
    cooldown_after: int,
    degraded_count_before: int,
    degraded_count_after: int,
    last_compute_ms_before: float,
    last_validation_lb_before: float,
) -> dict[str, Any]:
    """Build stable metadata payload for adaptive control diagnostics."""
    return {
        "enabled": bool(enabled),
        "decision": str(decision),
        "threshold_ms": float(threshold_ms),
        "validation_lb_min": float(validation_lb_min),
        "cooldown_calls": int(cooldown_calls),
        "cooldown_before": int(cooldown_before),
        "cooldown_after": int(cooldown_after),
        "degraded_count_before": int(degraded_count_before),
        "degraded_count_after": int(degraded_count_after),
        "last_compute_ms_before": float(last_compute_ms_before),
        "last_validation_lb_before": float(last_validation_lb_before),
    }


def update_adaptive_performance_state(
    *,
    set_state_fn: Any,
    study_key: str,
    enabled: bool,
    compute_ms: float,
    validation_lb: float,
    threshold_ms: float,
    validation_lb_min: float,
    cooldown_calls: int,
    cooldown_before: int,
    degraded_count_before: int,
) -> tuple[int, int]:
    """Update adaptive state counters and persist them through provided callback."""
    import time

    if not enabled:
        set_state_fn(
            study_key,
            {
                "degraded_count": int(degraded_count_before),
                "cooldown_remaining": int(max(0, cooldown_before)),
                "last_compute_ms": float(compute_ms),
                "last_validation_lb": float(validation_lb),
                "updated_at_ns": time.time_ns(),
                "enabled": False,
            },
        )
        return int(degraded_count_before), int(max(0, cooldown_before))

    is_degraded = float(compute_ms) >= float(threshold_ms) and float(
        validation_lb
    ) >= float(validation_lb_min)
    cooldown_after = int(max(0, cooldown_before))
    if is_degraded:
        degraded_count_after = int(degraded_count_before) + 1
        cooldown_after = max(cooldown_after, int(max(0, cooldown_calls)))
    else:
        cooldown_after = max(0, cooldown_after - 1)
        if cooldown_after > 0:
            degraded_count_after = max(1, int(degraded_count_before))
        else:
            degraded_count_after = max(0, int(degraded_count_before) - 1)
    set_state_fn(
        study_key,
        {
            "degraded_count": int(degraded_count_after),
            "cooldown_remaining": int(cooldown_after),
            "last_compute_ms": float(compute_ms),
            "last_validation_lb": float(validation_lb),
            "updated_at_ns": time.time_ns(),
            "enabled": True,
        },
    )
    return int(degraded_count_after), int(cooldown_after)


def resolve_runtime_settings(
    *,
    base_config: dict[str, Any],
    advisor_config: dict[str, Any] | None,
    direction: str,
    study_name: str,
    min_trials_any_default: int,
    min_trials_aggressive_default: int,
    top_k_fraction_default: float,
    top_k_min_default: int,
    rust_spearman_min_len_default: int,
    self_audit_period_trials_default: int,
    self_audit_min_prefix_default: int,
    self_audit_min_suffix_default: int,
    self_audit_max_prefixes_default: int,
    self_audit_min_group_total_default: int,
    self_audit_wilson_block_default: float,
    self_audit_min_match_points_default: int,
) -> RuntimeSettings:
    """Resolve per-call runtime settings from defaults and call overrides."""
    effective_cfg = {**base_config, **(advisor_config or {})}
    cfg_distribution_conflicts = effective_cfg.get("distribution_conflicts", [])
    configured_distribution_conflicts = (
        list(cfg_distribution_conflicts)
        if isinstance(cfg_distribution_conflicts, list)
        else []
    )

    explicit_surrogate = (
        isinstance(advisor_config, dict) and "enable_surrogate" in advisor_config
    )
    explicit_interactions = (
        isinstance(advisor_config, dict) and "enable_interactions" in advisor_config
    )
    explicit_internal_importances = (
        isinstance(advisor_config, dict)
        and "disable_internal_importances" in advisor_config
    )

    return RuntimeSettings(
        norm_direction=normalize_direction(direction),
        study_key=study_name or "__default__",
        effective_cfg=effective_cfg,
        effective_min_trials_any=int(
            effective_cfg.get("min_trials_any", min_trials_any_default)
        ),
        effective_min_trials_aggressive=int(
            effective_cfg.get("min_trials_aggressive", min_trials_aggressive_default)
        ),
        conditional_scope_threshold=float(
            effective_cfg.get("conditional_scope_threshold", 0.9)
        ),
        enable_surrogate=bool(effective_cfg.get("enable_surrogate", True)),
        enable_interactions=bool(effective_cfg.get("enable_interactions", True)),
        disable_internal_importances=bool(
            effective_cfg.get("disable_internal_importances", False)
        ),
        adaptive_perf_enabled=bool(effective_cfg.get("adaptive_perf_enabled", True)),
        adaptive_perf_ms_threshold=float(
            effective_cfg.get("adaptive_perf_ms_threshold", 1200.0)
        ),
        adaptive_perf_validation_lb_min=float(
            effective_cfg.get("adaptive_perf_validation_lb_min", 0.75)
        ),
        adaptive_perf_cooldown_calls=int(
            effective_cfg.get("adaptive_perf_cooldown_calls", 2)
        ),
        rust_spearman_min_len=int(
            effective_cfg.get("rust_spearman_min_len", rust_spearman_min_len_default)
        ),
        self_audit_period_trials=int(
            effective_cfg.get(
                "self_audit_period_trials", self_audit_period_trials_default
            )
        ),
        self_audit_min_prefix=int(
            effective_cfg.get("self_audit_min_prefix", self_audit_min_prefix_default)
        ),
        self_audit_min_suffix=int(
            effective_cfg.get("self_audit_min_suffix", self_audit_min_suffix_default)
        ),
        self_audit_max_prefixes=int(
            effective_cfg.get(
                "self_audit_max_prefixes", self_audit_max_prefixes_default
            )
        ),
        self_audit_min_group_total=int(
            effective_cfg.get(
                "self_audit_min_group_total", self_audit_min_group_total_default
            )
        ),
        self_audit_wilson_block=float(
            effective_cfg.get(
                "self_audit_wilson_block", self_audit_wilson_block_default
            )
        ),
        self_audit_min_match_points=int(
            effective_cfg.get(
                "self_audit_min_match_points", self_audit_min_match_points_default
            )
        ),
        categorical_min_topk_samples=int(
            effective_cfg.get("categorical_min_topk_samples", 3)
        ),
        categorical_min_topk_unique=int(
            effective_cfg.get("categorical_min_topk_unique", 1)
        ),
        categorical_min_effective_categories=float(
            effective_cfg.get("categorical_min_effective_categories", 1.0)
        ),
        configured_distribution_conflicts=configured_distribution_conflicts,
        configured_coverage_ratio=effective_cfg.get("search_space_coverage_ratio"),
        explicit_surrogate=explicit_surrogate,
        explicit_interactions=explicit_interactions,
        explicit_internal_importances=explicit_internal_importances,
    )


def resolve_adaptive_controls(
    *,
    adaptive_perf_enabled: bool,
    adaptive_perf_ms_threshold: float,
    adaptive_perf_validation_lb_min: float,
    degraded_count_before: int,
    cooldown_before: int,
    last_compute_ms_before: float,
    last_validation_lb_before: float,
    enable_surrogate: bool,
    enable_interactions: bool,
    disable_internal_importances: bool,
    explicit_surrogate: bool,
    explicit_interactions: bool,
    explicit_internal_importances: bool,
) -> AdaptiveControlDecision:
    """Decide adaptive control toggles for expensive advisor components."""
    adaptive_decision = "none"
    resolved_enable_surrogate = bool(enable_surrogate)
    resolved_enable_interactions = bool(enable_interactions)
    resolved_disable_internal_importances = bool(disable_internal_importances)

    if adaptive_perf_enabled:
        should_degrade_controls = cooldown_before > 0 or (
            last_compute_ms_before >= adaptive_perf_ms_threshold
            and last_validation_lb_before >= adaptive_perf_validation_lb_min
        )
        if should_degrade_controls:
            if resolved_enable_interactions and not explicit_interactions:
                resolved_enable_interactions = False
                adaptive_decision = "disable_interactions"
            if (
                degraded_count_before >= 2
                and resolved_enable_surrogate
                and not explicit_surrogate
            ):
                resolved_enable_surrogate = False
                adaptive_decision = "disable_surrogate"
            if (
                degraded_count_before >= 3
                and not resolved_disable_internal_importances
                and not explicit_internal_importances
            ):
                resolved_disable_internal_importances = True
                if adaptive_decision == "none":
                    adaptive_decision = "disable_internal_importances"

    return AdaptiveControlDecision(
        enable_surrogate=resolved_enable_surrogate,
        enable_interactions=resolved_enable_interactions,
        disable_internal_importances=resolved_disable_internal_importances,
        adaptive_decision=adaptive_decision,
    )


__all__ = [
    "AdaptiveControlDecision",
    "RuntimeSettings",
    "build_adaptive_performance_metadata",
    "resolve_adaptive_controls",
    "resolve_runtime_settings",
    "update_adaptive_performance_state",
]

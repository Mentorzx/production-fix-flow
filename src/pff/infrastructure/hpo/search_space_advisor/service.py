"""Search Space Advisor — computes actionable recommendations for HPO search space adjustments.

Given the current search space, completed trial data, and param importances,
this module produces per-parameter recommendations (expand/reduce/fix/alter distribution)
with evidence, confidence, and rationale.

Supports single-objective and multi-objective studies.
"""

from __future__ import annotations

import time
from typing import Any


from pff.infrastructure.hpo.config_loader import load_optimization_config
from pff.shared import logger

from .analysis_categorical import categorical_counts as _categorical_counts_shared
from .analysis_pipeline import (
    analyze_categorical_param as _analyze_categorical_param_shared,
    analyze_numeric_param as _analyze_numeric_param_shared,
    apply_direction as _apply_direction_shared,
    bootstrap_support_for_param as _bootstrap_support_for_param_shared,
    compute_spearman as _compute_spearman_shared,
    with_bootstrap_confidence as _with_bootstrap_confidence_shared,
)
from .advisor_runtime import (
    build_adaptive_performance_metadata as _build_adaptive_performance_metadata_shared,
    resolve_adaptive_controls as _resolve_adaptive_controls_shared,
    resolve_runtime_settings as _resolve_runtime_settings_shared,
    update_adaptive_performance_state as _update_adaptive_performance_state_shared,
)
from .cache import (
    AdvisorCache,
    AdvisorCacheSpec,
    build_objective_schema_hash,
    build_search_space_hash,
)
from .importance import (
    compute_internal_importances as _compute_internal_importances_shared,
    resolve_importances as _resolve_importances_shared,
)
from .multiobjective import build_multiobjective_projection
from .parsing import (
    is_cost_sensitive_param as _is_cost_sensitive_param_shared,
    is_log_scale_candidate as _is_log_scale_candidate_shared,
    normalize_trial_state as _normalize_trial_state_shared,
    parse_search_space_entry as _parse_search_space_entry_shared,
)
from .policy import (
    build_policy_metadata as _build_policy_metadata_shared,
    policy_stub as _policy_stub_shared,
)
from .recommendations import (
    build_dataset_heuristic_recommendations as _build_dataset_heuristic_recommendations_shared,
    compute_search_space_coverage,
    make_recommendation as _make_recommendation_shared,
    resolve_recommendation_scope,
)
from .reliability import (
    compute_reliability_summary as _compute_reliability_summary_shared,
    confidence_from_support as _confidence_from_support_shared,
    wilson_lower_bound as _wilson_lower_bound_shared,
)
from .runtime_state import AdvisorRuntimeStateStore
from .self_audit import (
    apply_self_audit_blocks as _apply_self_audit_blocks_shared,
    audit_prefix_sizes as _audit_prefix_sizes_shared,
    is_directional_action as _is_directional_action_shared,
    match_directional_suffix_trend as _match_directional_suffix_trend_shared,
)
from .self_audit_runner import (
    build_self_audit_summary as _build_self_audit_summary_runner_shared,
    resolve_self_audit_summary as _resolve_self_audit_summary_runner_shared,
)
from .statistics import (
    estimate_uncertainty as _estimate_uncertainty_shared,
    numeric_stats as _numeric_stats_shared,
    reservoir_sample as _reservoir_sample_shared,
    select_top_k as _select_top_k_shared,
)
from .surrogate import (
    build_param_meta as _build_param_meta_shared,
    build_surrogate_data as _build_surrogate_data_shared,
    compute_interaction_threshold,
    compute_interactions as _compute_interactions_shared,
    extract_anchor_params as _extract_anchor_params_shared,
    fit_surrogate as _fit_surrogate_shared,
    interaction_strength_for_param as _interaction_strength_for_param_shared,
    normalize_log_value as _normalize_log_value_shared,
)
from .trial_projection import (
    build_trial_summaries as _build_trial_summaries_shared,
)
from .trust import update_trust_bucket as _update_trust_bucket_shared
from .validation import recommendation_to_dict as _recommendation_to_dict_shared

try:
    import numpy as _np
except Exception:  # pragma: no cover - optional perf dependency at runtime
    _np = None

try:
    from pff_rust import fast_spearman_corr as _rust_fast_spearman_corr
except Exception:  # pragma: no cover - fallback to pure Python path
    _rust_fast_spearman_corr = None

ADVISOR_VERSION = "2.3.0"

_MIN_TRIALS_FOR_AGGRESSIVE = 20
_MIN_TRIALS_FOR_ANY = 5
_TOP_K_FRACTION = 0.25
_TOP_K_MIN = 3
_EDGE_THRESHOLD = 0.15
_CONCENTRATION_CV_THRESHOLD = 0.15
_CATEGORICAL_DOMINANCE_THRESHOLD = 0.6
_LOW_IMPORTANCE_THRESHOLD = 0.05
_BOOTSTRAP_SAMPLES = 50
_CONFIDENCE_HIGH = 0.9
_CONFIDENCE_MEDIUM = 0.7
_TRUST_SUCCESS = 3
_TRUST_FAILURE = 5
_UCB_STD_MULT = 1.96
_RESERVOIR_SIZE = 60
_SURROGATE_MIN_TRIALS = 8
_MAX_EXPANSION_FACTOR = 10.0
_CORRELATION_GATE_ABS = 0.15
_MIN_POINTS_FOR_CORRELATION = 8
_CONFIDENCE_SUPPORT_PRIOR_N = 20.0
_CONFIDENCE_SUPPORT_PRIOR_P = 0.5
_COST_SENSITIVE_UPPER_RHO = 0.25
_SELF_AUDIT_PERIOD_TRIALS = 10
_SELF_AUDIT_MIN_PREFIX = 8
_SELF_AUDIT_MIN_SUFFIX = 5
_SELF_AUDIT_MAX_PREFIXES = 6
_SELF_AUDIT_MIN_GROUP_TOTAL = 8
_SELF_AUDIT_WILSON_BLOCK = 0.35
_SELF_AUDIT_MIN_MATCH_POINTS = 5
_RUST_SPEARMAN_MIN_LEN = 512


class SearchSpaceAdvisor:
    """Computes search space recommendations from trial data.

    Usage:
        advisor = SearchSpaceAdvisor(config_thresholds={...})
        result = advisor.advise(
            search_space={"lr": {"name": "FloatDistribution", ...}, ...},
            trials_data=[{"id": 1, "value": 0.85, "params": {...}}, ...],
            importances={"lr": 0.42, ...},
            direction="maximize",
            study_name="my_study",
        )
    """

    def __init__(self, config_thresholds: dict[str, Any] | None = None) -> None:
        cfg_file: dict[str, Any] = {}
        try:
            optimization_cfg = load_optimization_config()
            if isinstance(optimization_cfg, dict):
                raw_advisor_cfg = optimization_cfg.get("search_space_advisor", {})
                if isinstance(raw_advisor_cfg, dict):
                    cfg_file = raw_advisor_cfg
        except Exception:
            cfg_file = {}
        cfg: dict[str, Any] = {**cfg_file, **(config_thresholds or {})}
        self.min_trials_aggressive = int(
            cfg.get("min_trials_aggressive", _MIN_TRIALS_FOR_AGGRESSIVE)
        )
        self.min_trials_any = int(cfg.get("min_trials_any", _MIN_TRIALS_FOR_ANY))
        self.top_k_fraction = float(cfg.get("top_k_fraction", _TOP_K_FRACTION))
        self.top_k_min = int(cfg.get("top_k_min", _TOP_K_MIN))
        self._advisor_cfg = dict(cfg)
        self._advice_cache = AdvisorCache(
            max_memory_items=int(cfg.get("cache_max_items", 128)),
            ttl_seconds=int(cfg.get("cache_ttl_seconds", 900)),
            enable_persistent_l2=bool(cfg.get("persistent_cache_enabled", True)),
        )
        self._runtime_state = AdvisorRuntimeStateStore(
            max_memory_items=int(cfg.get("runtime_state_max_items", 1024))
        )
        self.rust_spearman_min_len = int(cfg.get("rust_spearman_min_len", _RUST_SPEARMAN_MIN_LEN))
        self.self_audit_period_trials = int(
            cfg.get("self_audit_period_trials", _SELF_AUDIT_PERIOD_TRIALS)
        )
        self.self_audit_min_prefix = int(cfg.get("self_audit_min_prefix", _SELF_AUDIT_MIN_PREFIX))
        self.self_audit_min_suffix = int(cfg.get("self_audit_min_suffix", _SELF_AUDIT_MIN_SUFFIX))
        self.self_audit_max_prefixes = int(
            cfg.get("self_audit_max_prefixes", _SELF_AUDIT_MAX_PREFIXES)
        )
        self.self_audit_min_group_total = int(
            cfg.get("self_audit_min_group_total", _SELF_AUDIT_MIN_GROUP_TOTAL)
        )
        self.self_audit_wilson_block = float(
            cfg.get("self_audit_wilson_block", _SELF_AUDIT_WILSON_BLOCK)
        )
        self.self_audit_min_match_points = int(
            cfg.get("self_audit_min_match_points", _SELF_AUDIT_MIN_MATCH_POINTS)
        )

    def advise(
        self,
        search_space: dict[str, Any],
        trials_data: list[dict[str, Any]],
        importances: dict[str, float],
        direction: str = "maximize",
        study_name: str = "",
        dataset_fingerprint: str | None = None,
        dataset_profile: dict[str, Any] | None = None,
        study: Any | None = None,
        objective_directions: list[str] | None = None,
        advisor_config: dict[str, Any] | None = None,
        force_recompute: bool = False,
        enable_bootstrap: bool = True,
        enable_self_audit: bool = True,
    ) -> dict[str, Any]:
        """Produce search space recommendations.

        Returns a JSON-serializable dict with keys:
            recommendations, metadata
        """
        _ = study  # Reserved for future Optuna-storage integrations.
        t0 = time.monotonic()
        runtime_settings = _resolve_runtime_settings_shared(
            base_config=self._advisor_cfg,
            advisor_config=advisor_config,
            direction=direction,
            study_name=study_name,
            min_trials_any_default=self.min_trials_any,
            min_trials_aggressive_default=self.min_trials_aggressive,
            top_k_fraction_default=self.top_k_fraction,
            top_k_min_default=self.top_k_min,
            rust_spearman_min_len_default=self.rust_spearman_min_len,
            self_audit_period_trials_default=self.self_audit_period_trials,
            self_audit_min_prefix_default=self.self_audit_min_prefix,
            self_audit_min_suffix_default=self.self_audit_min_suffix,
            self_audit_max_prefixes_default=self.self_audit_max_prefixes,
            self_audit_min_group_total_default=self.self_audit_min_group_total,
            self_audit_wilson_block_default=self.self_audit_wilson_block,
            self_audit_min_match_points_default=self.self_audit_min_match_points,
        )
        norm_direction = runtime_settings.norm_direction
        study_key = runtime_settings.study_key
        effective_cfg = runtime_settings.effective_cfg
        effective_min_trials_any = runtime_settings.effective_min_trials_any
        effective_min_trials_aggressive = runtime_settings.effective_min_trials_aggressive
        conditional_scope_threshold = runtime_settings.conditional_scope_threshold
        enable_surrogate = runtime_settings.enable_surrogate
        enable_interactions = runtime_settings.enable_interactions
        disable_internal_importances = runtime_settings.disable_internal_importances
        adaptive_perf_enabled = runtime_settings.adaptive_perf_enabled
        adaptive_perf_ms_threshold = runtime_settings.adaptive_perf_ms_threshold
        adaptive_perf_validation_lb_min = runtime_settings.adaptive_perf_validation_lb_min
        adaptive_perf_cooldown_calls = runtime_settings.adaptive_perf_cooldown_calls
        rust_spearman_min_len = runtime_settings.rust_spearman_min_len

        self.rust_spearman_min_len = runtime_settings.rust_spearman_min_len
        self.self_audit_period_trials = runtime_settings.self_audit_period_trials
        self.self_audit_min_prefix = runtime_settings.self_audit_min_prefix
        self.self_audit_min_suffix = runtime_settings.self_audit_min_suffix
        self.self_audit_max_prefixes = runtime_settings.self_audit_max_prefixes
        self.self_audit_min_group_total = runtime_settings.self_audit_min_group_total
        self.self_audit_wilson_block = runtime_settings.self_audit_wilson_block
        self.self_audit_min_match_points = runtime_settings.self_audit_min_match_points

        def _spearman_for_runtime(
            values: list[float],
            scores: list[float],
            *,
            min_points: int = _MIN_POINTS_FOR_CORRELATION,
        ) -> float | None:
            return _compute_spearman_shared(
                values,
                scores,
                min_points=min_points,
                rust_fast_fn=_rust_fast_spearman_corr,
                rust_min_len=rust_spearman_min_len,
                np_module=_np,
            )

        perf_state_before = self._runtime_state.get_adaptive_state(study_key)
        degraded_count_before = int(perf_state_before.get("degraded_count", 0))
        cooldown_before = int(perf_state_before.get("cooldown_remaining", 0))
        last_compute_ms_before = float(perf_state_before.get("last_compute_ms", 0.0) or 0.0)
        last_validation_lb_before = float(perf_state_before.get("last_validation_lb", 0.0) or 0.0)
        adaptive_controls = _resolve_adaptive_controls_shared(
            adaptive_perf_enabled=adaptive_perf_enabled,
            adaptive_perf_ms_threshold=adaptive_perf_ms_threshold,
            adaptive_perf_validation_lb_min=adaptive_perf_validation_lb_min,
            degraded_count_before=degraded_count_before,
            cooldown_before=cooldown_before,
            last_compute_ms_before=last_compute_ms_before,
            last_validation_lb_before=last_validation_lb_before,
            enable_surrogate=enable_surrogate,
            enable_interactions=enable_interactions,
            disable_internal_importances=disable_internal_importances,
            explicit_surrogate=runtime_settings.explicit_surrogate,
            explicit_interactions=runtime_settings.explicit_interactions,
            explicit_internal_importances=runtime_settings.explicit_internal_importances,
        )
        enable_surrogate = adaptive_controls.enable_surrogate
        enable_interactions = adaptive_controls.enable_interactions
        disable_internal_importances = adaptive_controls.disable_internal_importances
        adaptive_decision = adaptive_controls.adaptive_decision
        configured_distribution_conflicts = runtime_settings.configured_distribution_conflicts
        configured_coverage_ratio = runtime_settings.configured_coverage_ratio
        categorical_min_topk_samples = runtime_settings.categorical_min_topk_samples
        categorical_min_topk_unique = runtime_settings.categorical_min_topk_unique
        categorical_min_effective_categories = runtime_settings.categorical_min_effective_categories
        policy_metadata = _build_policy_metadata_shared(
            advisor_version=ADVISOR_VERSION,
            direction=norm_direction,
            effective_cfg=effective_cfg,
            decision_thresholds={
                "edge_threshold": _EDGE_THRESHOLD,
                "concentration_cv_threshold": _CONCENTRATION_CV_THRESHOLD,
                "categorical_dominance_threshold": _CATEGORICAL_DOMINANCE_THRESHOLD,
                "categorical_min_topk_samples": int(categorical_min_topk_samples),
                "categorical_min_topk_unique": int(categorical_min_topk_unique),
                "categorical_min_effective_categories": float(
                    categorical_min_effective_categories
                ),
                "low_importance_threshold": _LOW_IMPORTANCE_THRESHOLD,
                "bootstrap_samples": _BOOTSTRAP_SAMPLES,
                "ucb_std_mult": _UCB_STD_MULT,
                "surrogate_min_trials": _SURROGATE_MIN_TRIALS,
                "self_audit_wilson_block": float(self.self_audit_wilson_block),
                "self_audit_min_group_total": int(self.self_audit_min_group_total),
            },
        )
        projection = build_multiobjective_projection(
            trials_data,
            fallback_direction=norm_direction,
            objective_directions=objective_directions,
        )
        projected_scores = projection.scores

        all_trials, completed = _build_trial_summaries_shared(
            trials_data,
            projected_scores,
            normalize_trial_state_fn=_normalize_trial_state_shared,
        )

        n_trials = len(completed)
        last_trial = max((t.number for t in completed), default=-1)
        resolved_dataset_fingerprint = dataset_fingerprint or "none"
        cache_spec = AdvisorCacheSpec(
            study_name=study_key,
            dataset_fingerprint=resolved_dataset_fingerprint,
            direction=norm_direction,
            advisor_version=ADVISOR_VERSION,
            last_trial=last_trial,
            search_space_hash=build_search_space_hash(search_space),
            objective_schema_hash=build_objective_schema_hash(
                objective_directions or projection.metadata.get("objective_directions")
            ),
        )
        cache_get = self._advice_cache.get_with_status(cache_spec)
        cached = cache_get.payload
        cache_layer = cache_get.layer_hit
        cache_status = cache_get.status
        cache_error_code = cache_get.error_code
        if isinstance(cached, dict) and not force_recompute:
            cached_recommendations = cached.get("recommendations")
            cached_metadata = cached.get("metadata", {})
            cached_insufficient = bool(cached_metadata.get("insufficient_evidence"))
            cached_is_empty = (
                isinstance(cached_recommendations, list) and len(cached_recommendations) == 0
            )
            if not (
                cached_is_empty and not cached_insufficient and n_trials >= effective_min_trials_any
            ):
                cached.setdefault("metadata", {})
                cached["metadata"]["cache_hit"] = True
                cached["metadata"]["cache_layer_hit"] = cache_layer
                cached["metadata"]["cache_status"] = cache_status
                cached["metadata"]["cache_error_code"] = cache_error_code
                acceleration = cached["metadata"].get("acceleration")
                if isinstance(acceleration, dict):
                    acceleration["rust_spearman_available"] = bool(
                        callable(_rust_fast_spearman_corr)
                    )
                    acceleration["rust_spearman_min_len"] = int(rust_spearman_min_len)
                cached["metadata"]["adaptive_performance"] = (
                    _build_adaptive_performance_metadata_shared(
                        enabled=adaptive_perf_enabled,
                        decision="cache_hit",
                        threshold_ms=adaptive_perf_ms_threshold,
                        validation_lb_min=adaptive_perf_validation_lb_min,
                        cooldown_calls=adaptive_perf_cooldown_calls,
                        cooldown_before=cooldown_before,
                        cooldown_after=cooldown_before,
                        degraded_count_before=degraded_count_before,
                        degraded_count_after=degraded_count_before,
                        last_compute_ms_before=last_compute_ms_before,
                        last_validation_lb_before=last_validation_lb_before,
                    )
                )
                cached["metadata"].setdefault("policy_version", policy_metadata["policy_version"])
                cached["metadata"].setdefault("policy_hash", policy_metadata["policy_hash"])
                cached["metadata"].setdefault(
                    "policy_thresholds", policy_metadata["policy_thresholds"]
                )
                if isinstance(cached_recommendations, list):
                    for recommendation in cached_recommendations:
                        if not isinstance(recommendation, dict):
                            continue
                        recommendation.setdefault(
                            "policy",
                            _policy_stub_shared(
                                version=policy_metadata["policy_version"],
                                policy_hash=policy_metadata["policy_hash"],
                            ),
                        )
                return cached

        recommendations: list[dict[str, Any]] = []

        if n_trials < effective_min_trials_any:
            recommendations = _build_dataset_heuristic_recommendations_shared(
                search_space=search_space,
                dataset_profile=dataset_profile,
                parse_search_space_entry_fn=_parse_search_space_entry_shared,
                make_recommendation_fn=lambda **kwargs: _make_recommendation_shared(
                    **kwargs,
                    estimate_uncertainty_fn=lambda n_param_trials, top_k_count: _estimate_uncertainty_shared(
                        n_trials=n_param_trials,
                        top_k_count=top_k_count,
                    ),
                    numeric_stats_fn=_numeric_stats_shared,
                    reservoir_sample_fn=lambda values, k, seed: _reservoir_sample_shared(
                        values,
                        k=k,
                        seed=seed,
                    ),
                    categorical_counts_fn=_categorical_counts_shared,
                    reservoir_size=_RESERVOIR_SIZE,
                ),
                recommendation_to_dict_fn=lambda rec: _recommendation_to_dict_shared(
                    rec,
                    confidence_prior_n=_CONFIDENCE_SUPPORT_PRIOR_N,
                    confidence_prior_p=_CONFIDENCE_SUPPORT_PRIOR_P,
                    max_expansion_factor=_MAX_EXPANSION_FACTOR,
                ),
            )
            coverage_ratio, missing_params = compute_search_space_coverage(
                search_space=search_space,
                observed_param_counts={},
                total_trials=n_trials,
            )
            self_audit_summary = (
                {
                    "enabled": False,
                    "ran": False,
                    "reason": "disabled",
                    "period_trials": self.self_audit_period_trials,
                    "source_last_trial": last_trial,
                }
                if not enable_self_audit
                else {
                    "enabled": True,
                    "ran": False,
                    "reason": "insufficient_trials_for_self_audit",
                    "period_trials": self.self_audit_period_trials,
                    "source_last_trial": last_trial,
                }
            )
            result: dict[str, Any] = {
                "recommendations": recommendations,
                "metadata": {
                    "study_name": study_name,
                    "dataset_fingerprint": dataset_fingerprint,
                    "dataset_profile": dataset_profile,
                    "last_trial": last_trial,
                    "advisor_version": ADVISOR_VERSION,
                    "n_completed_trials": n_trials,
                    "cache_hit": False,
                    "cache_layer_hit": "none",
                    "cache_status": cache_status,
                    "cache_error_code": cache_error_code,
                    "forced_recompute": force_recompute,
                    "compute_time_ms": round((time.monotonic() - t0) * 1000, 2),
                    "insufficient_evidence": True,
                    "min_trials_required": effective_min_trials_any,
                    "heuristics_used": bool(recommendations),
                    "reliability_summary": _compute_reliability_summary_shared(recommendations),
                    "self_audit": self_audit_summary,
                    "importance_source": "none",
                    "importance_quality": 0.0,
                    "search_space_coverage_ratio": (
                        float(configured_coverage_ratio)
                        if isinstance(configured_coverage_ratio, (int, float))
                        else coverage_ratio
                    ),
                    "missing_params": missing_params,
                    "distribution_conflicts": configured_distribution_conflicts,
                    "multiobjective_mode": projection.metadata.get(
                        "multiobjective_mode",
                        "single_objective",
                    ),
                    "acceleration": {
                        "rust_spearman_available": bool(callable(_rust_fast_spearman_corr)),
                        "rust_spearman_min_len": int(rust_spearman_min_len),
                        "surrogate_enabled": enable_surrogate,
                        "interactions_enabled": enable_interactions,
                        "internal_importances_disabled": disable_internal_importances,
                    },
                    "policy_version": policy_metadata["policy_version"],
                    "policy_hash": policy_metadata["policy_hash"],
                    "policy_thresholds": policy_metadata["policy_thresholds"],
                },
            }
            for recommendation in recommendations:
                if not isinstance(recommendation, dict):
                    continue
                recommendation["policy"] = _policy_stub_shared(
                    version=policy_metadata["policy_version"],
                    policy_hash=policy_metadata["policy_hash"],
                )
            validation_lb = float(
                result["metadata"]["reliability_summary"].get("validation_pass_wilson_lb", 0.0)
            )
            degraded_count_after, cooldown_after = _update_adaptive_performance_state_shared(
                set_state_fn=self._runtime_state.set_adaptive_state,
                study_key=study_key,
                enabled=adaptive_perf_enabled,
                compute_ms=float(result["metadata"]["compute_time_ms"]),
                validation_lb=validation_lb,
                threshold_ms=adaptive_perf_ms_threshold,
                validation_lb_min=adaptive_perf_validation_lb_min,
                cooldown_calls=adaptive_perf_cooldown_calls,
                cooldown_before=cooldown_before,
                degraded_count_before=degraded_count_before,
            )
            result["metadata"]["adaptive_performance"] = _build_adaptive_performance_metadata_shared(
                enabled=adaptive_perf_enabled,
                decision=adaptive_decision,
                threshold_ms=adaptive_perf_ms_threshold,
                validation_lb_min=adaptive_perf_validation_lb_min,
                cooldown_calls=adaptive_perf_cooldown_calls,
                cooldown_before=cooldown_before,
                cooldown_after=cooldown_after,
                degraded_count_before=degraded_count_before,
                degraded_count_after=degraded_count_after,
                last_compute_ms_before=last_compute_ms_before,
                last_validation_lb_before=last_validation_lb_before,
            )
            cache_write = self._advice_cache.set_with_status(cache_spec, result)
            result["metadata"]["cache_write_status"] = cache_write.status
            result["metadata"]["cache_write_error_code"] = cache_write.error_code
            return result

        effective_top_k_fraction = float(effective_cfg.get("top_k_fraction", self.top_k_fraction))
        effective_top_k_min = int(effective_cfg.get("top_k_min", self.top_k_min))
        adaptive_min_k = min(20, max(3, int(n_trials * 0.05)))
        effective_min_k = max(effective_top_k_min, adaptive_min_k)
        top_k = _select_top_k_shared(
            completed,
            direction=norm_direction,
            fraction=effective_top_k_fraction,
            min_k=effective_min_k,
        )

        param_meta_map = _build_param_meta_shared(
            search_space,
            parse_search_space_entry_fn=_parse_search_space_entry_shared,
            is_log_scale_candidate_fn=_is_log_scale_candidate_shared,
        )
        anchor_params = _extract_anchor_params_shared(
            param_meta_map,
            top_k or completed,
            categorical_counts_fn=_categorical_counts_shared,
        )
        surrogate = (
            _fit_surrogate_shared(
                all_trials,
                param_meta_map,
                direction=norm_direction,
                surrogate_min_trials=_SURROGATE_MIN_TRIALS,
                normalize_log_value_fn=lambda value, is_log: _normalize_log_value_shared(
                    value,
                    is_log=is_log,
                ),
                apply_direction_fn=_apply_direction_shared,
            )
            if enable_surrogate
            else None
        )
        interactions: dict[str, float] = {}
        if surrogate is not None and enable_interactions:
            rows, _, _ = _build_surrogate_data_shared(
                all_trials,
                param_meta_map,
                direction=norm_direction,
                normalize_log_value_fn=lambda value, is_log: _normalize_log_value_shared(
                    value,
                    is_log=is_log,
                ),
                apply_direction_fn=_apply_direction_shared,
            )
            interactions = _compute_interactions_shared(surrogate, rows)
        interaction_threshold = compute_interaction_threshold(interactions)
        resolved_importances, importance_source, importance_quality = _resolve_importances_shared(
            search_space=search_space,
            external_importances=importances,
            completed_trials=completed,
            direction=norm_direction,
            use_internal=not disable_internal_importances,
            compute_internal_importances_fn=lambda **kwargs: _compute_internal_importances_shared(
                completed_trials=kwargs["completed_trials"],
                direction=kwargs["direction"],
                search_space=kwargs["search_space"],
                spearman_rho_fn=lambda values, scores: _spearman_for_runtime(
                    values,
                    scores,
                    min_points=5,
                ),
                apply_direction_fn=_apply_direction_shared,
            ),
        )
        observed_param_counts: dict[str, int] = {}

        trust_bucket = self._runtime_state.get_trust_bucket(study_key)
        _update_trust_bucket_shared(
            trust_bucket=trust_bucket,
            param_meta_map=param_meta_map,
            completed_trials=completed,
            direction=norm_direction,
            edge_threshold=_EDGE_THRESHOLD,
            trust_failure_threshold=_TRUST_FAILURE,
            normalize_log_value_fn=lambda value, is_log: _normalize_log_value_shared(
                value,
                is_log=is_log,
            ),
        )

        for param_name, spec in search_space.items():
            parsed = _parse_search_space_entry_shared(param_name, spec)
            raw_importance = float(importances.get(param_name, 0.0))
            importance = float(resolved_importances.get(param_name, raw_importance))
            importance_for_decision = importance
            if raw_importance > 0.0 and _is_cost_sensitive_param_shared(param_name):
                importance_for_decision = min(importance, raw_importance)

            all_pairs = [
                (value, _apply_direction_shared(float(trial.value), norm_direction))
                for trial in completed
                if param_name in trial.params
                for value in [trial.params.get(param_name)]
                if value is not None
            ]
            top_k_values = [t.params.get(param_name) for t in top_k if param_name in t.params]
            all_values = [value for value, _ in all_pairs]
            all_scores = [score for _, score in all_pairs]
            top_k_values = [v for v in top_k_values if v is not None]

            if not all_values:
                continue
            observed_param_counts[param_name] = len(all_values)

            param_meta = param_meta_map.get(param_name)
            if not param_meta:
                continue

            n_param_trials = len(all_values)
            evidence_ratio = float(n_param_trials) / float(max(1, n_trials))
            if n_param_trials < effective_min_k or evidence_ratio < 0.5:
                base_confidence = "low"
            elif n_param_trials >= effective_min_trials_aggressive and importance > 0.1:
                base_confidence = "high"
            elif n_param_trials >= effective_min_trials_any:
                base_confidence = "medium"
            else:
                base_confidence = "low"
            interaction_strength = _interaction_strength_for_param_shared(interactions, param_name)
            trust_state = trust_bucket.get(param_name)

            if parsed["type"] in ("float", "int") and "low" in parsed and "high" in parsed:
                rec = _analyze_numeric_param_shared(
                    param_name=param_name,
                    parsed=parsed,
                    all_values=[float(v) for v in all_values],
                    all_scores=all_scores,
                    top_k_values=[float(v) for v in top_k_values],
                    importance=importance_for_decision,
                    n_trials=n_trials,
                    param_meta=param_meta,
                    param_meta_map=param_meta_map,
                    trust_state=trust_state,
                    surrogate=surrogate,
                    anchor_params=anchor_params,
                    interaction_strength=interaction_strength,
                    interaction_threshold=interaction_threshold,
                    confidence=base_confidence,
                    bootstrap_support=None,
                    edge_threshold=_EDGE_THRESHOLD,
                    concentration_cv_threshold=_CONCENTRATION_CV_THRESHOLD,
                    low_importance_threshold=_LOW_IMPORTANCE_THRESHOLD,
                    min_trials_aggressive=effective_min_trials_aggressive,
                    trust_success=_TRUST_SUCCESS,
                    correlation_gate_abs=_CORRELATION_GATE_ABS,
                    cost_sensitive_upper_rho=_COST_SENSITIVE_UPPER_RHO,
                    ucb_std_mult=_UCB_STD_MULT,
                    reservoir_size=_RESERVOIR_SIZE,
                    rust_fast_spearman_fn=_rust_fast_spearman_corr,
                    rust_spearman_min_len=rust_spearman_min_len,
                    np_module=_np,
                )
            elif parsed["type"] == "categorical":
                rec = _analyze_categorical_param_shared(
                    param_name=param_name,
                    parsed=parsed,
                    all_values=all_values,
                    top_k_values=top_k_values,
                    importance=importance_for_decision,
                    n_trials=n_trials,
                    param_meta=param_meta,
                    param_meta_map=param_meta_map,
                    surrogate=surrogate,
                    anchor_params=anchor_params,
                    interaction_strength=interaction_strength,
                    interaction_threshold=interaction_threshold,
                    confidence=base_confidence,
                    bootstrap_support=None,
                    categorical_dominance_threshold=_CATEGORICAL_DOMINANCE_THRESHOLD,
                    categorical_min_topk_samples=categorical_min_topk_samples,
                    categorical_min_topk_unique=categorical_min_topk_unique,
                    categorical_min_effective_categories=categorical_min_effective_categories,
                    low_importance_threshold=_LOW_IMPORTANCE_THRESHOLD,
                    ucb_std_mult=_UCB_STD_MULT,
                    reservoir_size=_RESERVOIR_SIZE,
                )
            else:
                continue

            bootstrap_support = None
            if enable_bootstrap:
                bootstrap_support = _bootstrap_support_for_param_shared(
                    trials=completed,
                    direction=norm_direction,
                    top_k_fraction=effective_top_k_fraction,
                    top_k_min=effective_min_k,
                    param_name=param_name,
                    parsed=parsed,
                    importance=importance,
                    param_meta=param_meta,
                    param_meta_map=param_meta_map,
                    trust_state=None,
                    surrogate=surrogate,
                    anchor_params=anchor_params,
                    interaction_strength=interaction_strength,
                    interaction_threshold=interaction_threshold,
                    seed=42,
                    final_action=rec.action,
                    min_trials_aggressive=effective_min_trials_aggressive,
                    bootstrap_samples=_BOOTSTRAP_SAMPLES,
                    edge_threshold=_EDGE_THRESHOLD,
                    concentration_cv_threshold=_CONCENTRATION_CV_THRESHOLD,
                    low_importance_threshold=_LOW_IMPORTANCE_THRESHOLD,
                    trust_success=_TRUST_SUCCESS,
                    correlation_gate_abs=_CORRELATION_GATE_ABS,
                    cost_sensitive_upper_rho=_COST_SENSITIVE_UPPER_RHO,
                    categorical_dominance_threshold=_CATEGORICAL_DOMINANCE_THRESHOLD,
                    categorical_min_topk_samples=categorical_min_topk_samples,
                    categorical_min_topk_unique=categorical_min_topk_unique,
                    categorical_min_effective_categories=categorical_min_effective_categories,
                    ucb_std_mult=_UCB_STD_MULT,
                    reservoir_size=_RESERVOIR_SIZE,
                    rust_fast_spearman_fn=_rust_fast_spearman_corr,
                    rust_spearman_min_len=rust_spearman_min_len,
                    np_module=_np,
                    select_top_k_fn=_select_top_k_shared,
                )
                rec = _with_bootstrap_confidence_shared(
                    rec,
                    base_confidence=base_confidence,
                    bootstrap_support=bootstrap_support,
                    interaction_strength=interaction_strength,
                    confidence_from_support_fn=_confidence_from_support_shared,
                )

            rec_payload = _recommendation_to_dict_shared(
                rec,
                confidence_prior_n=_CONFIDENCE_SUPPORT_PRIOR_N,
                confidence_prior_p=_CONFIDENCE_SUPPORT_PRIOR_P,
                max_expansion_factor=_MAX_EXPANSION_FACTOR,
            )
            rec_scope = resolve_recommendation_scope(
                evidence_ratio=evidence_ratio,
                conditional_threshold=conditional_scope_threshold,
            )
            rec_payload["scope"] = rec_scope
            if rec_scope == "conditional":
                current_conf = str(rec_payload.get("confidence", "low"))
                if current_conf == "high":
                    rec_payload["confidence"] = "medium"
                elif current_conf == "medium":
                    rec_payload["confidence"] = "low"
            if evidence_ratio < 0.5:
                rec_payload["confidence"] = "low"
            validation = rec_payload.get("validation", {})
            if isinstance(validation, dict) and validation.get("passed") is False:
                original_action = rec_payload.get("action", "keep")
                blocked_reason = validation.get("blocked_reason", "invalid_recommendation")
                rec_payload["blocked_action"] = original_action
                rec_payload["action"] = "keep"
                rec_payload["recommendation"] = {"delta": f"blocked:{blocked_reason}"}
                rationale = str(rec_payload.get("rationale", "")).strip()
                rec_payload["rationale"] = (
                    f"{rationale} Recommendation blocked: {blocked_reason}."
                    if rationale
                    else f"Recommendation blocked: {blocked_reason}."
                )
            rec_payload["policy"] = _policy_stub_shared(
                version=policy_metadata["policy_version"],
                policy_hash=policy_metadata["policy_hash"],
            )
            recommendations.append(rec_payload)

        recommendations.sort(key=lambda r: -r["importance"])
        if enable_self_audit:
            self_audit = _resolve_self_audit_summary_runner_shared(
                search_space=search_space,
                importances=importances,
                completed_trials=completed,
                direction=norm_direction,
                study_name=study_name,
                last_trial=last_trial,
                force_recompute=force_recompute,
                min_prefix=self.self_audit_min_prefix,
                min_suffix=self.self_audit_min_suffix,
                period_trials=self.self_audit_period_trials,
                get_cached_snapshot_fn=self._runtime_state.get_self_audit_snapshot,
                set_cached_snapshot_fn=self._runtime_state.set_self_audit_snapshot,
                build_summary_fn=lambda **kwargs: _build_self_audit_summary_runner_shared(
                    search_space=kwargs["search_space"],
                    importances=kwargs["importances"],
                    completed_trials=kwargs["completed_trials"],
                    direction=kwargs["direction"],
                    study_name=kwargs["study_name"],
                    last_trial=kwargs["last_trial"],
                    min_prefix=self.self_audit_min_prefix,
                    min_suffix=self.self_audit_min_suffix,
                    max_prefixes=self.self_audit_max_prefixes,
                    min_group_total=self.self_audit_min_group_total,
                    wilson_block=self.self_audit_wilson_block,
                    period_trials=self.self_audit_period_trials,
                    min_match_points=self.self_audit_min_match_points,
                    min_trials_aggressive=self.min_trials_aggressive,
                    min_trials_any=self.min_trials_any,
                    top_k_fraction=self.top_k_fraction,
                    top_k_min=self.top_k_min,
                    rust_spearman_min_len=self.rust_spearman_min_len,
                    advisor_factory=lambda cfg: SearchSpaceAdvisor(config_thresholds=cfg),
                    audit_prefix_sizes_fn=_audit_prefix_sizes_shared,
                    is_directional_action_fn=_is_directional_action_shared,
                    match_directional_suffix_trend_fn=lambda **audit_kwargs: _match_directional_suffix_trend_shared(
                        action=audit_kwargs["action"],
                        param_name=audit_kwargs["param_name"],
                        suffix_trials=audit_kwargs["suffix_trials"],
                        direction=audit_kwargs["direction"],
                        min_points=audit_kwargs["min_points"],
                        apply_direction=_apply_direction_shared,
                        spearman_rho=lambda values, scores: _spearman_for_runtime(
                            values,
                            scores,
                            min_points=audit_kwargs["min_points"],
                        ),
                    ),
                    wilson_lower_bound_fn=lambda **w_kwargs: _wilson_lower_bound_shared(
                        successes=w_kwargs["successes"],
                        total=w_kwargs["total"],
                    ),
                ),
            )
        else:
            self_audit = {
                "enabled": False,
                "ran": False,
                "reason": "disabled",
                "period_trials": self.self_audit_period_trials,
                "source_last_trial": last_trial,
            }

        _apply_self_audit_blocks_shared(
            recommendations=recommendations,
            self_audit=self_audit,
            wilson_block_threshold=self.self_audit_wilson_block,
        )
        if isinstance(self_audit, dict):
            self_audit["policy"] = _policy_stub_shared(
                version=policy_metadata["policy_version"],
                policy_hash=policy_metadata["policy_hash"],
            )

        invalid_recommendations = sum(
            1
            for recommendation in recommendations
            if not recommendation.get("validation", {}).get("passed", True)
        )
        reliability_summary = _compute_reliability_summary_shared(recommendations)
        coverage_ratio, missing_params = compute_search_space_coverage(
            search_space=search_space,
            observed_param_counts=observed_param_counts,
            total_trials=n_trials,
        )

        compute_ms = round((time.monotonic() - t0) * 1000, 2)
        result = {
            "recommendations": recommendations,
            "metadata": {
                "study_name": study_name,
                "dataset_fingerprint": dataset_fingerprint,
                "dataset_profile": dataset_profile,
                "last_trial": last_trial,
                "advisor_version": ADVISOR_VERSION,
                "n_completed_trials": n_trials,
                "n_pruned_trials": len([t for t in all_trials if t.state == "PRUNED"]),
                "n_top_k": len(top_k),
                "top_k_min_effective": effective_min_k,
                "direction": norm_direction,
                "direction_normalized": norm_direction,
                "cache_hit": False,
                "cache_layer_hit": "none",
                "cache_status": cache_status,
                "cache_error_code": cache_error_code,
                "forced_recompute": force_recompute,
                "compute_time_ms": compute_ms,
                "insufficient_evidence": False,
                "heuristics_used": False,
                "validation_flags": {
                    "invalid_recommendations": invalid_recommendations,
                    "all_recommendations_valid": invalid_recommendations == 0,
                },
                "reliability_summary": reliability_summary,
                "self_audit": self_audit,
                "importance_source": importance_source,
                "importance_quality": round(float(importance_quality), 4),
                "search_space_coverage_ratio": (
                    float(configured_coverage_ratio)
                    if isinstance(configured_coverage_ratio, (int, float))
                    else round(float(coverage_ratio), 4)
                ),
                "missing_params": missing_params,
                "distribution_conflicts": configured_distribution_conflicts,
                "multiobjective_mode": projection.metadata.get(
                    "multiobjective_mode",
                    "single_objective",
                ),
                "objective_count": projection.metadata.get("objective_count", 1),
                "objective_directions": projection.metadata.get("objective_directions", []),
                "pareto_front_size": projection.metadata.get("pareto_front_size", 0),
                "hypervolume": projection.metadata.get("hypervolume"),
                "acceleration": {
                    "rust_spearman_available": bool(callable(_rust_fast_spearman_corr)),
                    "rust_spearman_min_len": int(rust_spearman_min_len),
                    "surrogate_enabled": enable_surrogate,
                    "interactions_enabled": enable_interactions,
                    "internal_importances_disabled": disable_internal_importances,
                },
                "policy_version": policy_metadata["policy_version"],
                "policy_hash": policy_metadata["policy_hash"],
                "policy_thresholds": policy_metadata["policy_thresholds"],
            },
        }
        validation_lb = float(
            result["metadata"]["reliability_summary"].get("validation_pass_wilson_lb", 0.0)
        )
        degraded_count_after, cooldown_after = _update_adaptive_performance_state_shared(
            set_state_fn=self._runtime_state.set_adaptive_state,
            study_key=study_key,
            enabled=adaptive_perf_enabled,
            compute_ms=float(compute_ms),
            validation_lb=validation_lb,
            threshold_ms=adaptive_perf_ms_threshold,
            validation_lb_min=adaptive_perf_validation_lb_min,
            cooldown_calls=adaptive_perf_cooldown_calls,
            cooldown_before=cooldown_before,
            degraded_count_before=degraded_count_before,
        )
        result["metadata"]["adaptive_performance"] = _build_adaptive_performance_metadata_shared(
            enabled=adaptive_perf_enabled,
            decision=adaptive_decision,
            threshold_ms=adaptive_perf_ms_threshold,
            validation_lb_min=adaptive_perf_validation_lb_min,
            cooldown_calls=adaptive_perf_cooldown_calls,
            cooldown_before=cooldown_before,
            cooldown_after=cooldown_after,
            degraded_count_before=degraded_count_before,
            degraded_count_after=degraded_count_after,
            last_compute_ms_before=last_compute_ms_before,
            last_validation_lb_before=last_validation_lb_before,
        )

        cache_write = self._advice_cache.set_with_status(cache_spec, result)
        result["metadata"]["cache_write_status"] = cache_write.status
        result["metadata"]["cache_write_error_code"] = cache_write.error_code

        logger.debug(
            f"component_name=search_space_advisor "
            f"key_parameters=n_trials:{n_trials},n_params:{len(recommendations)},compute_ms:{compute_ms} "
            f"message='Search space advice computed'"
        )

        return result

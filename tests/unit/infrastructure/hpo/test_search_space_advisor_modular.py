"""Unit tests for modularized Search Space Advisor behavior."""

from __future__ import annotations

import asyncio
import gc
import warnings
from types import SimpleNamespace

from optuna.distributions import FloatDistribution, IntDistribution

from pff.infrastructure.hpo.callbacks_internal.visualizers import _serialize_search_space
from pff.infrastructure.hpo.search_space_advisor import SearchSpaceAdvisor
from pff.infrastructure.hpo.search_space_advisor.cache import (
    AdvisorCache,
    AdvisorCacheSpec,
    build_objective_schema_hash,
    build_search_space_hash,
)
from pff.infrastructure.hpo.search_space_advisor.multiobjective import (
    build_multiobjective_projection,
)
from pff.infrastructure.hpo.search_space_advisor.validation import validate_recommendation_payload


def test_multiobjective_projection_is_not_primary_only() -> None:
    trials = [
        {"values": [1.0, 0.0], "value": 1.0, "params": {"x": 1}},
        {"values": [0.8, 0.8], "value": 0.8, "params": {"x": 2}},
        {"values": [0.0, 1.0], "value": 0.0, "params": {"x": 3}},
    ]

    projection = build_multiobjective_projection(
        trials,
        fallback_direction="maximize",
        objective_directions=["maximize", "maximize"],
    )

    assert projection.metadata["multiobjective_mode"].startswith("pareto")
    assert projection.scores[1] is not None
    assert projection.scores[0] is not None
    assert float(projection.scores[1]) > float(projection.scores[0])


def test_advisor_metadata_and_scope_fields_present() -> None:
    advisor = SearchSpaceAdvisor(config_thresholds={"persistent_cache_enabled": False})
    search_space = {
        "p1": {"type": "float", "low": 0.0, "high": 1.0},
        "p2": {"type": "float", "low": 0.0, "high": 1.0},
    }
    trials = [
        {"id": 1, "state": "COMPLETE", "value": 0.30, "params": {"p1": 0.1, "p2": 0.9}},
        {"id": 2, "state": "COMPLETE", "value": 0.40, "params": {"p1": 0.2, "p2": 0.8}},
        {"id": 3, "state": "COMPLETE", "value": 0.45, "params": {"p1": 0.3}},
        {"id": 4, "state": "COMPLETE", "value": 0.50, "params": {"p1": 0.4}},
        {"id": 5, "state": "COMPLETE", "value": 0.55, "params": {"p1": 0.5, "p2": 0.6}},
        {"id": 6, "state": "COMPLETE", "value": 0.58, "params": {"p1": 0.6}},
    ]

    result = advisor.advise(
        search_space=search_space,
        trials_data=trials,
        importances={},
        direction="maximize",
        study_name="unit_scope",
        dataset_fingerprint="fp_scope",
        force_recompute=True,
        enable_self_audit=False,
        objective_directions=["maximize"],
    )

    metadata = result.get("metadata", {})
    assert "importance_source" in metadata
    assert "importance_quality" in metadata
    assert "search_space_coverage_ratio" in metadata
    assert "multiobjective_mode" in metadata
    assert "cache_layer_hit" in metadata
    assert int(metadata.get("acceleration", {}).get("rust_spearman_min_len", 0)) >= 1

    recommendations = result.get("recommendations", [])
    assert recommendations
    assert all("scope" in rec for rec in recommendations)


def test_advisor_flags_fixed_parameters_that_need_exploration_or_are_stable() -> None:
    advisor = SearchSpaceAdvisor(config_thresholds={"persistent_cache_enabled": False})
    search_space = {
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "embedding_dim": 512,
        "lambda_pc": 0.03,
    }
    trials = [
        {"id": 1, "state": "COMPLETE", "value": 0.31, "params": {"learning_rate": 1e-4}},
        {"id": 2, "state": "COMPLETE", "value": 0.34, "params": {"learning_rate": 2e-4}},
        {"id": 3, "state": "COMPLETE", "value": 0.38, "params": {"learning_rate": 4e-4}},
        {"id": 4, "state": "COMPLETE", "value": 0.42, "params": {"learning_rate": 8e-4}},
        {"id": 5, "state": "COMPLETE", "value": 0.43, "params": {"learning_rate": 1e-3}},
        {"id": 6, "state": "COMPLETE", "value": 0.44, "params": {"learning_rate": 2e-3}},
    ]

    result = advisor.advise(
        search_space=search_space,
        trials_data=trials,
        importances={"learning_rate": 0.4, "embedding_dim": 0.24, "lambda_pc": 0.01},
        direction="maximize",
        study_name="unit_fixed_diagnostics",
        force_recompute=True,
        enable_bootstrap=False,
        enable_self_audit=False,
        advisor_config={
            "enable_surrogate": False,
            "enable_interactions": False,
            "disable_internal_importances": True,
        },
    )

    recommendations = {
        str(rec.get("param_name")): rec for rec in result.get("recommendations", [])
    }
    embedding = recommendations["embedding_dim"]
    lambda_pc = recommendations["lambda_pc"]

    assert embedding["action"] == "keep"
    assert embedding["recommendation"]["diagnostic"] == "needs_exploration"
    assert embedding["recommendation"]["suggested_action"] == "convert_fixed_to_bounded_search"
    assert "cannot estimate sensitivity from a single value" in embedding["rationale"]
    assert lambda_pc["action"] == "keep"
    assert lambda_pc["recommendation"]["diagnostic"] == "stable_fixed_value"
    assert lambda_pc["recommendation"]["suggested_action"] == "keep_fixed"


def test_cache_hash_is_deterministic_and_sensitive_to_objective_schema() -> None:
    search_space = {"a": {"type": "float", "low": 1e-4, "high": 1e-2}}
    h1 = build_search_space_hash(search_space)
    h2 = build_search_space_hash({"a": {"high": 1e-2, "low": 1e-4, "type": "float"}})
    assert h1 == h2

    obj1 = build_objective_schema_hash(["maximize"])
    obj2 = build_objective_schema_hash(["maximize", "minimize"])
    assert obj1 != obj2

    spec1 = AdvisorCacheSpec(
        study_name="s",
        dataset_fingerprint="fp",
        direction="maximize",
        advisor_version="2.3.0",
        last_trial=10,
        search_space_hash=h1,
        objective_schema_hash=obj1,
    )
    spec2 = AdvisorCacheSpec(
        study_name="s",
        dataset_fingerprint="fp",
        direction="maximize",
        advisor_version="2.3.0",
        last_trial=10,
        search_space_hash=h1,
        objective_schema_hash=obj2,
    )
    assert spec1.cache_key() != spec2.cache_key()


def test_visualizer_serializes_search_space_from_all_trials() -> None:
    t1 = SimpleNamespace(
        params={"a": 0.2},
        distributions={"a": FloatDistribution(0.0, 1.0)},
    )
    t2 = SimpleNamespace(
        params={"b": 3},
        distributions={"b": IntDistribution(1, 5)},
    )

    search_space, coverage = _serialize_search_space([t1, t2])

    assert "a" in search_space
    assert "b" in search_space
    assert coverage["search_space_coverage_ratio"] >= 1.0
    assert coverage["missing_params"] == []


def test_advisor_lightweight_config_disables_internal_importance() -> None:
    advisor = SearchSpaceAdvisor(config_thresholds={"persistent_cache_enabled": False})
    search_space = {
        "lr": {"type": "float", "low": 1e-4, "high": 1e-2},
        "batch_size": {"type": "int", "low": 128, "high": 1024},
    }
    trials = [
        {"id": 1, "state": "COMPLETE", "value": 0.30, "params": {"lr": 1e-4, "batch_size": 128}},
        {"id": 2, "state": "COMPLETE", "value": 0.36, "params": {"lr": 5e-4, "batch_size": 256}},
        {"id": 3, "state": "COMPLETE", "value": 0.39, "params": {"lr": 8e-4, "batch_size": 512}},
        {"id": 4, "state": "COMPLETE", "value": 0.41, "params": {"lr": 9e-4, "batch_size": 512}},
        {"id": 5, "state": "COMPLETE", "value": 0.44, "params": {"lr": 1e-3, "batch_size": 768}},
        {"id": 6, "state": "COMPLETE", "value": 0.46, "params": {"lr": 2e-3, "batch_size": 1024}},
    ]

    result = advisor.advise(
        search_space=search_space,
        trials_data=trials,
        importances={"lr": 0.8, "batch_size": 0.2},
        direction="maximize",
        study_name="unit_lightweight",
        force_recompute=True,
        enable_self_audit=False,
        advisor_config={
            "enable_surrogate": False,
            "enable_interactions": False,
            "disable_internal_importances": True,
        },
    )

    metadata = result.get("metadata", {})
    assert metadata.get("importance_source") == "external"
    assert metadata.get("multiobjective_mode") in {
        "single_objective",
        "pareto_scalarized",
        "pareto_hypervolume",
    }


def test_adaptive_performance_degrades_costly_paths_when_stable() -> None:
    advisor = SearchSpaceAdvisor(
        config_thresholds={
            "persistent_cache_enabled": False,
            "adaptive_perf_enabled": True,
            "adaptive_perf_ms_threshold": 0.0,
            "adaptive_perf_validation_lb_min": 0.0,
        }
    )
    search_space = {
        "lr": {"type": "float", "low": 1e-4, "high": 1e-2},
        "batch_size": {"type": "int", "low": 128, "high": 1024},
    }
    trials = [
        {"id": 1, "state": "COMPLETE", "value": 0.30, "params": {"lr": 1e-4, "batch_size": 128}},
        {"id": 2, "state": "COMPLETE", "value": 0.36, "params": {"lr": 2e-4, "batch_size": 192}},
        {"id": 3, "state": "COMPLETE", "value": 0.39, "params": {"lr": 4e-4, "batch_size": 256}},
        {"id": 4, "state": "COMPLETE", "value": 0.41, "params": {"lr": 6e-4, "batch_size": 384}},
        {"id": 5, "state": "COMPLETE", "value": 0.44, "params": {"lr": 9e-4, "batch_size": 512}},
        {"id": 6, "state": "COMPLETE", "value": 0.46, "params": {"lr": 1.4e-3, "batch_size": 640}},
        {"id": 7, "state": "COMPLETE", "value": 0.47, "params": {"lr": 2e-3, "batch_size": 768}},
        {"id": 8, "state": "COMPLETE", "value": 0.48, "params": {"lr": 3e-3, "batch_size": 896}},
    ]
    common = dict(
        search_space=search_space,
        trials_data=trials,
        importances={"lr": 0.8, "batch_size": 0.2},
        direction="maximize",
        study_name="unit_adaptive_perf",
        force_recompute=True,
        enable_self_audit=False,
    )

    run1 = advisor.advise(**common)
    run2 = advisor.advise(**common)
    run3 = advisor.advise(**common)

    meta1 = run1.get("metadata", {})
    meta2 = run2.get("metadata", {})
    meta3 = run3.get("metadata", {})
    perf1 = meta1.get("adaptive_performance", {})
    perf2 = meta2.get("adaptive_performance", {})
    perf3 = meta3.get("adaptive_performance", {})

    assert perf1.get("enabled") is True
    assert int(perf1.get("degraded_count_after", -1)) >= 1
    assert perf2.get("decision") in {"disable_interactions", "disable_surrogate"}
    assert meta2.get("acceleration", {}).get("interactions_enabled") is False
    assert perf3.get("decision") == "disable_surrogate"
    assert meta3.get("acceleration", {}).get("surrogate_enabled") is False


def test_cache_hit_preserves_adaptive_performance_metadata() -> None:
    advisor = SearchSpaceAdvisor(config_thresholds={"persistent_cache_enabled": False})
    search_space = {"x": {"type": "float", "low": 0.0, "high": 1.0}}
    trials = [
        {"id": 1, "state": "COMPLETE", "value": 0.10, "params": {"x": 0.1}},
        {"id": 2, "state": "COMPLETE", "value": 0.15, "params": {"x": 0.2}},
        {"id": 3, "state": "COMPLETE", "value": 0.20, "params": {"x": 0.3}},
        {"id": 4, "state": "COMPLETE", "value": 0.25, "params": {"x": 0.4}},
        {"id": 5, "state": "COMPLETE", "value": 0.30, "params": {"x": 0.5}},
    ]
    args = dict(
        search_space=search_space,
        trials_data=trials,
        importances={"x": 1.0},
        direction="maximize",
        study_name="unit_cache_adaptive_metadata",
        enable_self_audit=False,
    )

    advisor.advise(force_recompute=True, **args)
    cached = advisor.advise(force_recompute=False, **args)

    metadata = cached.get("metadata", {})
    adaptive = metadata.get("adaptive_performance", {})
    acceleration = metadata.get("acceleration", {})
    assert metadata.get("cache_hit") is True
    assert adaptive.get("decision") == "cache_hit"
    assert isinstance(acceleration.get("rust_spearman_available"), bool)
    assert int(acceleration.get("rust_spearman_min_len", 0)) >= 1


def test_policy_metadata_and_recommendation_trace_present() -> None:
    advisor = SearchSpaceAdvisor(config_thresholds={"persistent_cache_enabled": False})
    result = advisor.advise(
        search_space={"x": {"type": "float", "low": 0.0, "high": 1.0}},
        trials_data=[
            {"id": 1, "state": "COMPLETE", "value": 0.10, "params": {"x": 0.1}},
            {"id": 2, "state": "COMPLETE", "value": 0.20, "params": {"x": 0.2}},
            {"id": 3, "state": "COMPLETE", "value": 0.30, "params": {"x": 0.3}},
            {"id": 4, "state": "COMPLETE", "value": 0.40, "params": {"x": 0.4}},
            {"id": 5, "state": "COMPLETE", "value": 0.50, "params": {"x": 0.5}},
        ],
        importances={"x": 1.0},
        direction="maximize",
        study_name="unit_policy_trace",
        force_recompute=True,
        enable_self_audit=False,
    )
    metadata = result.get("metadata", {})
    assert isinstance(metadata.get("policy_version"), str)
    assert isinstance(metadata.get("policy_hash"), str)
    assert isinstance(metadata.get("policy_thresholds"), dict)
    recommendations = result.get("recommendations", [])
    assert recommendations
    assert all(
        isinstance(rec.get("policy"), dict) for rec in recommendations if isinstance(rec, dict)
    )
    assert all(
        rec.get("policy", {}).get("hash") == metadata.get("policy_hash")
        for rec in recommendations
        if isinstance(rec, dict)
    )


def test_validation_blocks_non_positive_log_uniform_distribution() -> None:
    validation = validate_recommendation_payload(
        {
            "action": "change_distribution",
            "recommendation": {"distribution": "log_uniform", "low": 0.0, "high": 0.1},
        }
    )
    assert validation["passed"] is False
    assert validation["blocked_reason"] == "change_distribution_non_positive_log_bounds"


def test_categorical_reduction_requires_minimum_topk_evidence() -> None:
    advisor = SearchSpaceAdvisor(
        config_thresholds={
            "min_trials_any": 3,
            "persistent_cache_enabled": False,
            "categorical_min_topk_samples": 6,
            "categorical_min_topk_unique": 2,
            "categorical_min_effective_categories": 1.2,
        }
    )
    trials = [
        {"id": 1, "state": "COMPLETE", "value": 0.95, "params": {"choice": "A"}},
        {"id": 2, "state": "COMPLETE", "value": 0.90, "params": {"choice": "A"}},
        {"id": 3, "state": "COMPLETE", "value": 0.89, "params": {"choice": "A"}},
        {"id": 4, "state": "COMPLETE", "value": 0.20, "params": {"choice": "B"}},
        {"id": 5, "state": "COMPLETE", "value": 0.10, "params": {"choice": "C"}},
        {"id": 6, "state": "COMPLETE", "value": 0.08, "params": {"choice": "B"}},
    ]
    result = advisor.advise(
        search_space={"choice": {"type": "categorical", "choices": ["A", "B", "C"]}},
        trials_data=trials,
        importances={"choice": 0.8},
        direction="maximize",
        study_name="unit_categorical_evidence_gate",
        force_recompute=True,
        enable_self_audit=False,
    )
    rec = result["recommendations"][0]
    assert rec["action"] == "keep"
    assert "Category reduction blocked" in rec["rationale"]


def test_cache_status_metadata_is_observable() -> None:
    advisor = SearchSpaceAdvisor(config_thresholds={"persistent_cache_enabled": False})
    result = advisor.advise(
        search_space={"x": {"type": "float", "low": 0.0, "high": 1.0}},
        trials_data=[
            {"id": 1, "state": "COMPLETE", "value": 0.10, "params": {"x": 0.1}},
            {"id": 2, "state": "COMPLETE", "value": 0.20, "params": {"x": 0.2}},
            {"id": 3, "state": "COMPLETE", "value": 0.30, "params": {"x": 0.3}},
            {"id": 4, "state": "COMPLETE", "value": 0.40, "params": {"x": 0.4}},
            {"id": 5, "state": "COMPLETE", "value": 0.50, "params": {"x": 0.5}},
        ],
        importances={"x": 1.0},
        direction="maximize",
        study_name="unit_cache_status",
        force_recompute=True,
        enable_self_audit=False,
    )
    metadata = result.get("metadata", {})
    assert metadata.get("cache_status") in {"ok", "disabled", "degraded"}
    assert "cache_error_code" in metadata


def test_cache_event_loop_running_closes_coroutines_without_runtime_warning(monkeypatch) -> None:
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: object())

    cache = AdvisorCache(max_memory_items=8, ttl_seconds=60, enable_persistent_l2=True)
    spec = AdvisorCacheSpec(
        study_name="unit_loop_running_cache",
        dataset_fingerprint="fp",
        direction="maximize",
        advisor_version="2.3.0",
        last_trial=7,
        search_space_hash=build_search_space_hash(
            {"x": {"type": "float", "low": 0.0, "high": 1.0}}
        ),
        objective_schema_hash=build_objective_schema_hash(["maximize"]),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        read_result = cache.get_with_status(spec)
        write_result = cache.set_with_status(spec, {"recommendations": [], "metadata": {}})
        gc.collect()

    assert read_result.status == "disabled"
    assert read_result.error_code == "event_loop_running"
    assert write_result.status == "disabled"
    assert write_result.error_code == "event_loop_running"
    assert not any("was never awaited" in str(w.message) for w in caught)


def test_adaptive_performance_cooldown_prevents_flapping() -> None:
    advisor = SearchSpaceAdvisor(
        config_thresholds={
            "persistent_cache_enabled": False,
            "adaptive_perf_enabled": True,
            "adaptive_perf_ms_threshold": 0.0,
            "adaptive_perf_validation_lb_min": 0.0,
            "adaptive_perf_cooldown_calls": 2,
        }
    )
    search_space = {"x": {"type": "float", "low": 0.0, "high": 1.0}}
    trials = [
        {"id": 1, "state": "COMPLETE", "value": 0.10, "params": {"x": 0.1}},
        {"id": 2, "state": "COMPLETE", "value": 0.20, "params": {"x": 0.2}},
        {"id": 3, "state": "COMPLETE", "value": 0.30, "params": {"x": 0.3}},
        {"id": 4, "state": "COMPLETE", "value": 0.40, "params": {"x": 0.4}},
        {"id": 5, "state": "COMPLETE", "value": 0.50, "params": {"x": 0.5}},
    ]
    args = dict(
        search_space=search_space,
        trials_data=trials,
        importances={"x": 1.0},
        direction="maximize",
        study_name="unit_adaptive_cooldown",
        force_recompute=True,
        enable_self_audit=False,
    )

    advisor.advise(**args)
    run2 = advisor.advise(advisor_config={"adaptive_perf_ms_threshold": 1e9}, **args)

    adaptive = run2.get("metadata", {}).get("adaptive_performance", {})
    acceleration = run2.get("metadata", {}).get("acceleration", {})
    assert int(adaptive.get("cooldown_before", 0)) >= 1
    assert acceleration.get("interactions_enabled") is False

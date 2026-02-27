"""Unit tests for extracted Search Space Advisor runtime helper modules."""

from __future__ import annotations

from pff.infrastructure.hpo.search_space_advisor.advisor_runtime import (
    build_adaptive_performance_metadata,
    resolve_adaptive_controls,
    resolve_runtime_settings,
    update_adaptive_performance_state,
)
from pff.infrastructure.hpo.search_space_advisor.models import ParamMeta, TrialSummary, TrustState
from pff.infrastructure.hpo.search_space_advisor.self_audit import apply_self_audit_blocks
from pff.infrastructure.hpo.search_space_advisor.self_audit_runner import build_self_audit_summary
from pff.infrastructure.hpo.search_space_advisor.trial_projection import (
    build_trial_summaries,
    extract_pruned_value,
)
from pff.infrastructure.hpo.search_space_advisor.trust import update_trust_bucket


def test_resolve_runtime_settings_merges_and_flags_explicit() -> None:
    settings = resolve_runtime_settings(
        base_config={"enable_surrogate": True, "min_trials_any": 5},
        advisor_config={"enable_surrogate": False, "self_audit_period_trials": 7},
        direction="MAXIMIZE",
        study_name="study_x",
        min_trials_any_default=5,
        min_trials_aggressive_default=20,
        top_k_fraction_default=0.25,
        top_k_min_default=3,
        rust_spearman_min_len_default=512,
        self_audit_period_trials_default=10,
        self_audit_min_prefix_default=8,
        self_audit_min_suffix_default=5,
        self_audit_max_prefixes_default=6,
        self_audit_min_group_total_default=8,
        self_audit_wilson_block_default=0.35,
        self_audit_min_match_points_default=5,
    )

    assert settings.norm_direction == "maximize"
    assert settings.study_key == "study_x"
    assert settings.enable_surrogate is False
    assert settings.self_audit_period_trials == 7
    assert settings.explicit_surrogate is True


def test_resolve_adaptive_controls_disables_costly_paths() -> None:
    decision = resolve_adaptive_controls(
        adaptive_perf_enabled=True,
        adaptive_perf_ms_threshold=1000.0,
        adaptive_perf_validation_lb_min=0.7,
        degraded_count_before=3,
        cooldown_before=1,
        last_compute_ms_before=1500.0,
        last_validation_lb_before=0.9,
        enable_surrogate=True,
        enable_interactions=True,
        disable_internal_importances=False,
        explicit_surrogate=False,
        explicit_interactions=False,
        explicit_internal_importances=False,
    )

    assert decision.enable_interactions is False
    assert decision.enable_surrogate is False
    assert decision.disable_internal_importances is True
    assert decision.adaptive_decision in {
        "disable_surrogate",
        "disable_internal_importances",
        "disable_interactions",
    }


def test_build_adaptive_performance_metadata_shape() -> None:
    metadata = build_adaptive_performance_metadata(
        enabled=True,
        decision="disable_surrogate",
        threshold_ms=1200.0,
        validation_lb_min=0.75,
        cooldown_calls=2,
        cooldown_before=1,
        cooldown_after=2,
        degraded_count_before=1,
        degraded_count_after=2,
        last_compute_ms_before=1500.0,
        last_validation_lb_before=0.8,
    )

    assert metadata["enabled"] is True
    assert metadata["decision"] == "disable_surrogate"
    assert metadata["cooldown_after"] == 2


def test_update_adaptive_performance_state_enabled_path() -> None:
    recorded: dict[str, dict] = {}

    def _set_state(study_key: str, payload: dict) -> None:
        recorded[study_key] = payload

    degraded_count_after, cooldown_after = update_adaptive_performance_state(
        set_state_fn=_set_state,
        study_key="study_a",
        enabled=True,
        compute_ms=2000.0,
        validation_lb=0.9,
        threshold_ms=1200.0,
        validation_lb_min=0.75,
        cooldown_calls=2,
        cooldown_before=0,
        degraded_count_before=0,
    )

    assert degraded_count_after >= 1
    assert cooldown_after >= 1
    assert recorded["study_a"]["enabled"] is True


def test_extract_pruned_value_and_build_trial_summaries() -> None:
    assert extract_pruned_value({1: 0.1, 3: 0.2}) == 0.2

    trials_data = [
        {"id": 1, "state": "COMPLETE", "value": 0.1, "params": {"x": 1}},
        {
            "id": 2,
            "state": "PRUNED",
            "value": None,
            "params": {"x": 2},
            "intermediate_values": {2: 0.3, 4: 0.5},
        },
    ]
    all_trials, completed = build_trial_summaries(
        trials_data,
        projected_scores=[0.15, None],
        normalize_trial_state_fn=lambda state: str(state).upper(),
    )

    assert len(all_trials) == 2
    assert len(completed) == 1
    assert all_trials[1].value == 0.5


def test_update_trust_bucket_tracks_edge_success() -> None:
    trust_bucket: dict[str, TrustState] = {}
    param_meta_map = {
        "lr": ParamMeta(
            name="lr",
            param_type="float",
            is_categorical=False,
            is_log=False,
            low=0.0,
            high=1.0,
        )
    }
    completed = [
        TrialSummary(number=1, value=0.2, params={"lr": 0.9}, state="COMPLETE"),
        TrialSummary(number=2, value=0.3, params={"lr": 0.95}, state="COMPLETE"),
    ]

    best_trial = update_trust_bucket(
        trust_bucket=trust_bucket,
        param_meta_map=param_meta_map,
        completed_trials=completed,
        direction="maximize",
        edge_threshold=0.15,
        trust_failure_threshold=5,
        normalize_log_value_fn=lambda value, _is_log: value,
    )

    assert best_trial is not None
    assert trust_bucket["lr"].upper_success >= 1


def test_apply_self_audit_blocks_directional_recommendations() -> None:
    recommendations = [
        {
            "param_name": "lr",
            "action": "expand_upper",
            "recommendation": {"old_high": 0.1, "new_high": 0.2},
            "rationale": "x",
        }
    ]
    self_audit = {
        "villains": [{"param_name": "lr", "action": "expand_upper", "hit_rate_wilson_lb": 0.1}]
    }

    blocked = apply_self_audit_blocks(
        recommendations=recommendations,
        self_audit=self_audit,
        wilson_block_threshold=0.35,
    )

    assert blocked == 1
    assert recommendations[0]["action"] == "keep"
    assert recommendations[0]["blocked_by"] == "self_audit"


def test_build_self_audit_summary_includes_param_and_action_diagnostics() -> None:
    class _AdvisorStub:
        def advise(self, **_kwargs):
            return {
                "recommendations": [
                    {"param_name": "lr", "action": "expand_upper"},
                    {"param_name": "lambda_pc", "action": "expand_lower"},
                ]
            }

    completed_trials = [
        TrialSummary(number=i, value=0.1 + (i * 0.01), params={"lr": i}, state="COMPLETE")
        for i in range(8)
    ]

    summary = build_self_audit_summary(
        search_space={"lr": {"type": "float", "low": 0.0, "high": 1.0}},
        importances={},
        completed_trials=completed_trials,
        direction="maximize",
        study_name="unit_self_audit_diag",
        last_trial=7,
        min_prefix=3,
        min_suffix=2,
        max_prefixes=2,
        min_group_total=2,
        wilson_block=0.35,
        period_trials=5,
        min_match_points=2,
        min_trials_aggressive=5,
        min_trials_any=3,
        top_k_fraction=0.25,
        top_k_min=3,
        rust_spearman_min_len=64,
        advisor_factory=lambda _cfg: _AdvisorStub(),
        audit_prefix_sizes_fn=lambda *_args, **_kwargs: [4, 5],
        is_directional_action_fn=lambda action: action in {"expand_upper", "expand_lower"},
        match_directional_suffix_trend_fn=lambda **kwargs: (
            (False, -0.4) if kwargs.get("param_name") == "lr" else (True, 0.3)
        ),
        wilson_lower_bound_fn=lambda successes, total: (successes / total) if total else 0.0,
    )

    diagnostics = summary.get("diagnostics", {})
    assert summary.get("ran") is True
    assert isinstance(diagnostics.get("params"), list)
    assert isinstance(diagnostics.get("actions"), list)
    assert "worst_params" in diagnostics
    assert "worst_actions" in diagnostics

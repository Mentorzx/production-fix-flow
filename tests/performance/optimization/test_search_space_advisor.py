"""Tests for SearchSpaceAdvisor module.

Covers:
- Numeric parameter analysis (edge proximity, concentration, log-scale detection)
- Categorical parameter analysis (dominance, low importance)
- Cache hit/miss behavior
- Insufficient evidence handling
- Patch generation
- Multi-objective / direction handling
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pff.infrastructure.hpo.search_space_advisor import (
    ADVISOR_VERSION,
    SearchSpaceAdvisor,
    compute_dataset_profile_fingerprint,
    generate_search_space_patch,
)
from pff.infrastructure.hpo.search_space_advisor.confidence import (
    compute_confidence_score as _compute_confidence_score,
)
from pff.infrastructure.hpo.search_space_advisor.cache import (
    AdvisorCacheSpec,
    build_objective_schema_hash,
    build_search_space_hash,
    compute_cache_key,
)
from pff.infrastructure.hpo.search_space_advisor.models import TrialSummary as _TrialSummary
from pff.infrastructure.hpo.search_space_advisor.analysis_categorical import (
    categorical_counts as _categorical_counts,
)
from pff.infrastructure.hpo.search_space_advisor.parsing import (
    is_log_scale_candidate as _is_log_scale_candidate,
    parse_search_space_entry as _parse_search_space_entry,
)
from pff.infrastructure.hpo.search_space_advisor.statistics import (
    numeric_stats as _numeric_stats,
    select_top_k as _select_top_k,
)


class TestNumericStats:
    """Group tests for TestNumericStats."""

    def test_empty(self):
        """Validate test empty."""
        assert _numeric_stats([]) == {}

    def test_single_value(self):
        """Validate test single value."""
        stats = _numeric_stats([5.0])
        assert stats["mean"] == 5.0
        assert stats["count"] == 1
        assert stats["min"] == 5.0
        assert stats["max"] == 5.0

    def test_known_distribution(self):
        """Validate test known distribution."""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = _numeric_stats(vals)
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["count"] == 5
        assert stats["q50"] == pytest.approx(3.0)


class TestCategoricalCounts:
    """Group tests for TestCategoricalCounts."""

    def test_empty(self):
        """Validate test empty."""
        assert _categorical_counts([]) == {}

    def test_basic_counts(self):
        """Validate test basic counts."""
        counts = _categorical_counts(["a", "b", "a", "c", "a"])
        assert counts == {"a": 3, "b": 1, "c": 1}

    def test_numeric_as_string(self):
        """Validate test numeric as string."""
        counts = _categorical_counts([1, 2, 1])
        assert counts == {"1": 2, "2": 1}


class TestLogScaleCandidate:
    """Group tests for TestLogScaleCandidate."""

    def test_lr_name(self):
        """Validate test lr name."""
        assert _is_log_scale_candidate("learning_rate", 1e-5, 1e-1) is True

    def test_weight_decay_name(self):
        """Validate test weight decay name."""
        assert _is_log_scale_candidate("weight_decay", 1e-6, 1e-2) is True

    def test_large_ratio(self):
        """Validate test large ratio."""
        assert _is_log_scale_candidate("alpha", 0.001, 1.0) is True

    def test_small_ratio(self):
        """Validate test small ratio."""
        assert _is_log_scale_candidate("dropout", 0.1, 0.5) is False

    def test_zero_low(self):
        """Validate test zero low."""
        assert _is_log_scale_candidate("some_param", 0.0, 1.0) is False


class TestParseSearchSpaceEntry:
    """Group tests for TestParseSearchSpaceEntry."""

    def test_dict_with_low_high_float(self):
        """Validate test dict with low high float."""
        spec = {"low": 0.1, "high": 1.0}
        parsed = _parse_search_space_entry("lr", spec)
        assert parsed["type"] == "float"
        assert parsed["low"] == 0.1
        assert parsed["high"] == 1.0

    def test_dict_with_choices(self):
        """Validate test dict with choices."""
        spec = {"choices": ["adam", "sgd", "rmsprop"]}
        parsed = _parse_search_space_entry("optimizer", spec)
        assert parsed["type"] == "categorical"
        assert parsed["choices"] == ["adam", "sgd", "rmsprop"]

    def test_list_as_range(self):
        """Validate test list as range."""
        parsed = _parse_search_space_entry("dim", [64, 512])
        assert parsed["type"] == "int"
        assert parsed["low"] == 64
        assert parsed["high"] == 512

    def test_list_as_categorical(self):
        """Validate test list as categorical."""
        parsed = _parse_search_space_entry("act", ["relu", "gelu", "tanh"])
        assert parsed["type"] == "categorical"

    def test_fixed_value(self):
        """Validate test fixed value."""
        parsed = _parse_search_space_entry("seed", 42)
        assert parsed["type"] == "fixed"
        assert parsed["value"] == 42

    def test_none_spec(self):
        """Validate test none spec."""
        parsed = _parse_search_space_entry("x", None)
        assert parsed["type"] == "unknown"

    def test_optuna_json_format(self):
        """Validate test optuna json format."""
        spec = {
            "name": "FloatDistribution",
            "attributes": {"low": 1e-5, "high": 1e-1, "log": True},
        }
        parsed = _parse_search_space_entry("lr", spec)
        assert parsed["type"] == "float"
        assert parsed["log"] is True

    def test_optuna_json_string_format(self):
        """Validate stringified Optuna distribution parsing."""
        spec = '{"name":"FloatDistribution","attributes":{"low":1e-5,"high":1e-1,"log":true}}'
        parsed = _parse_search_space_entry("lr", spec)
        assert parsed["type"] == "float"
        assert parsed["low"] == pytest.approx(1e-5)
        assert parsed["high"] == pytest.approx(1e-1)
        assert parsed["log"] is True


class TestSelectTopK:
    """Group tests for TestSelectTopK."""

    def _make_trials(self, values: list[float]) -> list[_TrialSummary]:
        return [_TrialSummary(number=i, value=v, params={}) for i, v in enumerate(values)]

    def test_empty(self):
        """Validate test empty."""
        assert _select_top_k([], "maximize") == []

    def test_maximize(self):
        """Validate test maximize."""
        trials = self._make_trials([0.1, 0.5, 0.9, 0.3, 0.7])
        top = _select_top_k(trials, "maximize", fraction=0.4, min_k=1)
        assert top[0].value == 0.9

    def test_minimize(self):
        """Validate test minimize."""
        trials = self._make_trials([0.1, 0.5, 0.9, 0.3, 0.7])
        top = _select_top_k(trials, "minimize", fraction=0.4, min_k=1)
        assert top[0].value == 0.1

    def test_min_k_enforced(self):
        """Validate test min k enforced."""
        trials = self._make_trials([0.1, 0.5, 0.9])
        top = _select_top_k(trials, "maximize", fraction=0.1, min_k=2)
        assert len(top) == 2


class TestCacheKey:
    """Group tests for TestCacheKey."""

    def test_deterministic(self):
        """Validate test deterministic."""
        k1 = compute_cache_key(study_name="study1", last_trial_number=10, dataset_fingerprint="fp123")
        k2 = compute_cache_key(study_name="study1", last_trial_number=10, dataset_fingerprint="fp123")
        assert k1 == k2

    def test_varies_with_trial(self):
        """Validate test varies with trial."""
        k1 = compute_cache_key(study_name="study1", last_trial_number=10, dataset_fingerprint="fp123")
        k2 = compute_cache_key(study_name="study1", last_trial_number=11, dataset_fingerprint="fp123")
        assert k1 != k2

    def test_varies_with_fingerprint(self):
        """Validate test varies with fingerprint."""
        k1 = compute_cache_key(study_name="study1", last_trial_number=10, dataset_fingerprint="fp123")
        k2 = compute_cache_key(study_name="study1", last_trial_number=10, dataset_fingerprint="fp456")
        assert k1 != k2

    def test_none_fingerprint(self):
        """Validate test none fingerprint."""
        k = compute_cache_key(study_name="study1", last_trial_number=10, dataset_fingerprint=None)
        assert isinstance(k, str)
        assert len(k) == 24


def _make_trial_data(
    trial_id: int,
    value: float,
    params: dict,
    state: str = "COMPLETE",
) -> dict:
    return {
        "id": trial_id,
        "number": trial_id,
        "value": value,
        "params": params,
        "state": state,
    }


class TestSearchSpaceAdvisorInsufficientEvidence:
    """Group tests for TestSearchSpaceAdvisorInsufficientEvidence."""

    def test_fewer_than_min_trials(self):
        """Validate test fewer than min trials."""
        advisor = SearchSpaceAdvisor(config_thresholds={"persistent_cache_enabled": False})
        trials = [_make_trial_data(0, 0.5, {"lr": 0.01})]
        result = advisor.advise(
            search_space={"lr": {"low": 0.001, "high": 0.1}},
            trials_data=trials,
            importances={"lr": 0.5},
            direction="maximize",
            study_name="test",
        )
        assert result["metadata"]["insufficient_evidence"] is True
        assert result["recommendations"] == []

    def test_no_trials(self):
        """Validate test no trials."""
        advisor = SearchSpaceAdvisor(config_thresholds={"persistent_cache_enabled": False})
        result = advisor.advise(
            search_space={"lr": {"low": 0.001, "high": 0.1}},
            trials_data=[],
            importances={},
            direction="maximize",
            study_name="test",
        )
        assert result["metadata"]["insufficient_evidence"] is True


class TestSearchSpaceAdvisorNumeric:
    """Group tests for TestSearchSpaceAdvisorNumeric."""

    def _build_trials_near_upper(self, n: int = 12) -> list[dict]:
        return [_make_trial_data(i, 0.8 + i * 0.01, {"lr": 0.09 + i * 0.001}) for i in range(n)]

    def test_expand_upper(self):
        """Validate test expand upper."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = self._build_trials_near_upper(12)
        result = advisor.advise(
            search_space={"lr": {"low": 0.0, "high": 0.1}},
            trials_data=trials,
            importances={"lr": 0.4},
            direction="maximize",
            study_name="test",
        )
        recs = result["recommendations"]
        assert len(recs) == 1
        assert recs[0]["action"] == "expand_upper"
        assert "new_high" in recs[0]["recommendation"]

    def test_expand_lower(self):
        """Validate test expand lower."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = [_make_trial_data(i, 0.9 - i * 0.01, {"dim": 65 + i}) for i in range(12)]
        result = advisor.advise(
            search_space={"dim": {"low": 64, "high": 512}},
            trials_data=trials,
            importances={"dim": 0.3},
            direction="maximize",
            study_name="test",
        )
        recs = result["recommendations"]
        assert len(recs) == 1
        assert recs[0]["action"] == "expand_lower"

    def test_low_importance_fix(self):
        """Validate test low importance fix."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = [
            _make_trial_data(i, 0.5 + i * 0.01, {"dropout": 0.3 + i * 0.02}) for i in range(12)
        ]
        result = advisor.advise(
            search_space={"dropout": {"low": 0.0, "high": 1.0}},
            trials_data=trials,
            importances={"dropout": 0.01},
            direction="maximize",
            study_name="test",
        )
        recs = result["recommendations"]
        assert len(recs) == 1
        assert recs[0]["action"] in ("fix", "narrow", "keep")

    def test_log_scale_suggestion(self):
        """Validate test log scale suggestion."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = [
            _make_trial_data(i, 0.5 + i * 0.01, {"learning_rate": 0.001 + i * 0.005})
            for i in range(12)
        ]
        result = advisor.advise(
            search_space={"learning_rate": {"low": 0.0001, "high": 1.0}},
            trials_data=trials,
            importances={"learning_rate": 0.02},
            direction="maximize",
            study_name="test",
        )
        recs = result["recommendations"]
        assert len(recs) == 1
        rationale = recs[0]["rationale"]
        assert "log" in rationale.lower() or recs[0]["action"] == "change_distribution"

    def test_stringified_search_space_is_not_dropped(self):
        """Validate recommendations are generated from stringified distributions."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = self._build_trials_near_upper(12)
        result = advisor.advise(
            search_space={
                "lr": (
                    '{"name":"FloatDistribution","attributes":{"low":0.0,"high":0.1,"log":false}}'
                )
            },
            trials_data=trials,
            importances={"lr": 0.4},
            direction="maximize",
            study_name="test_json_str_space",
        )
        recs = result["recommendations"]
        assert len(recs) == 1
        assert recs[0]["param_name"] == "lr"


class TestSearchSpaceAdvisorCategorical:
    """Group tests for TestSearchSpaceAdvisorCategorical."""

    def test_dominant_category(self):
        """Validate test dominant category."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = []
        for i in range(12):
            opt = "adam" if i < 10 else "sgd"
            trials.append(_make_trial_data(i, 0.8 if opt == "adam" else 0.5, {"optimizer": opt}))
        result = advisor.advise(
            search_space={"optimizer": {"choices": ["adam", "sgd", "rmsprop"]}},
            trials_data=trials,
            importances={"optimizer": 0.3},
            direction="maximize",
            study_name="test",
        )
        recs = result["recommendations"]
        assert len(recs) == 1
        assert recs[0]["action"] == "reduce_categories"
        assert "adam" in recs[0]["recommendation"].get("keep", [])

    def test_balanced_categories_keep(self):
        """Validate test balanced categories keep."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        import random

        rng = random.Random(42)
        cats = ["relu", "gelu", "tanh"]
        trials = [
            _make_trial_data(i, 0.5 + rng.random() * 0.01, {"act": cats[i % 3]}) for i in range(12)
        ]
        result = advisor.advise(
            search_space={"act": {"choices": cats}},
            trials_data=trials,
            importances={"act": 0.2},
            direction="maximize",
            study_name="test",
        )
        recs = result["recommendations"]
        assert len(recs) == 1
        assert recs[0]["action"] == "keep"


class TestSearchSpaceAdvisorSotaSignals:
    """Group tests for SOTA signals in advisor output."""

    def test_bootstrap_support_and_interaction_fields(self):
        """Validate bootstrap support + interaction fields are present."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        search_space = {"lr": {"low": 1e-4, "high": 1e-2}}
        trials = [_make_trial_data(i, 0.5 + i * 0.01, {"lr": 1e-4 + i * 1e-4}) for i in range(12)]
        result = advisor.advise(
            search_space=search_space,
            trials_data=trials,
            importances={"lr": 0.2},
            study_name="bootstrap_test",
        )
        rec = result["recommendations"][0]
        assert rec["bootstrap_support"] is None or 0.0 <= rec["bootstrap_support"] <= 1.0
        assert "interaction_strength" in rec

    def test_pruned_trials_counted(self):
        """Validate pruned trials contribute to metadata count."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        search_space = {"lr": {"low": 1e-4, "high": 1e-2}}
        trials = [_make_trial_data(i, 0.4 + i * 0.01, {"lr": 1e-4 + i * 1e-4}) for i in range(5)]
        trials.append(
            {
                "id": 99,
                "number": 99,
                "value": None,
                "params": {"lr": 1e-3},
                "state": "PRUNED",
                "intermediate_values": {1: 0.2, 2: 0.25},
            }
        )
        result = advisor.advise(
            search_space=search_space,
            trials_data=trials,
            importances={"lr": 0.2},
            study_name="pruned_test",
        )
        assert result["metadata"]["n_pruned_trials"] == 1


class TestSearchSpaceAdvisorCache:
    """Group tests for TestSearchSpaceAdvisorCache."""

    def test_cache_hit(self):
        """Validate test cache hit."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = [_make_trial_data(i, 0.5 + i * 0.01, {"lr": 0.05}) for i in range(5)]
        kwargs = dict(
            search_space={"lr": {"low": 0.01, "high": 0.1}},
            trials_data=trials,
            importances={"lr": 0.1},
            direction="maximize",
            study_name="cache_test",
        )
        r1 = advisor.advise(**kwargs)
        assert r1["metadata"]["cache_hit"] is False

        r2 = advisor.advise(**kwargs)
        assert r2["metadata"]["cache_hit"] is True

    def test_cache_invalidated_by_new_trial(self):
        """Validate test cache invalidated by new trial."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = [_make_trial_data(i, 0.5 + i * 0.01, {"lr": 0.05}) for i in range(5)]
        kwargs = dict(
            search_space={"lr": {"low": 0.01, "high": 0.1}},
            trials_data=trials,
            importances={"lr": 0.1},
            direction="maximize",
            study_name="cache_test2",
        )
        r1 = advisor.advise(**kwargs)
        assert r1["metadata"]["cache_hit"] is False

        trials.append(_make_trial_data(5, 0.56, {"lr": 0.06}))
        r2 = advisor.advise(**kwargs)
        assert r2["metadata"]["cache_hit"] is False

    def test_cache_self_heals_empty_recommendations(self):
        """Recompute instead of trusting stale empty cache payloads."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = [_make_trial_data(i, 0.5 + i * 0.01, {"lr": 0.02 + i * 0.01}) for i in range(6)]
        kwargs = dict(
            search_space={"lr": {"low": 0.01, "high": 0.2}},
            trials_data=trials,
            importances={"lr": 0.1},
            direction="maximize",
            study_name="cache_heal_test",
            dataset_fingerprint="fp-heal",
        )
        first = advisor.advise(**kwargs)
        assert len(first["recommendations"]) > 0
        cache_spec = AdvisorCacheSpec(
            study_name="cache_heal_test",
            dataset_fingerprint="fp-heal",
            direction="maximize",
            advisor_version=ADVISOR_VERSION,
            last_trial=5,
            search_space_hash=build_search_space_hash(kwargs["search_space"]),
            objective_schema_hash=build_objective_schema_hash(["maximize"]),
        )
        advisor._advice_cache._l1.set(cache_spec.cache_key(), {
            "recommendations": [],
            "metadata": {
                "insufficient_evidence": False,
                "n_completed_trials": 6,
                "cache_hit": False,
            },
        })
        healed = advisor.advise(**kwargs)
        assert healed["metadata"]["cache_hit"] is False
        assert len(healed["recommendations"]) > 0

    def test_force_recompute_bypasses_cache(self):
        """Force recompute should ignore cache even when key matches."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = [_make_trial_data(i, 0.5 + i * 0.01, {"lr": 0.05}) for i in range(5)]
        kwargs = dict(
            search_space={"lr": {"low": 0.01, "high": 0.1}},
            trials_data=trials,
            importances={"lr": 0.1},
            direction="maximize",
            study_name="cache_force_test",
        )
        first = advisor.advise(**kwargs)
        assert first["metadata"]["cache_hit"] is False

        second = advisor.advise(**kwargs)
        assert second["metadata"]["cache_hit"] is True

        forced = advisor.advise(**kwargs, force_recompute=True)
        assert forced["metadata"]["cache_hit"] is False
        assert forced["metadata"]["forced_recompute"] is True

    def test_sparse_parameter_evidence_downgrades_confidence(self):
        """Confidence should remain low when parameter appears in few trials."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = []
        for idx in range(30):
            params = {"aux": idx}
            if idx < 5:
                params["rare_param"] = 0.9 + idx * 0.01
            trials.append(_make_trial_data(idx, 0.5 + idx * 0.01, params))

        result = advisor.advise(
            search_space={"rare_param": {"low": 0.0, "high": 1.0}},
            trials_data=trials,
            importances={"rare_param": 0.8},
            direction="maximize",
            study_name="sparse_evidence_confidence",
        )

        rec = result["recommendations"][0]
        assert rec["confidence"] == "low"


class TestConfidenceScoreCalibration:
    """Group tests for confidence score calibration behavior."""

    def test_confidence_score_shrinks_with_low_evidence(self):
        """Low evidence should pull bootstrap support closer to neutral prior."""
        rec_low = {
            "confidence": "high",
            "bootstrap_support": 1.0,
            "attempts_summary": {"count": 2},
            "uncertainty": 0.1,
        }
        rec_high = {
            "confidence": "high",
            "bootstrap_support": 1.0,
            "attempts_summary": {"count": 120},
            "uncertainty": 0.1,
        }
        score_low = _compute_confidence_score(rec_low)
        score_high = _compute_confidence_score(rec_high)
        assert score_high > score_low
        assert 0.0 <= score_low <= 1.0
        assert 0.0 <= score_high <= 1.0


class TestSearchSpaceAdvisorMinimize:
    """Group tests for TestSearchSpaceAdvisorMinimize."""

    def test_minimize_direction(self):
        """Validate test minimize direction."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = [_make_trial_data(i, 10.0 - i * 0.5, {"lr": 0.01 + i * 0.001}) for i in range(10)]
        result = advisor.advise(
            search_space={"lr": {"low": 0.01, "high": 0.1}},
            trials_data=trials,
            importances={"lr": 0.3},
            direction="minimize",
            study_name="min_test",
        )
        meta = result["metadata"]
        assert meta["direction"] == "minimize"
        assert meta["insufficient_evidence"] is False


class TestGenerateSearchSpacePatch:
    """Group tests for TestGenerateSearchSpacePatch."""

    def test_keep_produces_no_patch(self):
        """Validate test keep produces no patch."""
        recs = [{"param_name": "x", "action": "keep", "recommendation": {}, "current_space": {}}]
        patch = generate_search_space_patch(recs)
        assert patch == {}

    def test_expand_upper_patch(self):
        """Validate test expand upper patch."""
        recs = [
            {
                "param_name": "lr",
                "action": "expand_upper",
                "recommendation": {"new_high": 0.2, "old_high": 0.1},
                "current_space": {"type": "float", "low": 0.001, "high": 0.1},
            }
        ]
        patch = generate_search_space_patch(recs)
        assert "lr" in patch
        assert patch["lr"]["high"] == 0.2
        assert patch["lr"]["low"] == 0.001

    def test_fix_patch(self):
        """Validate test fix patch."""
        recs = [
            {
                "param_name": "dropout",
                "action": "fix",
                "recommendation": {"fix_value": 0.3},
                "current_space": {"type": "float", "low": 0.0, "high": 1.0},
            }
        ]
        patch = generate_search_space_patch(recs)
        assert patch["dropout"]["type"] == "fixed"
        assert patch["dropout"]["value"] == 0.3

    def test_reduce_categories_patch(self):
        """Validate test reduce categories patch."""
        recs = [
            {
                "param_name": "opt",
                "action": "reduce_categories",
                "recommendation": {"keep": ["adam", "sgd"], "remove": ["rmsprop"]},
                "current_space": {"type": "categorical", "choices": ["adam", "sgd", "rmsprop"]},
            }
        ]
        patch = generate_search_space_patch(recs)
        assert patch["opt"]["choices"] == ["adam", "sgd"]

    def test_narrow_patch(self):
        """Validate test narrow patch."""
        recs = [
            {
                "param_name": "dim",
                "action": "narrow",
                "recommendation": {"new_low": 128, "new_high": 384, "old_low": 64, "old_high": 512},
                "current_space": {"type": "int", "low": 64, "high": 512},
            }
        ]
        patch = generate_search_space_patch(recs)
        assert patch["dim"]["low"] == 128
        assert patch["dim"]["high"] == 384

    def test_change_distribution_patch(self):
        """Validate test change distribution patch."""
        recs = [
            {
                "param_name": "lr",
                "action": "change_distribution",
                "recommendation": {"distribution": "log_uniform", "low": 1e-5, "high": 1e-1},
                "current_space": {"type": "float", "low": 1e-5, "high": 1e-1},
            }
        ]
        patch = generate_search_space_patch(recs)
        assert patch["lr"]["log"] is True


class TestAdvisorMetadata:
    """Group tests for TestAdvisorMetadata."""

    def test_metadata_fields(self):
        """Validate test metadata fields."""
        advisor = SearchSpaceAdvisor(config_thresholds={"persistent_cache_enabled": False})
        trials = [_make_trial_data(i, 0.5 + i * 0.01, {"x": float(i)}) for i in range(5)]
        result = advisor.advise(
            search_space={"x": {"low": 0.0, "high": 10.0}},
            trials_data=trials,
            importances={"x": 0.1},
            direction="maximize",
            study_name="meta_test",
            dataset_fingerprint="abc123",
        )
        meta = result["metadata"]
        assert meta["study_name"] == "meta_test"
        assert meta["dataset_fingerprint"] == "abc123"
        assert meta["advisor_version"] == ADVISOR_VERSION
        assert meta["n_completed_trials"] == 5
        assert isinstance(meta["compute_time_ms"], float)
        assert "cache_hit" in meta
        assert "forced_recompute" in meta
        assert meta["direction_normalized"] == "maximize"
        assert "validation_flags" in meta
        assert "reliability_summary" in meta
        assert "self_audit" in meta


class TestDatasetFingerprintAndHeuristics:
    """Group tests for dataset fingerprint and dataset-based heuristics."""

    def test_compute_dataset_profile_fingerprint_deterministic(self, tmp_path: Path):
        """Validate fingerprint and profile generation from parquet metadata."""
        train = pl.DataFrame({"s": [0, 1, 2], "p": [0, 1, 1], "o": [1, 2, 0]})
        valid = pl.DataFrame({"s": [2], "p": [1], "o": [0]})
        train_path = tmp_path / "train.parquet"
        valid_path = tmp_path / "valid.parquet"
        train.write_parquet(train_path)
        valid.write_parquet(valid_path)

        fp1, profile1 = compute_dataset_profile_fingerprint([train_path, valid_path])
        fp2, profile2 = compute_dataset_profile_fingerprint([train_path, valid_path])
        assert fp1 is not None
        assert fp1 == fp2
        assert profile1 == profile2
        assert profile1 is not None
        assert profile1["n_triples"] == 4
        assert profile1["n_entities"] == 3
        assert profile1["n_relations"] == 2

    def test_low_trial_uses_dataset_heuristics(self):
        """Validate heuristic recommendations are emitted with insufficient trials."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        result = advisor.advise(
            search_space={
                "embedding_dim": {"type": "int", "low": 64, "high": 1024},
                "negative_sample_size": {"type": "int", "low": 16, "high": 512},
            },
            trials_data=[_make_trial_data(0, 0.1, {"embedding_dim": 128})],
            importances={},
            direction="maximize",
            study_name="heuristic_test",
            dataset_fingerprint="fp_1",
            dataset_profile={
                "n_entities": 4096,
                "n_relations": 32,
                "n_triples": 200000,
                "density": 0.002,
            },
        )
        assert result["metadata"]["insufficient_evidence"] is True
        assert result["metadata"]["heuristics_used"] is True
        assert result["recommendations"]
        names = {rec["param_name"] for rec in result["recommendations"]}
        assert "embedding_dim" in names
        assert "negative_sample_size" in names


class TestPrunedTrialsIgnored:
    """Group tests for TestPrunedTrialsIgnored."""

    def test_only_complete_trials_used(self):
        """Validate test only complete trials used."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 2, "persistent_cache_enabled": False})
        trials = [
            _make_trial_data(0, 0.5, {"x": 1.0}, state="COMPLETE"),
            _make_trial_data(1, 0.6, {"x": 2.0}, state="COMPLETE"),
            _make_trial_data(2, None, {"x": 3.0}, state="PRUNED"),
            _make_trial_data(3, 0.1, {"x": 4.0}, state="FAIL"),
        ]
        result = advisor.advise(
            search_space={"x": {"low": 0.0, "high": 10.0}},
            trials_data=trials,
            importances={"x": 0.1},
            direction="maximize",
            study_name="prune_test",
        )
        assert result["metadata"]["n_completed_trials"] == 2


class TestSearchSpaceAdvisorReliabilityFixes:
    """Regression coverage for reliability and safety fixes."""

    def test_direction_is_case_insensitive(self):
        """Uppercase direction should produce the same recommendation as lowercase."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = [_make_trial_data(i, float(i), {"x": i / 19.0}) for i in range(20)]
        search_space = {"x": {"low": 0.0, "high": 1.0}}

        lower = advisor.advise(
            search_space=search_space,
            trials_data=trials,
            importances={"x": 0.4},
            direction="maximize",
            study_name="case_lower",
        )
        upper = advisor.advise(
            search_space=search_space,
            trials_data=trials,
            importances={"x": 0.4},
            direction="MAXIMIZE",
            study_name="case_upper",
        )

        lower_rec = lower["recommendations"][0]
        upper_rec = upper["recommendations"][0]
        assert lower_rec["action"] == upper_rec["action"]
        assert lower_rec["recommendation"] == upper_rec["recommendation"]

    def test_non_positive_lambda_bound_does_not_explode_log_expand(self):
        """Ranges with low<=0 must avoid huge log-space expansions."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = [
            _make_trial_data(i, 0.6 + i * 0.01, {"lambda_logic": 0.03 + i * 0.001})
            for i in range(12)
        ]
        result = advisor.advise(
            search_space={"lambda_logic": {"low": 0.0, "high": 0.05, "log": False}},
            trials_data=trials,
            importances={"lambda_logic": 0.3},
            direction="maximize",
            study_name="lambda_zero_low",
        )

        rec = result["recommendations"][0]
        if rec["action"] == "expand_upper":
            assert rec["recommendation"]["new_high"] > 0.05
            assert rec["recommendation"]["new_high"] < 1.0
        else:
            assert rec["action"] in {"keep", "narrow", "fix", "expand_lower"}

    def test_reduce_categories_partition_is_disjoint_and_type_stable(self):
        """keep/remove must not overlap and should preserve category value types."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = [
            _make_trial_data(0, 0.50, {"embedding_dim": 512}),
            _make_trial_data(1, 0.51, {"embedding_dim": 512}),
            _make_trial_data(2, 0.52, {"embedding_dim": 512}),
            _make_trial_data(3, 0.53, {"embedding_dim": 512}),
            _make_trial_data(4, 0.49, {"embedding_dim": 256}),
            _make_trial_data(5, 0.48, {"embedding_dim": 128}),
        ]
        result = advisor.advise(
            search_space={"embedding_dim": {"choices": [128, 256, 512]}},
            trials_data=trials,
            importances={"embedding_dim": 0.2},
            direction="maximize",
            study_name="cat_partition",
        )
        rec = result["recommendations"][0]
        assert rec["action"] == "reduce_categories"
        keep = rec["recommendation"]["keep"]
        remove = rec["recommendation"]["remove"]
        assert set(map(str, keep)).isdisjoint(set(map(str, remove)))
        assert all(isinstance(value, int) for value in keep)
        assert all(isinstance(value, int) for value in remove)

    def test_generate_patch_skips_invalid_recommendation(self):
        """Invalid recommendations must not be included in generated patches."""
        recs = [
            {
                "param_name": "lambda_logic",
                "action": "expand_upper",
                "recommendation": {"new_high": 11180.0, "old_high": 0.05},
                "current_space": {"type": "float", "low": 0.0, "high": 0.05},
                "validation": {"passed": False, "blocked_reason": "expand_upper_excessive_factor"},
            }
        ]
        patch = generate_search_space_patch(recs)
        assert patch == {}

    def test_directional_expand_is_blocked_without_monotonic_support(self):
        """Edge-only upper expansion should be blocked when global trend is contradictory."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = []
        for trial_id in range(20):
            x_value = trial_id / 19.0
            score = 1.0 - x_value
            if trial_id >= 16:
                score = 1.2 + 0.01 * trial_id
            trials.append(_make_trial_data(trial_id, score, {"x": x_value}))

        result = advisor.advise(
            search_space={"x": {"low": 0.0, "high": 1.0}},
            trials_data=trials,
            importances={"x": 0.4},
            direction="maximize",
            study_name="monotonic_gate",
        )

        rec = result["recommendations"][0]
        assert rec["action"] == "keep"
        assert "Directional expansion blocked" in rec["rationale"]

    def test_directional_expand_blocked_for_low_cardinality_without_monotonic_evidence(self):
        """Low-cardinality parameter with insufficient monotonic evidence should not expand."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = []
        for idx in range(12):
            params = {"aux": idx}
            score = 0.1 + idx * 0.01
            if idx < 5:
                params["num_global_negatives"] = [256, 256, 256, 128, 64][idx]
                score = [0.91, 0.90, 0.89, 0.40, 0.30][idx]
            trials.append(_make_trial_data(idx, score, params))

        result = advisor.advise(
            search_space={"num_global_negatives": {"low": 32, "high": 256}},
            trials_data=trials,
            importances={"num_global_negatives": 0.2},
            direction="maximize",
            study_name="low_cardinality_gate",
        )

        rec = result["recommendations"][0]
        assert rec["action"] in {"keep", "change_distribution"}
        assert "weak monotonic evidence" in rec["rationale"]

    def test_cost_sensitive_expand_upper_requires_strong_monotonic_gain(self):
        """Cost-sensitive params should avoid expand_upper without strong monotonic evidence."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = []
        values = [16, 32, 64, 128] * 6
        scores = [
            0.51,
            0.50,
            0.53,
            0.54,
            0.52,
            0.48,
            0.52,
            0.55,
            0.50,
            0.49,
            0.51,
            0.56,
            0.49,
            0.47,
            0.50,
            0.52,
            0.48,
            0.46,
            0.49,
            0.53,
            0.47,
            0.45,
            0.48,
            0.51,
        ]
        for idx, value in enumerate(values):
            trials.append(
                _make_trial_data(
                    idx,
                    scores[idx],
                    {"num_global_negatives": value, "aux": idx},
                )
            )

        result = advisor.advise(
            search_space={"num_global_negatives": {"low": 16, "high": 128}},
            trials_data=trials,
            importances={"num_global_negatives": 0.09},
            direction="maximize",
            study_name="cost_sensitive_gate",
        )

        rec = result["recommendations"][0]
        assert rec["action"] != "expand_upper"
        assert "cost-sensitive parameter" in rec["rationale"]

    def test_invalid_recommendation_is_downgraded_to_keep(self):
        """Unsafe recommendation should be exposed as keep with blocked_action metadata."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = [
            _make_trial_data(i, 0.9 + i * 0.01, {"kl_weight": 0.04 + i * 0.0005}) for i in range(12)
        ]
        result = advisor.advise(
            search_space={"kl_weight": {"low": 1e-12, "high": 0.05, "log": True}},
            trials_data=trials,
            importances={"kl_weight": 0.3},
            direction="maximize",
            study_name="blocked_action_regression",
        )
        rec = result["recommendations"][0]
        assert rec["action"] == "keep"
        assert rec.get("blocked_action") in {"expand_upper", "expand_lower"}
        assert rec["validation"]["passed"] is False


class TestSearchSpaceAdvisorSelfAudit:
    """Regression coverage for periodic self-audit and directional blocking."""

    @staticmethod
    def _make_regime_shift_trials() -> list[dict]:
        trials = []
        for trial_id in range(18):
            x_value = 0.72 + 0.015 * trial_id
            score = 2.0 + 2.0 * x_value
            trials.append(_make_trial_data(trial_id, score, {"x": x_value}))
        for offset in range(12):
            trial_id = 18 + offset
            x_value = 0.72 + 0.02 * offset
            score = 1.5 - 1.6 * x_value
            trials.append(_make_trial_data(trial_id, score, {"x": x_value}))
        return trials

    def test_self_audit_blocks_directional_villain(self):
        """When periodic backtest shows low directional LB, recommendation is blocked to keep."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        result = advisor.advise(
            search_space={"x": {"low": 0.0, "high": 1.0}},
            trials_data=self._make_regime_shift_trials(),
            importances={"x": 0.5},
            direction="maximize",
            study_name="self_audit_block",
            force_recompute=True,
        )

        recommendation = result["recommendations"][0]
        self_audit = result["metadata"]["self_audit"]
        assert self_audit["enabled"] is True
        assert self_audit["ran"] is True
        assert self_audit["villains_count"] >= 1
        assert recommendation["action"] == "keep"
        assert recommendation.get("blocked_action") == "expand_upper"
        assert recommendation.get("blocked_by") == "self_audit"
        assert "self-audit" in recommendation["rationale"]

    def test_self_audit_reused_between_periods(self):
        """Self-audit should reuse the last report between periodic checkpoints."""
        advisor = SearchSpaceAdvisor(config_thresholds={"min_trials_any": 3, "persistent_cache_enabled": False})
        trials = self._make_regime_shift_trials()
        first = advisor.advise(
            search_space={"x": {"low": 0.0, "high": 1.0}},
            trials_data=trials,
            importances={"x": 0.5},
            direction="maximize",
            study_name="self_audit_reuse",
            force_recompute=True,
        )
        trials.append(_make_trial_data(30, 0.2, {"x": 0.85}))
        second = advisor.advise(
            search_space={"x": {"low": 0.0, "high": 1.0}},
            trials_data=trials,
            importances={"x": 0.5},
            direction="maximize",
            study_name="self_audit_reuse",
        )

        first_audit = first["metadata"]["self_audit"]
        second_audit = second["metadata"]["self_audit"]
        assert first_audit["ran"] is True
        assert second_audit["ran"] is False
        assert second_audit["reused"] is True

from __future__ import annotations

import copy

import pytest

from pff.config import ENSEMBLE_HPO_CONFIG_PATH
from scripts.optimization.shared import get_file_manager
from scripts.optimization.trials.bounds import get_range, load_metric_bounds, normalize_metric
from scripts.optimization.trials.config_loader import get_cached_config, load_trial_constraints
from scripts.optimization.trials.pipeline import TrialEvaluationPipeline


def _build_pipeline_stub(generalization_gap: float) -> TrialEvaluationPipeline:
    """Create a pipeline stub with configurable generalization_gap for penalty tests."""
    pipeline = TrialEvaluationPipeline.__new__(TrialEvaluationPipeline)
    pipeline.file_manager = get_file_manager()
    constraints = load_trial_constraints(pipeline.file_manager)
    pipeline.coverage_gate = constraints["coverage_gate"]
    pipeline.dominance_gate = constraints["dominance_gate"]
    pipeline.params = {"neural_weight": 0.25, "rules_weight": 0.25, "lightgbm_weight": 0.70}
    pipeline.anyburl_metrics = {
        "avg_confidence": 0.6,
        "coverage": 0.4,
        "relation_coverage": 0.35,
        "rules_per_relation": 12.0,
    }
    pipeline.anyburl_classifier_metrics = {}
    pipeline.hybrid_eval_metrics = {}
    pipeline.xgboost_metrics = {}
    pipeline.kge_metrics = {"mrr": 0.3}
    pipeline.lightgbm_metrics = {
        "val_auc": 0.85,
        "pr_auc": 0.65,
        "mcc": 0.3,
        "generalization_gap": generalization_gap,
    }
    pipeline.base_learner_agreement = 0.5
    pipeline.symbolic_contribution_ratio = 0.35
    pipeline.hybrid_contribution_ratio = 0.45
    pipeline.dominance_violation_message = ""
    pipeline.ensemble_summary_metrics = {}
    pipeline.ensemble_ece = None
    pipeline.ensemble_entropy = None
    pipeline.symbolic_params = {}
    pipeline.rule_metadata_lookup = {}
    pipeline.trial_dir = None
    pipeline.trial_number = 0
    pipeline.elapsed_time = 0.0
    return pipeline


def test_high_gap_penalizes_score() -> None:
    """Generalization gap above threshold should penalize composite score."""
    high_gap = _build_pipeline_stub(0.25)
    low_gap = _build_pipeline_stub(0.05)

    high_gap._compute_score()
    low_gap._compute_score()

    assert high_gap.ensemble_metrics["weighted_score"] < low_gap.ensemble_metrics["weighted_score"]
    assert high_gap.ensemble_metrics["generalization_gap_penalty_ratio"] > 0.0
    assert low_gap.ensemble_metrics["generalization_gap_penalty_ratio"] == pytest.approx(0.0)


def test_gap_threshold_derived_from_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Threshold must follow config bound (half of high bound)."""
    pipeline = _build_pipeline_stub(0.20)
    fm = pipeline.file_manager
    config = fm.read(ENSEMBLE_HPO_CONFIG_PATH) or {}
    metric_bounds = config.get("metrics_bounds", {})
    custom_bounds = copy.deepcopy(metric_bounds)
    custom_bounds.setdefault("learner", {}).setdefault("generalization_gap", {})["high"] = 0.30

    monkeypatch.setattr("scripts.optimization.trials.pipeline.load_metric_bounds", lambda _fm: custom_bounds)

    pipeline._compute_score()

    assert pipeline.ensemble_metrics["generalization_gap_threshold"] == pytest.approx(0.15, rel=1e-6)


def test_gap_penalty_coeff_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Penalty coefficient must be read from scoring config."""
    pipeline = _build_pipeline_stub(0.25)
    fm = pipeline.file_manager
    base_config = fm.read(ENSEMBLE_HPO_CONFIG_PATH) or {}
    custom_config = copy.deepcopy(base_config)
    custom_config.setdefault("scoring", {})["generalization_gap_penalty_coeff"] = 0.30

    def _fake_get_cached_config(_path, _fm):
        return custom_config

    monkeypatch.setattr("scripts.optimization.trials.pipeline.get_cached_config", _fake_get_cached_config)

    pipeline._compute_score()

    # Gap=0.25 with high=0.20 → penalty ratio clamps to 1.0
    assert pipeline.ensemble_metrics["generalization_gap_penalty"] == pytest.approx(0.30, rel=1e-3)

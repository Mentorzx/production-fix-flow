from __future__ import annotations

import pytest

from scripts.optimization.trials.pipeline import TrialEvaluationPipeline


def test_synergy_returns_difference() -> None:
    synergy = TrialEvaluationPipeline._compute_neural_symbolic_synergy(0.78, 0.70)
    assert synergy == pytest.approx(0.08)


def test_synergy_none_when_missing_inputs() -> None:
    assert TrialEvaluationPipeline._compute_neural_symbolic_synergy(None, 0.70) is None
    assert TrialEvaluationPipeline._compute_neural_symbolic_synergy(0.70, None) is None


def _build_synergy_pipeline(
    ensemble_f1: float,
    baseline_f1: float,
    *,
    summary_synergy: bool = False,
) -> TrialEvaluationPipeline:
    """Build a minimal pipeline to exercise synergy adjustments."""
    from scripts.optimization.shared import get_file_manager
    from scripts.optimization.trials.config_loader import load_trial_constraints

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
        "rules_per_relation": 10.0,
    }
    pipeline.anyburl_classifier_metrics = {}
    pipeline.hybrid_eval_metrics = {}
    pipeline.xgboost_metrics = {}
    pipeline.kge_metrics = {"mrr": 0.3}
    pipeline.lightgbm_metrics = {
        "val_auc": 0.85,
        "val_f1": baseline_f1,
        "pr_auc": 0.65,
        "mcc": 0.3,
        "generalization_gap": 0.02,
    }
    pipeline.base_learner_agreement = 0.5
    pipeline.symbolic_contribution_ratio = 0.35
    pipeline.hybrid_contribution_ratio = 0.45
    pipeline.dominance_violation_message = ""
    pipeline.ensemble_summary_metrics = {"f1": ensemble_f1}
    if summary_synergy:
        pipeline.ensemble_summary_metrics["neural_symbolic_synergy"] = ensemble_f1 - baseline_f1
    pipeline.ensemble_ece = None
    pipeline.ensemble_entropy = None
    pipeline.symbolic_params = {}
    pipeline.rule_metadata_lookup = {}
    pipeline.trial_dir = None
    pipeline.trial_number = 0
    pipeline.elapsed_time = 0.0
    return pipeline


def test_positive_synergy_gives_clamped_bonus() -> None:
    """Positive synergy should apply capped bonus."""
    pipeline = _build_synergy_pipeline(1.0, 0.0)
    pipeline._compute_score()

    assert pipeline.ensemble_metrics["synergy_adjustment"] == pytest.approx(0.08)


def test_negative_synergy_gives_clamped_penalty() -> None:
    """Negative synergy should apply capped penalty."""
    pipeline = _build_synergy_pipeline(0.1, 0.9)
    pipeline._compute_score()

    assert pipeline.ensemble_metrics["synergy_adjustment"] == pytest.approx(-0.04)


def test_synergy_not_applied_twice() -> None:
    """If synergy already reported in summary metrics, adjustment should be skipped."""
    pipeline = _build_synergy_pipeline(0.8, 0.7, summary_synergy=True)
    pipeline._compute_score()

    assert pipeline.ensemble_metrics["synergy_adjustment"] == 0.0

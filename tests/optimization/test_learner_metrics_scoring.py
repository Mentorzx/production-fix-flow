from __future__ import annotations

import pytest

from pff.config import ENSEMBLE_HPO_CONFIG_PATH
from scripts.optimization.shared import get_file_manager
from scripts.optimization.trials.bounds import blend_scores, get_range, load_metric_bounds, normalize_metric
from scripts.optimization.trials.config_loader import get_cached_config, load_trial_constraints
from scripts.optimization.trials.pipeline import TrialEvaluationPipeline


def _build_pipeline_stub(lightgbm_overrides: dict[str, float] | None = None) -> TrialEvaluationPipeline:
    """Create a lightweight pipeline instance with prefilled metrics for scoring tests."""
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
        "pr_auc": 0.6,
        "mcc": 0.25,
        "generalization_gap": 0.05,
    }
    if lightgbm_overrides:
        pipeline.lightgbm_metrics.update(lightgbm_overrides)
    pipeline.base_learner_agreement = 0.55
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


def test_pr_auc_increases_score() -> None:
    """Higher PR-AUC should increase learner blend proportionally to its weight."""
    pipeline_high = _build_pipeline_stub({"pr_auc": 0.9})
    pipeline_low = _build_pipeline_stub({"pr_auc": 0.5})

    pipeline_high._compute_score()
    pipeline_low._compute_score()

    weights = get_cached_config(ENSEMBLE_HPO_CONFIG_PATH, pipeline_high.file_manager).get("scoring", {}).get(
        "learner_weights", {}
    )
    metric_bounds = load_metric_bounds(pipeline_high.file_manager)
    pr_low, pr_high = get_range(metric_bounds, ["learner", "lgbm_pr_auc"], 0.5, 0.99)
    delta_component = normalize_metric(0.9, low=pr_low, high=pr_high) - normalize_metric(0.5, low=pr_low, high=pr_high)
    total_weight = sum(weights.values()) or 1.0
    expected_delta = delta_component * weights.get("pr_auc", 0.25) / total_weight

    assert pipeline_high.ensemble_metrics["normalized_learner"] > pipeline_low.ensemble_metrics["normalized_learner"]
    assert pipeline_high.ensemble_metrics["normalized_learner"] - pipeline_low.ensemble_metrics["normalized_learner"] == pytest.approx(
        expected_delta, rel=1e-3
    )


def test_mcc_increases_score() -> None:
    """Higher MCC should lift learner blend with configured weight."""
    pipeline_high = _build_pipeline_stub({"mcc": 0.5})
    pipeline_low = _build_pipeline_stub({"mcc": 0.1})

    pipeline_high._compute_score()
    pipeline_low._compute_score()

    weights = get_cached_config(ENSEMBLE_HPO_CONFIG_PATH, pipeline_high.file_manager).get("scoring", {}).get(
        "learner_weights", {}
    )
    metric_bounds = load_metric_bounds(pipeline_high.file_manager)
    mcc_low, mcc_high = get_range(metric_bounds, ["learner", "lgbm_mcc"], 0.0, 0.9)
    delta_component = normalize_metric(0.5, low=mcc_low, high=mcc_high) - normalize_metric(0.1, low=mcc_low, high=mcc_high)
    total_weight = sum(weights.values()) or 1.0
    expected_delta = delta_component * weights.get("mcc", 0.15) / total_weight

    assert pipeline_high.ensemble_metrics["normalized_learner"] > pipeline_low.ensemble_metrics["normalized_learner"]
    assert pipeline_high.ensemble_metrics["normalized_learner"] - pipeline_low.ensemble_metrics["normalized_learner"] == pytest.approx(
        expected_delta, rel=1e-3
    )


def test_learner_weights_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Learner blend must honor weights provided by config instead of hardcoded defaults."""
    pipeline = _build_pipeline_stub({"val_auc": 0.9, "pr_auc": 0.8, "mcc": 0.4})
    metric_bounds = load_metric_bounds(pipeline.file_manager)
    custom_weights = {
        "auc": 0.10,
        "pr_auc": 0.40,
        "mcc": 0.30,
        "hybrid_f1": 0.10,
        "xgb_f1": 0.05,
        "agreement": 0.05,
    }

    def _fake_get_cached_config(_path, _fm):
        return {"scoring": {"learner_weights": custom_weights}}

    monkeypatch.setattr("scripts.optimization.trials.pipeline.get_cached_config", _fake_get_cached_config)

    pipeline._compute_score()

    lgb_low, lgb_high = get_range(metric_bounds, ["learner", "lgbm_auc"], 0.6, 0.99)
    pr_low, pr_high = get_range(metric_bounds, ["learner", "lgbm_pr_auc"], 0.5, 0.99)
    mcc_low, mcc_high = get_range(metric_bounds, ["learner", "lgbm_mcc"], 0.0, 0.9)
    hybrid_low, hybrid_high = get_range(metric_bounds, ["learner", "hybrid_f1"], 0.45, 0.9)
    agreement_low, agreement_high = get_range(metric_bounds, ["learner", "base_learner_agreement"], 0.4, 0.95)

    expected = blend_scores(
        [
            (normalize_metric(pipeline.lightgbm_metrics["val_auc"], low=lgb_low, high=lgb_high), custom_weights["auc"]),
            (normalize_metric(pipeline.lightgbm_metrics["pr_auc"], low=pr_low, high=pr_high), custom_weights["pr_auc"]),
            (normalize_metric(pipeline.lightgbm_metrics["mcc"], low=mcc_low, high=mcc_high), custom_weights["mcc"]),
            (normalize_metric(0.0, low=hybrid_low, high=hybrid_high), custom_weights["hybrid_f1"]),
            (normalize_metric(0.0, low=hybrid_low, high=hybrid_high), custom_weights["xgb_f1"]),
            (
                normalize_metric(pipeline.base_learner_agreement, low=agreement_low, high=agreement_high),
                custom_weights["agreement"],
            ),
        ]
    )

    assert pipeline.ensemble_metrics["normalized_learner"] == pytest.approx(expected, rel=1e-3)

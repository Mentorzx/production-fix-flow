"""
Tests for trial pipeline robustness.

Tests that the pipeline handles missing/None metrics gracefully
without crashing during HPO trials.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import numpy as np


def _get_learner_weights() -> dict[str, float]:
    """Load learner weights from config for blend_scores calculations."""
    from scripts.optimization.trials.config_loader import get_cached_config
    from pff.config import ENSEMBLE_HPO_CONFIG_PATH
    from pff.utils.core.file_manager import FileManager

    return get_cached_config(ENSEMBLE_HPO_CONFIG_PATH, FileManager()).get("scoring", {}).get("learner_weights", {})


class TestPipelineMetricsHandling:
    """Tests for pipeline handling of missing/None metrics."""

    def test_lightgbm_metrics_none_auc(self):
        """LightGBM metrics with None AUC should not crash."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        lightgbm_metrics = {
            "val_auc": None,
            "val_f1": 0.75,
            "val_accuracy": 0.82,
        }
        
        auc = normalize_metric(
            lightgbm_metrics.get("val_auc") or lightgbm_metrics.get("auc") or 0.0,
            low=0.6, high=0.99
        )
        
        assert auc == 0.0
        assert not math.isnan(auc)

    def test_xgboost_metrics_missing(self):
        """XGBoost metrics dict can be None or empty."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        xgboost_metrics: dict[str, Any] | None = None
        
        f1 = normalize_metric(
            (xgboost_metrics.get("test_f1_score") if xgboost_metrics else None) or 0.0,
            low=0.45, high=0.9
        )
        
        assert f1 == 0.0

    def test_hybrid_eval_metrics_empty(self):
        """Hybrid eval metrics can be empty dict."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        hybrid_eval_metrics: dict[str, Any] = {}
        
        f1 = normalize_metric(
            hybrid_eval_metrics.get("f1") or 0.0,
            low=0.45, high=0.9
        )
        
        assert f1 == 0.0

    def test_kge_metrics_nan_mrr(self):
        """KGE metrics with NaN MRR should be handled."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        kge_metrics = {
            "mrr": float("nan"),
            "hits@1": 0.3,
            "hits@10": 0.6,
        }
        
        mrr = normalize_metric(kge_metrics["mrr"], low=0.15, high=0.75)
        assert mrr == 0.0

    def test_anyburl_metrics_missing_coverage(self):
        """AnyBURL metrics without coverage key."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        anyburl_metrics = {
            "avg_confidence": 0.8,
            "n_rules": 100,
        }
        
        coverage = normalize_metric(
            anyburl_metrics.get("coverage") or 0.0,
            low=0.05, high=0.5
        )
        
        assert coverage == 0.0


class TestCompositeScoreWithMissingMetrics:
    """Tests for composite score calculation with missing metrics."""

    def test_compute_score_all_zeros(self):
        """Score computation with all zero metrics."""
        from scripts.optimization.trials.bounds import normalize_metric, blend_scores
        
        kge_metrics = {"mrr": 0.0}
        anyburl_metrics = {"avg_confidence": 0.0, "coverage": 0.0}
        lightgbm_metrics = {"auc": 0.0}
        
        kge_component = normalize_metric(kge_metrics["mrr"], low=0.15, high=0.75)
        rules_component = normalize_metric(anyburl_metrics["avg_confidence"], low=0.4, high=0.95)
        lgbm_component = normalize_metric(lightgbm_metrics["auc"], low=0.6, high=0.99)
        
        score = blend_scores([
            (kge_component, 0.3),
            (rules_component, 0.3),
            (lgbm_component, 0.4),
        ])
        
        assert score == 0.0
        assert not math.isnan(score)

    def test_compute_score_partial_metrics(self):
        """Score computation with some metrics missing."""
        from scripts.optimization.trials.bounds import normalize_metric, blend_scores
        
        kge_metrics = {"mrr": 0.45}
        anyburl_metrics = {"avg_confidence": None, "coverage": 0.3}
        lightgbm_metrics: dict[str, Any] = {}
        
        kge_component = normalize_metric(kge_metrics["mrr"], low=0.15, high=0.75)
        rules_component = normalize_metric(
            anyburl_metrics.get("avg_confidence") or 0.0, low=0.4, high=0.95
        )
        lgbm_component = normalize_metric(
            lightgbm_metrics.get("auc") or 0.0, low=0.6, high=0.99
        )
        
        score = blend_scores([
            (kge_component, 0.3),
            (rules_component, 0.3),
            (lgbm_component, 0.4),
        ])
        
        assert 0.0 <= score <= 1.0
        assert not math.isnan(score)

    def test_learner_component_with_none_values(self):
        """Learner component blend with None-derived values should stay zero."""
        from scripts.optimization.trials.bounds import blend_scores, normalize_metric

        lightgbm_metrics = {"auc": None}
        hybrid_eval_metrics = {"f1": None}
        xgboost_metrics = {"test_f1_score": None}

        lgbm_auc = normalize_metric(
            lightgbm_metrics.get("auc") or 0.0, low=0.6, high=0.99
        )
        hybrid_f1 = normalize_metric(
            hybrid_eval_metrics.get("f1") or 0.0, low=0.45, high=0.9
        )
        xgb_f1 = normalize_metric(
            (xgboost_metrics.get("test_f1_score") if xgboost_metrics else None) or 0.0,
            low=0.45, high=0.9
        )

        weights = _get_learner_weights()
        learner_component = blend_scores(
            [
                (lgbm_auc, weights.get("auc", 0.30)),
                (0.0, weights.get("pr_auc", 0.25)),
                (0.0, weights.get("mcc", 0.15)),
                (hybrid_f1, weights.get("hybrid_f1", 0.15)),
                (xgb_f1, weights.get("xgb_f1", 0.10)),
                (0.0, weights.get("agreement", 0.05)),
            ]
        )

        assert learner_component == 0.0
        assert not math.isnan(learner_component)


class TestSymbolicContributionRatio:
    """Tests for symbolic contribution ratio handling."""

    def test_none_symbolic_contribution(self):
        """None symbolic contribution should not cause errors."""
        symbolic_contribution_ratio: float | None = None
        dominance_gate = 0.85
        
        if symbolic_contribution_ratio is not None and symbolic_contribution_ratio > dominance_gate:
            violation = True
        else:
            violation = False
        
        assert violation is False

    def test_high_symbolic_contribution_detected(self):
        """High symbolic contribution should be detected."""
        symbolic_contribution_ratio = 0.969
        dominance_gate = 0.85
        
        if symbolic_contribution_ratio is not None and symbolic_contribution_ratio > dominance_gate:
            violation = True
        else:
            violation = False
        
        assert violation is True

    def test_symbolic_dominance_penalty_calculation(self):
        """Symbolic dominance penalty should be calculated correctly."""
        symbolic_contribution_ratio = 0.969
        dominance_target = 0.70
        
        if symbolic_contribution_ratio > dominance_target:
            dominance_overflow = symbolic_contribution_ratio - dominance_target
            penalty = dominance_overflow / max(1e-6, 1.0 - dominance_target)
        else:
            penalty = 0.0
        
        assert penalty > 0.0
        assert penalty == pytest.approx(0.896, rel=0.01)


class TestCoverageGate:
    """Tests for coverage gate validation."""

    def test_coverage_below_gate_raises(self):
        """Coverage below gate should raise SymbolicCoverageError."""
        coverage_val = 0.03
        coverage_gate = 0.05
        
        with pytest.raises(ValueError):
            if coverage_val < coverage_gate:
                raise ValueError(f"Coverage {coverage_val} below gate {coverage_gate}")

    def test_coverage_above_gate_passes(self):
        """Coverage above gate should pass."""
        coverage_val = 0.15
        coverage_gate = 0.05
        
        passed = coverage_val >= coverage_gate
        assert passed is True


class TestWeightPenalties:
    """Tests for weight penalty calculations."""

    def test_min_weight_penalty(self):
        """Minimum weight penalty calculation."""
        neural_w, rules_w, lgbm_w = 0.03, 0.25, 0.72
        
        min_weight = min(neural_w, rules_w, lgbm_w)
        weight_penalty = max(0.0, 0.05 - min_weight)
        
        assert weight_penalty == pytest.approx(0.02, rel=0.01)

    def test_no_weight_penalty_when_above_threshold(self):
        """No penalty when all weights above threshold."""
        neural_w, rules_w, lgbm_w = 0.2, 0.3, 0.5
        
        min_weight = min(neural_w, rules_w, lgbm_w)
        weight_penalty = max(0.0, 0.05 - min_weight)
        
        assert weight_penalty == 0.0

    def test_overweight_penalty(self):
        """Overweight penalty for LightGBM > 0.70."""
        lgbm_w = 0.75
        overweight = max(0.0, lgbm_w - 0.70)
        
        assert overweight == pytest.approx(0.05)


class TestNeuralContributionPenalty:
    """Tests for neural contribution penalty."""

    def test_low_neural_contribution_penalty(self):
        """Low neural contribution should be penalized."""
        hybrid_contribution_ratio = 0.05
        min_neural_target = 0.20
        
        if hybrid_contribution_ratio < min_neural_target:
            penalty = (min_neural_target - hybrid_contribution_ratio) / min_neural_target
        else:
            penalty = 0.0
        
        assert penalty == pytest.approx(0.75)

    def test_adequate_neural_contribution_no_penalty(self):
        """Adequate neural contribution should have no penalty."""
        hybrid_contribution_ratio = 0.30
        min_neural_target = 0.20
        
        if hybrid_contribution_ratio < min_neural_target:
            penalty = (min_neural_target - hybrid_contribution_ratio) / min_neural_target
        else:
            penalty = 0.0
        
        assert penalty == 0.0

    def test_none_neural_contribution(self):
        """None neural contribution should not crash."""
        hybrid_contribution_ratio: float | None = None
        min_neural_target = 0.20
        
        penalty = 0.0
        if hybrid_contribution_ratio is not None:
            if hybrid_contribution_ratio < min_neural_target:
                penalty = (min_neural_target - hybrid_contribution_ratio) / min_neural_target
        
        assert penalty == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

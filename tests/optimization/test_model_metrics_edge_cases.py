"""
Tests for model metrics handling.

Ensures all model trainers handle edge cases gracefully:
- None metrics
- Empty results
- NaN values  
- Missing keys
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _get_learner_weights() -> dict[str, float]:
    """Load learner blend weights from config for learner component assertions."""
    from scripts.optimization.trials.config_loader import get_cached_config
    from pff.config import ENSEMBLE_HPO_CONFIG_PATH
    from pff.utils.core.file_manager import FileManager

    return get_cached_config(ENSEMBLE_HPO_CONFIG_PATH, FileManager()).get("scoring", {}).get("learner_weights", {})


class TestRotatEMetricsEdgeCases:
    """Tests for RotatE model metrics handling."""

    def test_none_mrr_normalized_safely(self):
        """MRR of None should normalize to 0."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"mrr": None, "hits@1": 0.3, "hits@10": 0.6}
        mrr = normalize_metric(metrics.get("mrr") or 0.0, low=0.15, high=0.75)
        
        assert mrr == 0.0

    def test_nan_hits_at_1_handled(self):
        """Hits@1 with NaN should normalize to 0."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"mrr": 0.45, "hits@1": float("nan"), "hits@10": 0.6}
        h1 = normalize_metric(metrics["hits@1"], low=0.05, high=0.60)
        
        assert h1 == 0.0

    def test_empty_kge_metrics_dict(self):
        """Empty KGE metrics dict should not crash."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics: dict[str, Any] = {}
        mrr = normalize_metric(metrics.get("mrr") or 0.0, low=0.15, high=0.75)
        
        assert mrr == 0.0

    def test_none_kge_metrics_dict(self):
        """None KGE metrics should be handled."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics: dict[str, Any] | None = None
        mrr = normalize_metric(
            (metrics.get("mrr") if metrics else None) or 0.0,
            low=0.15, high=0.75
        )
        
        assert mrr == 0.0

    def test_very_low_mrr_clamps_to_zero(self):
        """MRR below low bound should clamp to 0."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"mrr": 0.05}  # Below low=0.15
        mrr = normalize_metric(metrics["mrr"], low=0.15, high=0.75)
        
        assert mrr == 0.0

    def test_excellent_mrr_clamps_to_one(self):
        """MRR above high bound should clamp to 1."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"mrr": 0.90}  # Above high=0.75
        mrr = normalize_metric(metrics["mrr"], low=0.15, high=0.75)
        
        assert mrr == 1.0


class TestLightGBMMetricsEdgeCases:
    """Tests for LightGBM model metrics handling."""

    def test_none_auc_normalized_safely(self):
        """AUC of None should normalize to 0."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"auc": None, "val_auc": None, "val_f1": 0.7}
        auc = normalize_metric(
            metrics.get("val_auc") or metrics.get("auc") or 0.0,
            low=0.6, high=0.99
        )
        
        assert auc == 0.0

    def test_nan_f1_handled(self):
        """F1 with NaN should normalize to 0."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"val_f1": float("nan"), "val_auc": 0.85}
        f1 = normalize_metric(metrics["val_f1"], low=0.45, high=0.9)
        
        assert f1 == 0.0

    def test_empty_lightgbm_metrics(self):
        """Empty LightGBM metrics dict should not crash."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics: dict[str, Any] = {}
        auc = normalize_metric(metrics.get("auc") or 0.0, low=0.6, high=0.99)
        
        assert auc == 0.0

    def test_negative_auc_clamps_to_zero(self):
        """Negative AUC (invalid) should clamp to 0."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"auc": -0.1}  # Invalid
        auc = normalize_metric(metrics["auc"], low=0.6, high=0.99)
        
        assert auc == 0.0

    def test_inf_auc_clamps_to_one(self):
        """Infinity AUC should clamp to 1."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"auc": float("inf")}
        auc = normalize_metric(metrics["auc"], low=0.6, high=0.99)
        
        assert auc == 1.0

    def test_neg_inf_auc_clamps_to_zero(self):
        """Negative infinity AUC should clamp to 0."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"auc": float("-inf")}
        auc = normalize_metric(metrics["auc"], low=0.6, high=0.99)
        
        assert auc == 0.0


class TestXGBoostMetricsEdgeCases:
    """Tests for XGBoost model metrics handling."""

    def test_none_test_f1_score(self):
        """None test_f1_score should normalize to 0."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"test_f1_score": None}
        f1 = normalize_metric(
            metrics.get("test_f1_score") or 0.0,
            low=0.45, high=0.9
        )
        
        assert f1 == 0.0

    def test_nan_test_auc(self):
        """NaN test_auc should normalize to 0."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"test_auc": float("nan"), "test_f1_score": 0.75}
        auc = normalize_metric(metrics["test_auc"], low=0.6, high=0.99)
        
        assert auc == 0.0

    def test_missing_required_metric(self):
        """Missing required metric should not crash."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"train_loss": 0.25}  # Missing test_f1_score
        f1 = normalize_metric(
            metrics.get("test_f1_score") or 0.0,
            low=0.45, high=0.9
        )
        
        assert f1 == 0.0

    def test_xgboost_metrics_dict_none(self):
        """XGBoost metrics dict being None should be handled."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        xgboost_metrics: dict[str, Any] | None = None
        f1 = normalize_metric(
            (xgboost_metrics.get("test_f1_score") if xgboost_metrics else None) or 0.0,
            low=0.45, high=0.9
        )
        
        assert f1 == 0.0


class TestAnyBURLMetricsEdgeCases:
    """Tests for AnyBURL rule metrics handling."""

    def test_none_coverage(self):
        """None coverage should normalize to 0."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"coverage": None, "avg_confidence": 0.8}
        coverage = normalize_metric(
            metrics.get("coverage") or 0.0,
            low=0.05, high=0.5
        )
        
        assert coverage == 0.0

    def test_nan_avg_confidence(self):
        """NaN avg_confidence should normalize to 0."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"avg_confidence": float("nan"), "coverage": 0.3}
        conf = normalize_metric(metrics["avg_confidence"], low=0.4, high=0.95)
        
        assert conf == 0.0

    def test_zero_n_rules(self):
        """Zero rules should still allow normalized confidence."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"n_rules": 0, "avg_confidence": 0.0, "coverage": 0.0}
        conf = normalize_metric(metrics["avg_confidence"], low=0.4, high=0.95)
        
        assert conf == 0.0

    def test_empty_anyburl_metrics(self):
        """Empty AnyBURL metrics dict should not crash."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics: dict[str, Any] = {}
        coverage = normalize_metric(metrics.get("coverage") or 0.0, low=0.05, high=0.5)
        conf = normalize_metric(metrics.get("avg_confidence") or 0.0, low=0.4, high=0.95)
        
        assert coverage == 0.0
        assert conf == 0.0


class TestHybridEvalMetricsEdgeCases:
    """Tests for hybrid evaluation metrics handling."""

    def test_none_hybrid_f1(self):
        """None hybrid F1 should normalize to 0."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"f1": None, "precision": 0.8, "recall": 0.7}
        f1 = normalize_metric(metrics.get("f1") or 0.0, low=0.45, high=0.9)
        
        assert f1 == 0.0

    def test_nan_hybrid_precision(self):
        """NaN hybrid precision should normalize to 0."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics = {"precision": float("nan"), "recall": 0.7, "f1": 0.75}
        precision = normalize_metric(metrics["precision"], low=0.5, high=0.95)
        
        assert precision == 0.0

    def test_empty_hybrid_metrics(self):
        """Empty hybrid metrics dict should not crash."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        metrics: dict[str, Any] = {}
        f1 = normalize_metric(metrics.get("f1") or 0.0, low=0.45, high=0.9)
        
        assert f1 == 0.0


class TestLearnerBlendWithEdgeCases:
    """Tests for learner component blending with edge case values."""

    def test_blend_of_all_zeros(self):
        """Blending all zero metrics should return 0.0."""
        from scripts.optimization.trials.bounds import blend_scores, normalize_metric

        lgbm_auc = normalize_metric(0.0, low=0.6, high=0.99)
        hybrid_f1 = normalize_metric(0.0, low=0.45, high=0.9)
        xgb_f1 = normalize_metric(0.0, low=0.45, high=0.9)

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

    def test_blend_with_one_valid(self):
        """Blending with a single valid metric should scale by its weight."""
        from scripts.optimization.trials.bounds import blend_scores, normalize_metric

        lgbm_auc = normalize_metric(None, low=0.6, high=0.99)  # 0.0
        hybrid_f1 = normalize_metric(0.7, low=0.45, high=0.9)  # ~0.56
        xgb_f1 = normalize_metric(None, low=0.45, high=0.9)    # 0.0

        weights = _get_learner_weights()
        total_weight = sum(weights.values()) or 1.0
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

        expected = (hybrid_f1 * weights.get("hybrid_f1", 0.15)) / total_weight
        assert learner_component > 0.0
        assert learner_component == pytest.approx(expected, rel=0.01)

    def test_blend_respects_primary_auc_weight(self):
        """AUC should dominate the learner blend when it is strongest metric."""
        from scripts.optimization.trials.bounds import blend_scores, normalize_metric

        lgbm_auc = normalize_metric(0.95, low=0.6, high=0.99)  # ~0.90
        hybrid_f1 = normalize_metric(0.75, low=0.45, high=0.9)  # ~0.67
        xgb_f1 = normalize_metric(0.80, low=0.45, high=0.9)     # ~0.78

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

        total_weight = sum(weights.values()) or 1.0
        expected = (
            lgbm_auc * weights.get("auc", 0.30)
            + hybrid_f1 * weights.get("hybrid_f1", 0.15)
            + xgb_f1 * weights.get("xgb_f1", 0.10)
        ) / total_weight
        assert learner_component == pytest.approx(expected, rel=0.01)


class TestBlendScoresWithModelMetrics:
    """Tests for blend_scores with realistic model metrics."""

    def test_blend_with_all_models_producing_metrics(self):
        """Blend scores with all models producing valid metrics."""
        from scripts.optimization.trials.bounds import normalize_metric, blend_scores
        
        kge = normalize_metric(0.45, low=0.15, high=0.75)  # ~0.50
        rules = normalize_metric(0.7, low=0.4, high=0.95)   # ~0.55
        lgbm = normalize_metric(0.85, low=0.6, high=0.99)   # ~0.64
        
        score = blend_scores([
            (kge, 0.3),
            (rules, 0.3),
            (lgbm, 0.4),
        ])
        
        assert 0.0 <= score <= 1.0
        assert not math.isnan(score)

    def test_blend_with_one_model_failing(self):
        """Blend scores when one model produces None."""
        from scripts.optimization.trials.bounds import normalize_metric, blend_scores
        
        kge = normalize_metric(0.45, low=0.15, high=0.75)
        rules = normalize_metric(None, low=0.4, high=0.95)  # Failed
        lgbm = normalize_metric(0.85, low=0.6, high=0.99)
        
        score = blend_scores([
            (kge, 0.3),
            (rules, 0.3),
            (lgbm, 0.4),
        ])
        
        # Score should still be computed
        assert score >= 0.0
        assert not math.isnan(score)

    def test_blend_with_all_models_failing(self):
        """Blend scores when all models produce None."""
        from scripts.optimization.trials.bounds import normalize_metric, blend_scores
        
        kge = normalize_metric(None, low=0.15, high=0.75)
        rules = normalize_metric(None, low=0.4, high=0.95)
        lgbm = normalize_metric(None, low=0.6, high=0.99)
        
        score = blend_scores([
            (kge, 0.3),
            (rules, 0.3),
            (lgbm, 0.4),
        ])
        
        assert score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for optimization bounds module.

Tests normalize_metric, blend_scores, get_range, and related functions
with edge cases including None, NaN, infinity, and boundary values.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pff.domain.hpo.bounds import (
    blend_scores,
    get_range,
    normalize_metric,
)
from pff.infrastructure.hpo.config_loader import (
    get_rule_component_weights,
    get_rules_coverage_weight,
    load_metric_bounds,
)


class TestNormalizeMetric:
    """Tests for normalize_metric function."""

    def test_none_returns_zero(self):
        """None value should return 0.0."""
        assert normalize_metric(None, low=0.0, high=1.0) == 0.0
        assert normalize_metric(None, low=0.5, high=0.9) == 0.0
        assert normalize_metric(None, low=-1.0, high=1.0) == 0.0

    def test_nan_returns_zero(self):
        """NaN value should return 0.0."""
        assert normalize_metric(float("nan"), low=0.0, high=1.0) == 0.0
        assert normalize_metric(math.nan, low=0.5, high=0.9) == 0.0

    def test_inf_clamped_to_bounds(self):
        """Infinity should be clamped to [0, 1]."""
        assert normalize_metric(float("inf"), low=0.0, high=1.0) == 1.0
        assert normalize_metric(float("-inf"), low=0.0, high=1.0) == 0.0

    def test_normal_values(self):
        """Normal values should be normalized correctly."""
        assert normalize_metric(0.5, low=0.0, high=1.0) == 0.5
        assert normalize_metric(0.0, low=0.0, high=1.0) == 0.0
        assert normalize_metric(1.0, low=0.0, high=1.0) == 1.0

    def test_value_below_low_clamped_to_zero(self):
        """Values below low should be clamped to 0."""
        assert normalize_metric(-0.5, low=0.0, high=1.0) == 0.0
        assert normalize_metric(0.3, low=0.5, high=1.0) == 0.0

    def test_value_above_high_clamped_to_one(self):
        """Values above high should be clamped to 1."""
        assert normalize_metric(1.5, low=0.0, high=1.0) == 1.0
        assert normalize_metric(0.95, low=0.5, high=0.9) == 1.0

    def test_value_above_high_without_cap_keeps_headroom(self):
        """With cap disabled, values above high should exceed 1.0."""
        result = normalize_metric(0.4, low=0.0, high=0.3, cap=False)
        assert result == pytest.approx(1.333, rel=0.01)
        assert result > 1.0

    def test_value_below_low_without_cap_floored(self):
        """With cap disabled, values below low should still floor at 0."""
        assert normalize_metric(-0.2, low=0.0, high=1.0, cap=False) == 0.0

    def test_inverted_bounds_returns_clamped_value(self):
        """When high <= low, should return clamped value."""
        assert normalize_metric(0.5, low=1.0, high=0.0) == 0.5
        assert normalize_metric(1.5, low=1.0, high=0.0) == 1.0
        assert normalize_metric(-0.5, low=1.0, high=0.0) == 0.0

    def test_equal_bounds_returns_clamped_value(self):
        """When high == low, should return clamped value."""
        assert normalize_metric(0.5, low=0.5, high=0.5) == 0.5
        assert normalize_metric(1.0, low=0.5, high=0.5) == 1.0
        assert normalize_metric(0.0, low=0.5, high=0.5) == 0.0

    def test_different_ranges(self):
        """Test normalization with different ranges."""
        assert normalize_metric(75.0, low=50.0, high=100.0) == 0.5
        assert normalize_metric(0.7, low=0.6, high=0.99) == pytest.approx(0.256, rel=0.01)

    def test_negative_ranges(self):
        """Test normalization with negative ranges."""
        assert normalize_metric(0.0, low=-1.0, high=1.0) == 0.5
        assert normalize_metric(-1.0, low=-1.0, high=1.0) == 0.0
        assert normalize_metric(1.0, low=-1.0, high=1.0) == 1.0


class TestBlendScores:
    """Tests for blend_scores function."""

    def test_empty_scores_returns_zero(self):
        """Empty scores should return 0."""
        assert blend_scores([]) == 0.0

    def test_zero_weights_returns_zero(self):
        """All zero weights should return 0."""
        assert blend_scores([(0.5, 0.0), (0.7, 0.0)]) == 0.0

    def test_negative_weights_ignored(self):
        """Negative weights should be ignored."""
        assert blend_scores([(0.5, -1.0), (0.8, 1.0)]) == 0.8

    def test_nan_values_skipped(self):
        """NaN values should be skipped."""
        result = blend_scores([(float("nan"), 1.0), (0.8, 1.0)])
        assert result == 0.8

    def test_all_nan_returns_zero(self):
        """All NaN values should return 0."""
        assert blend_scores([(float("nan"), 1.0), (float("nan"), 1.0)]) == 0.0

    def test_weighted_average(self):
        """Test weighted average calculation."""
        result = blend_scores([(0.5, 1.0), (1.0, 1.0)])
        assert result == 0.75

        result = blend_scores([(0.5, 2.0), (1.0, 1.0)])
        assert result == pytest.approx(0.666, rel=0.01)

    def test_single_score(self):
        """Single score should return that score."""
        assert blend_scores([(0.7, 1.0)]) == 0.7
        assert blend_scores([(0.3, 5.0)]) == 0.3

    def test_mixed_valid_and_invalid(self):
        """Mix of valid and invalid values."""
        result = blend_scores(
            [
                (float("nan"), 1.0),
                (0.5, 0.0),
                (0.8, 2.0),
                (0.6, 1.0),
            ]
        )
        expected = (0.8 * 2.0 + 0.6 * 1.0) / 3.0
        assert result == pytest.approx(expected, rel=0.001)


class TestGetRange:
    """Tests for get_range function."""

    def test_valid_nested_path(self):
        """Valid nested path should return correct range."""
        bounds = {"kge": {"mrr": {"low": 0.15, "high": 0.75}}}
        low, high = get_range(bounds, ["kge", "mrr"], 0.0, 1.0)
        assert low == 0.15
        assert high == 0.75

    def test_missing_path_returns_defaults(self):
        """Missing path should return defaults."""
        bounds = {"kge": {"mrr": {"low": 0.15, "high": 0.75}}}
        low, high = get_range(bounds, ["missing", "path"], 0.1, 0.9)
        assert low == 0.1
        assert high == 0.9

    def test_empty_bounds_returns_defaults(self):
        """Empty bounds dict should return defaults."""
        low, high = get_range({}, ["kge", "mrr"], 0.2, 0.8)
        assert low == 0.2
        assert high == 0.8

    def test_inverted_bounds_returns_defaults(self):
        """Inverted bounds (high < low) should return defaults."""
        bounds = {"kge": {"mrr": {"low": 0.9, "high": 0.1}}}
        low, high = get_range(bounds, ["kge", "mrr"], 0.15, 0.75)
        assert low == 0.15
        assert high == 0.75

    def test_partial_bounds_uses_defaults(self):
        """Partial bounds should use defaults for missing values."""
        bounds = {"kge": {"mrr": {"low": 0.2}}}
        low, high = get_range(bounds, ["kge", "mrr"], 0.1, 0.9)
        assert low == 0.2
        assert high == 0.9

    def test_non_dict_node_returns_defaults(self):
        """Non-dict node in path should return defaults."""
        bounds = {"kge": "not_a_dict"}
        low, high = get_range(bounds, ["kge", "mrr"], 0.1, 0.9)
        assert low == 0.1
        assert high == 0.9

    def test_none_value_in_path(self):
        """None value in path should return defaults."""
        bounds = {"kge": None}
        low, high = get_range(bounds, ["kge", "mrr"], 0.1, 0.9)
        assert low == 0.1
        assert high == 0.9


class TestLoadMetricBounds:
    """Tests for load_metric_bounds function."""

    def test_returns_dict(self):
        """Should always return a dict."""
        result = load_metric_bounds()
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        """Should have required top-level keys."""
        result = load_metric_bounds()
        assert "kge" in result or "learner" in result or "rules" in result

    def test_with_file_manager(self):
        """Should work with provided FileManager."""
        from pff.shared import FileManager

        fm = FileManager()
        result = load_metric_bounds(fm)
        assert isinstance(result, dict)


class TestGetRulesCoverageWeight:
    """Tests for get_rules_coverage_weight function."""

    def test_returns_float(self):
        """Should return a float."""
        result = get_rules_coverage_weight()
        assert isinstance(result, float)

    def test_clamped_to_valid_range(self):
        """Result should be in [0.15, 0.40]."""
        result = get_rules_coverage_weight()
        assert 0.15 <= result <= 0.40


class TestGetRuleComponentWeights:
    """Tests for get_rule_component_weights function."""

    def test_returns_three_floats(self):
        """Should return tuple of 3 floats."""
        conf, recall, coverage = get_rule_component_weights()
        assert isinstance(conf, float)
        assert isinstance(recall, float)
        assert isinstance(coverage, float)

    def test_weights_sum_to_one(self):
        """Weights should sum to approximately 1.0."""
        conf, recall, coverage = get_rule_component_weights()
        total = conf + recall + coverage
        assert total == pytest.approx(1.0, rel=0.01)

    def test_all_weights_non_negative(self):
        """All weights should be non-negative."""
        conf, recall, coverage = get_rule_component_weights()
        assert conf >= 0.0
        assert recall >= 0.0
        assert coverage >= 0.0


class TestEdgeCasesIntegration:
    """Integration tests for edge cases that caused production bugs."""

    def test_none_metrics_in_pipeline_scenario(self):
        """Simulate pipeline scenario where metrics can be None."""
        metrics = {
            "auc": None,
            "f1": 0.75,
            "precision": None,
            "recall": 0.8,
        }

        auc_norm = normalize_metric(metrics.get("auc") or 0.0, low=0.6, high=0.99)
        f1_norm = normalize_metric(metrics.get("f1") or 0.0, low=0.45, high=0.9)
        precision_norm = normalize_metric(metrics.get("precision") or 0.0, low=0.5, high=0.95)
        recall_norm = normalize_metric(metrics.get("recall") or 0.0, low=0.5, high=0.95)

        assert auc_norm == 0.0
        assert f1_norm == pytest.approx(0.666, rel=0.01)
        assert precision_norm == 0.0
        assert recall_norm == pytest.approx(0.666, rel=0.01)

    def test_empty_metrics_dict(self):
        """Empty metrics dict should not raise errors."""
        metrics: dict[str, Any] = {}

        auc_norm = normalize_metric(metrics.get("auc") or 0.0, low=0.6, high=0.99)
        assert auc_norm == 0.0

    def test_blend_with_none_normalized_values(self):
        """Blend scores should work when some normalized values come from None."""
        metrics = {"auc": None, "f1": 0.75}

        auc_norm = normalize_metric(metrics.get("auc") or 0.0, low=0.6, high=0.99)
        f1_norm = normalize_metric(metrics.get("f1") or 0.0, low=0.45, high=0.9)

        score = blend_scores(
            [
                (auc_norm, 0.5),
                (f1_norm, 0.5),
            ]
        )

        assert score > 0.0
        assert score < 1.0

    def test_all_none_metrics_returns_zero_score(self):
        """All None metrics should result in zero score."""
        metrics = {"auc": None, "f1": None, "precision": None}

        auc_norm = normalize_metric(metrics.get("auc") or 0.0, low=0.6, high=0.99)
        f1_norm = normalize_metric(metrics.get("f1") or 0.0, low=0.45, high=0.9)
        precision_norm = normalize_metric(metrics.get("precision") or 0.0, low=0.5, high=0.95)

        score = blend_scores(
            [
                (auc_norm, 0.3),
                (f1_norm, 0.4),
                (precision_norm, 0.3),
            ]
        )

        assert score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

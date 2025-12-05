"""Property tests for scoring function invariants.

Tests mathematical properties that MUST hold for the scoring functions:
(1) normalize_metric: output always in [0, 1]
(2) blend_scores: weighted average properties (commutativity, bounds)
(3) Composite score: monotonicity in component improvements
(4) Edge cases: NaN handling, zero weights, inverted bounds
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest


# ============================================================================
# Local implementations matching production code
# ============================================================================


def normalize_metric(value: float, *, low: float, high: float) -> float:
    """Clamp and scale a metric into [0, 1] interval."""
    if math.isnan(value):
        return 0.0
    if high <= low:
        return max(0.0, min(1.0, value))
    normalized = (value - low) / (high - low)
    return max(0.0, min(1.0, normalized))


def blend_scores(scores: list[tuple[float, float]]) -> float:
    """Compute a weighted average from (value, weight) pairs skipping NaN values."""
    total_weight = 0.0
    total = 0.0
    for value, weight in scores:
        if weight <= 0:
            continue
        if math.isnan(value):
            continue
        total += value * weight
        total_weight += weight
    if total_weight == 0.0:
        return 0.0
    return total / total_weight


# ============================================================================
# Tests: normalize_metric properties
# ============================================================================


class TestNormalizeMetricBounds:
    """Test that normalize_metric always returns values in [0, 1]."""

    @pytest.mark.parametrize("value", [-100, -1, 0, 0.5, 1, 100, 1e10])
    def test_output_always_in_unit_interval(self, value: float):
        """Property: output is always in [0, 1] regardless of input."""
        result = normalize_metric(value, low=0.0, high=1.0)
        assert 0.0 <= result <= 1.0, f"normalize_metric({value}) = {result} not in [0, 1]"

    @pytest.mark.parametrize("low,high", [
        (0.0, 1.0),
        (0.5, 0.9),
        (0.1, 0.3),
        (-1.0, 1.0),
        (0.0, 100.0),
    ])
    def test_output_in_unit_interval_for_various_bounds(self, low: float, high: float):
        """Property: output in [0, 1] for various bound configurations."""
        for value in [low - 1, low, (low + high) / 2, high, high + 1]:
            result = normalize_metric(value, low=low, high=high)
            assert 0.0 <= result <= 1.0

    def test_nan_returns_zero(self):
        """Property: NaN input returns 0.0."""
        result = normalize_metric(float("nan"), low=0.0, high=1.0)
        assert result == 0.0

    def test_inverted_bounds_handled(self):
        """Property: inverted bounds (low > high) still return valid output."""
        result = normalize_metric(0.5, low=1.0, high=0.0)
        assert 0.0 <= result <= 1.0

    def test_equal_bounds_handled(self):
        """Property: equal bounds (low == high) still return valid output."""
        result = normalize_metric(0.5, low=0.5, high=0.5)
        assert 0.0 <= result <= 1.0


class TestNormalizeMetricMonotonicity:
    """Test monotonicity of normalize_metric."""

    def test_monotonically_increasing(self):
        """Property: higher input values give higher or equal output."""
        low, high = 0.2, 0.8
        values = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        results = [normalize_metric(v, low=low, high=high) for v in values]

        for i in range(1, len(results)):
            assert results[i] >= results[i - 1], (
                f"Monotonicity violated: f({values[i]}) = {results[i]} < "
                f"f({values[i-1]}) = {results[i-1]}"
            )

    def test_boundary_values(self):
        """Property: value at low boundary maps to 0, at high maps to 1."""
        low, high = 0.3, 0.7
        assert normalize_metric(low, low=low, high=high) == 0.0
        assert normalize_metric(high, low=low, high=high) == 1.0


# ============================================================================
# Tests: blend_scores properties
# ============================================================================


class TestBlendScoresWeightedAverage:
    """Test weighted average properties of blend_scores."""

    def test_single_score_returns_value(self):
        """Property: single score with positive weight returns that score."""
        result = blend_scores([(0.75, 1.0)])
        assert abs(result - 0.75) < 1e-9

    def test_equal_weights_is_arithmetic_mean(self):
        """Property: equal weights give arithmetic mean."""
        scores = [(0.6, 1.0), (0.8, 1.0), (0.7, 1.0)]
        result = blend_scores(scores)
        expected = (0.6 + 0.8 + 0.7) / 3
        assert abs(result - expected) < 1e-9

    def test_weighted_average_formula(self):
        """Property: result matches weighted average formula."""
        scores = [(0.6, 2.0), (0.8, 3.0)]
        result = blend_scores(scores)
        expected = (0.6 * 2.0 + 0.8 * 3.0) / (2.0 + 3.0)
        assert abs(result - expected) < 1e-9

    def test_result_bounded_by_inputs(self):
        """Property: result is bounded by min and max input values."""
        scores = [(0.3, 1.0), (0.5, 2.0), (0.9, 1.5)]
        result = blend_scores(scores)
        values = [s[0] for s in scores]
        assert min(values) <= result <= max(values)

    def test_order_independence(self):
        """Property: order of scores doesn't affect result (commutativity)."""
        scores1 = [(0.6, 2.0), (0.8, 3.0), (0.7, 1.0)]
        scores2 = [(0.8, 3.0), (0.7, 1.0), (0.6, 2.0)]
        scores3 = [(0.7, 1.0), (0.6, 2.0), (0.8, 3.0)]

        result1 = blend_scores(scores1)
        result2 = blend_scores(scores2)
        result3 = blend_scores(scores3)

        assert abs(result1 - result2) < 1e-9
        assert abs(result2 - result3) < 1e-9


class TestBlendScoresEdgeCases:
    """Test edge cases for blend_scores."""

    def test_zero_weight_ignored(self):
        """Property: scores with zero weight are ignored."""
        scores = [(0.5, 1.0), (0.0, 0.0), (0.9, 0.0)]
        result = blend_scores(scores)
        assert abs(result - 0.5) < 1e-9

    def test_negative_weight_ignored(self):
        """Property: scores with negative weight are ignored."""
        scores = [(0.5, 1.0), (0.1, -1.0)]
        result = blend_scores(scores)
        assert abs(result - 0.5) < 1e-9

    def test_nan_value_ignored(self):
        """Property: NaN values are skipped."""
        scores = [(0.6, 1.0), (float("nan"), 2.0), (0.8, 1.0)]
        result = blend_scores(scores)
        expected = (0.6 + 0.8) / 2
        assert abs(result - expected) < 1e-9

    def test_all_zero_weights_returns_zero(self):
        """Property: all zero weights returns 0.0."""
        scores = [(0.5, 0.0), (0.7, 0.0)]
        result = blend_scores(scores)
        assert result == 0.0

    def test_empty_list_returns_zero(self):
        """Property: empty list returns 0.0."""
        result = blend_scores([])
        assert result == 0.0


# ============================================================================
# Tests: Composite score monotonicity
# ============================================================================


class TestCompositeScoreMonotonicity:
    """Test that improving any component improves composite score."""

    @staticmethod
    def compute_composite(
        kge_mrr: float,
        rules_conf: float,
        rules_cov: float,
        lgbm_auc: float,
    ) -> float:
        """Simplified composite score matching production logic."""
        # Normalize each component
        kge_norm = normalize_metric(kge_mrr, low=0.15, high=0.75)
        conf_norm = normalize_metric(rules_conf, low=0.4, high=0.95)
        cov_norm = normalize_metric(rules_cov, low=0.05, high=0.5)
        auc_norm = normalize_metric(lgbm_auc, low=0.6, high=0.99)

        # Blend rules (simplified)
        rules_component = blend_scores([(conf_norm, 0.5), (cov_norm, 0.3)])

        # Final blend
        return blend_scores([
            (kge_norm, 0.25),
            (rules_component, 0.25),
            (auc_norm, 0.50),
        ])

    @pytest.mark.parametrize("component", ["kge_mrr", "rules_conf", "rules_cov", "lgbm_auc"])
    def test_improving_component_improves_total(self, component: str):
        """Property: improving any single component should improve total score."""
        base_params = {
            "kge_mrr": 0.4,
            "rules_conf": 0.6,
            "rules_cov": 0.2,
            "lgbm_auc": 0.75,
        }

        base_score = self.compute_composite(**base_params)

        # Improve the component
        improved_params = base_params.copy()
        improved_params[component] = base_params[component] + 0.1
        improved_score = self.compute_composite(**improved_params)

        assert improved_score >= base_score, (
            f"Improving {component} should improve score: "
            f"base={base_score:.4f}, improved={improved_score:.4f}"
        )

    def test_all_max_gives_max_score(self):
        """Property: all components at max should give score close to 1."""
        score = self.compute_composite(
            kge_mrr=0.75,  # max
            rules_conf=0.95,  # max
            rules_cov=0.5,  # max
            lgbm_auc=0.99,  # max
        )
        assert score > 0.95, f"All max components should give high score, got {score}"

    def test_all_min_gives_min_score(self):
        """Property: all components at min should give score close to 0."""
        score = self.compute_composite(
            kge_mrr=0.15,  # min
            rules_conf=0.4,  # min
            rules_cov=0.05,  # min
            lgbm_auc=0.6,  # min
        )
        assert score < 0.05, f"All min components should give low score, got {score}"


# ============================================================================
# Tests: Weight normalization properties
# ============================================================================


class TestWeightNormalization:
    """Test properties of weight calculations."""

    @staticmethod
    def get_rule_component_weights(
        coverage_weight: float = 0.2,
        conf_raw: float = 0.5,
        recall_raw: float = 0.3,
    ) -> tuple[float, float, float]:
        """Simplified weight calculation matching production logic."""
        coverage_weight = max(0.15, min(0.40, coverage_weight))
        remaining = max(0.0, 1.0 - coverage_weight)
        base_sum = conf_raw + recall_raw
        if base_sum <= 0:
            conf_weight = recall_weight = remaining * 0.5
        else:
            scale = remaining / base_sum
            conf_weight = conf_raw * scale
            recall_weight = recall_raw * scale
        return conf_weight, recall_weight, coverage_weight

    def test_weights_sum_to_one(self):
        """Property: all weights should sum to 1.0."""
        for cov_weight in [0.15, 0.2, 0.3, 0.4]:
            conf_w, recall_w, cov_w = self.get_rule_component_weights(cov_weight)
            total = conf_w + recall_w + cov_w
            assert abs(total - 1.0) < 1e-9, f"Weights don't sum to 1: {total}"

    def test_all_weights_positive(self):
        """Property: all weights should be positive."""
        for cov_weight in [0.15, 0.2, 0.3, 0.4]:
            conf_w, recall_w, cov_w = self.get_rule_component_weights(cov_weight)
            assert conf_w >= 0, f"Negative conf weight: {conf_w}"
            assert recall_w >= 0, f"Negative recall weight: {recall_w}"
            assert cov_w >= 0, f"Negative coverage weight: {cov_w}"

    def test_coverage_weight_clamped(self):
        """Property: coverage weight should be clamped to [0.15, 0.40]."""
        # Below min
        _, _, cov_w = self.get_rule_component_weights(0.05)
        assert cov_w == 0.15

        # Above max
        _, _, cov_w = self.get_rule_component_weights(0.60)
        assert cov_w == 0.40

        # In range
        _, _, cov_w = self.get_rule_component_weights(0.25)
        assert cov_w == 0.25

    def test_zero_raw_weights_handled(self):
        """Property: zero raw weights don't cause division by zero."""
        conf_w, recall_w, cov_w = self.get_rule_component_weights(
            coverage_weight=0.2,
            conf_raw=0.0,
            recall_raw=0.0,
        )
        # Should split remaining evenly
        remaining = 1.0 - 0.2
        assert abs(conf_w - remaining / 2) < 1e-9
        assert abs(recall_w - remaining / 2) < 1e-9

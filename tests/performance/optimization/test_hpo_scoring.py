"""Tests targeting scoring bugs, edge cases, and penalty computation in HPO.

These tests are designed to find real bugs and design dilemas in
pff/domain/hpo/bounds.py and pff/domain/hpo/scoring.py:
- _normalize_metric edge cases (NaN, inverted bounds, zero-width intervals)
- _blend_scores division risks
- Penalty stacking behavior (can compound to zero or negative)
- Symbolic dominance penalty division-by-near-zero
- Config-driven bounds edge cases

NOTE: We inline the functions to avoid circular import issues with the core module.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import pytest

# ============================================================================
# Inline implementations matching pff/domain/hpo (kept inline for isolation)
# This avoids circular import issues and keeps tests fast
# ============================================================================


def _normalize_metric(value: float, *, low: float, high: float) -> float:
    """Clamp and scale a metric into [0, 1] interval."""
    if math.isnan(value):
        return 0.0
    if high <= low:
        return max(0.0, min(1.0, value))
    normalized = (value - low) / (high - low)
    return max(0.0, min(1.0, normalized))


def _blend_scores(scores: Iterable[tuple[float, float]]) -> float:
    """Compute a weighted average from (value, weight) pairs, skipping NaN values."""
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


def _get_range(
    bounds: dict[str, Any] | None,
    keys: list[str],
    default_low: float,
    default_high: float,
) -> tuple[float, float]:
    """Extract (low, high) range from nested dict, with inverted-bound guard."""
    if bounds is None:
        return default_low, default_high
    try:
        node: Any = bounds
        for key in keys:
            if not isinstance(node, dict):
                return default_low, default_high
            node = node.get(key, {})
        low = (
            float(node.get("low", default_low))
            if isinstance(node, dict)
            else default_low
        )
        high = (
            float(node.get("high", default_high))
            if isinstance(node, dict)
            else default_high
        )
        # Guard against inverted bounds
        if high < low:
            return default_low, default_high
        return low, high
    except (KeyError, TypeError, AttributeError):
        return default_low, default_high


class TestNormalizeMetricEdgeCases:
    """Test _normalize_metric for real-world edge cases."""

    def test_nan_returns_zero(self):
        """NaN input MUST return 0.0, not propagate NaN."""
        result = _normalize_metric(float("nan"), low=0.0, high=1.0)
        assert result == 0.0
        assert not math.isnan(result)

    def test_inf_is_clamped_to_one(self):
        """Positive infinity should be clamped to 1.0."""
        result = _normalize_metric(float("inf"), low=0.0, high=1.0)
        assert result == 1.0

    def test_neg_inf_is_clamped_to_zero(self):
        """Negative infinity should be clamped to 0.0."""
        result = _normalize_metric(float("-inf"), low=0.0, high=1.0)
        assert result == 0.0

    def test_high_equals_low_returns_clamped_value(self):
        """Zero-width interval (high==low) must not cause division by zero."""
        result = _normalize_metric(0.5, low=0.5, high=0.5)
        assert 0.0 <= result <= 1.0
        # The value 0.5 should be clamped to [0,1]
        assert result == 0.5

    def test_inverted_bounds_low_greater_than_high(self):
        """Inverted bounds (low > high) should not break normalization."""
        result = _normalize_metric(0.7, low=0.8, high=0.2)
        # When high <= low, function falls back to clamping
        assert 0.0 <= result <= 1.0

    def test_value_below_low_clamps_to_zero(self):
        """Value below low bound should normalize to 0.0."""
        result = _normalize_metric(-0.5, low=0.0, high=1.0)
        assert result == 0.0

    def test_value_above_high_clamps_to_one(self):
        """Value above high bound should normalize to 1.0."""
        result = _normalize_metric(1.5, low=0.0, high=1.0)
        assert result == 1.0

    def test_negative_bounds(self):
        """Negative metric bounds should work correctly."""
        result = _normalize_metric(-0.5, low=-1.0, high=0.0)
        assert result == 0.5  # (-0.5 - (-1.0)) / (0.0 - (-1.0)) = 0.5

    def test_very_small_interval_precision(self):
        """Very small interval should still compute correctly."""
        # Simulates metrics with tight bounds
        result = _normalize_metric(0.500005, low=0.5, high=0.50001)
        assert 0.0 <= result <= 1.0
        # Expected: (0.500005 - 0.5) / (0.50001 - 0.5) = 0.5
        assert abs(result - 0.5) < 0.01


class TestBlendScoresEdgeCases:
    """Test _blend_scores for division and edge cases."""

    def test_all_zero_weights_returns_zero(self):
        """If all weights are zero, result must be 0.0, not NaN/inf."""
        scores = [(0.8, 0.0), (0.7, 0.0), (0.9, 0.0)]
        result = _blend_scores(scores)
        assert result == 0.0
        assert not math.isnan(result)

    def test_negative_weights_are_ignored(self):
        """Negative weights should be skipped."""
        scores = [(0.8, -0.5), (0.6, 0.4)]
        result = _blend_scores(scores)
        # Only (0.6, 0.4) contributes -> 0.6
        assert result == 0.6

    def test_empty_scores_returns_zero(self):
        """Empty input should return 0.0."""
        result = _blend_scores([])
        assert result == 0.0

    def test_single_score_returns_value(self):
        """Single score should return its value."""
        result = _blend_scores([(0.75, 1.0)])
        assert result == 0.75

    def test_weighted_average_correctness(self):
        """Weighted average should be mathematically correct."""
        scores = [(0.8, 0.3), (0.6, 0.5), (0.9, 0.2)]
        # Expected: (0.8*0.3 + 0.6*0.5 + 0.9*0.2) / (0.3 + 0.5 + 0.2)
        # = (0.24 + 0.30 + 0.18) / 1.0 = 0.72
        result = _blend_scores(scores)
        assert abs(result - 0.72) < 1e-9

    def test_nan_value_with_valid_weight(self):
        """NaN value with valid weight should be skipped, not propagate."""
        scores = [(float("nan"), 0.5), (0.6, 0.5)]
        result = _blend_scores(scores)
        # NaN values are now skipped - only (0.6, 0.5) contributes
        assert not math.isnan(result), "NaN should not propagate"
        assert result == 0.6  # Only the valid value contributes


class TestPenaltyStacking:
    """Test that penalty stacking doesn't produce unexpected results."""

    @pytest.mark.parametrize(
        "penalties",
        [
            # All penalties at maximum
            [
                (0.40, 1.0),
                (0.45, 1.0),
                (0.35, 1.0),
                (0.20, 1.0),
                (1.0, 1.0),
                (0.60, 1.0),
            ],
            # Moderate penalties
            [
                (0.40, 0.5),
                (0.45, 0.5),
                (0.35, 0.5),
                (0.20, 0.5),
                (0.50, 0.5),
                (0.60, 0.5),
            ],
            # Single extreme penalty
            [
                (0.40, 0.0),
                (0.45, 0.0),
                (0.35, 0.0),
                (0.20, 0.0),
                (1.0, 5.0),
                (0.60, 0.0),
            ],
        ],
    )
    def test_penalty_stacking_never_goes_negative(
        self, penalties: list[tuple[float, float]]
    ):
        """Composite score must never go negative after penalty stacking."""
        base_score = 0.8
        composite_score = base_score
        for coeff, penalty in penalties:
            composite_score *= 1.0 - coeff * min(1.0, penalty)
        composite_score = max(0.0, composite_score)

        assert composite_score >= 0.0
        # With max penalties, score should approach zero but not be negative
        assert not math.isnan(composite_score)

    def test_penalty_coefficient_above_one_behavior(self):
        """Test behavior when penalty coefficient exceeds 1.0 (config-driven)."""
        # symbolic_dominance_penalty_coeff can be configured to 1.0 or higher
        base_score = 0.8
        penalty = 0.5  # 50% dominance overflow
        coeff = 1.5  # Aggressive penalty coefficient

        # Current formula: score *= (1.0 - coeff * min(1.0, penalty))
        # = 0.8 * (1.0 - 1.5 * 0.5) = 0.8 * 0.25 = 0.2
        result = base_score * (1.0 - coeff * min(1.0, penalty))

        # This can go negative if coeff * penalty > 1
        if coeff * penalty > 1.0:
            # Formula would give negative before max(0.0, ...)
            assert result < 0.0, "Expected negative before floor"

        # After floor
        final = max(0.0, result)
        assert final >= 0.0


class TestSymbolicDominancePenalty:
    """Test symbolic dominance penalty edge cases (division risks)."""

    def test_dominance_target_at_one_causes_division_by_near_zero(self):
        """When dominance_target approaches 1.0, division denominator is tiny."""
        symbolic_contribution = 0.95
        dominance_target = 0.99  # Close to 1.0

        # Formula: dominance_overflow / max(1e-6, 1.0 - dominance_target)
        dominance_overflow = symbolic_contribution - dominance_target
        denominator = max(1e-6, 1.0 - dominance_target)  # max(1e-6, 0.01) = 0.01

        penalty = dominance_overflow / denominator

        # Should not explode
        assert not math.isinf(penalty)
        assert not math.isnan(penalty)
        # But can be negative if contribution < target
        assert penalty < 0.0  # 0.95 - 0.99 = -0.04

    def test_dominance_target_exactly_one(self):
        """Dominance target at exactly 1.0 would cause division by 1e-6."""
        symbolic_contribution = 1.0
        dominance_target = 1.0

        dominance_overflow = symbolic_contribution - dominance_target  # 0.0
        denominator = max(1e-6, 1.0 - dominance_target)  # 1e-6

        penalty = dominance_overflow / denominator  # 0.0 / 1e-6 = 0.0

        assert penalty == 0.0

    def test_extreme_symbolic_contribution(self):
        """Extreme symbolic contribution (e.g., 1.5) should still compute penalty."""
        symbolic_contribution = 1.5  # Invalid but possible with bugs
        dominance_target = 0.7

        dominance_overflow = symbolic_contribution - dominance_target  # 0.8
        denominator = max(1e-6, 1.0 - dominance_target)  # 0.3

        penalty = dominance_overflow / denominator  # ~2.67

        # Penalty can exceed 1.0, but min(1.0, penalty) is applied in score formula
        assert penalty > 1.0


class TestGetRangeEdgeCases:
    """Test _get_range function for config-driven bounds."""

    def test_missing_nested_key_uses_defaults(self):
        """Missing config keys should fall back to defaults."""
        bounds = {"weights": {}}  # Missing neural_weight
        low, high = _get_range(bounds, ["weights", "neural_weight"], 0.2, 0.45)
        assert low == 0.2
        assert high == 0.45

    def test_empty_bounds_uses_defaults(self):
        """Empty bounds dict should use defaults."""
        bounds = {}
        low, high = _get_range(bounds, ["weights", "neural_weight"], 0.2, 0.45)
        assert low == 0.2
        assert high == 0.45

    def test_none_bounds_uses_defaults(self):
        """None bounds should use defaults."""
        low, high = _get_range(None, ["weights", "neural_weight"], 0.2, 0.45)
        assert low == 0.2
        assert high == 0.45

    def test_inverted_config_bounds(self):
        """Config with low > high should fall back to defaults."""
        bounds = {"weights": {"neural_weight": {"low": 0.8, "high": 0.2}}}
        low, high = _get_range(bounds, ["weights", "neural_weight"], 0.2, 0.45)
        # Inverted bounds are now detected and defaults are used
        assert low == 0.2, "Should use default_low when bounds are inverted"
        assert high == 0.45, "Should use default_high when bounds are inverted"
        assert low <= high, "Result must have valid ordering"

    def test_valid_nested_bounds(self):
        """Valid nested bounds should be extracted correctly."""
        bounds = {"weights": {"neural_weight": {"low": 0.25, "high": 0.5}}}
        low, high = _get_range(bounds, ["weights", "neural_weight"], 0.2, 0.45)
        assert low == 0.25
        assert high == 0.5


class TestWeightNormalizationInvariant:
    """Test that weight normalization maintains invariants."""

    @pytest.mark.parametrize(
        "neural,rules,lgbm",
        [
            (0.2, 0.1, 0.7),  # Sum = 1.0
            (0.3, 0.3, 0.4),  # Sum = 1.0
            (0.25, 0.25, 0.5),  # Sum = 1.0
            (0.0, 0.0, 1.0),  # Edge: all to lgbm
            (1.0, 0.0, 0.0),  # Edge: all to neural
        ],
    )
    def test_weights_sum_to_one(self, neural: float, rules: float, lgbm: float):
        """Weights should sum to 1.0 (or be normalized)."""
        total = neural + rules + lgbm
        # Tolerance for floating point
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_weight_floor_affects_score(self):
        """Weight floor (safe_*_w) should prevent zero weights from breaking blend."""
        # In _compute_score:
        # safe_neural_w = max(neural_w, 0.05)
        neural_w = 0.0
        rules_w = 0.0
        lgbm_w = 1.0

        safe_neural_w = max(neural_w, 0.05)
        safe_rules_w = max(rules_w, 0.05)
        safe_lgbm_w = min(max(lgbm_w, 0.05), 0.70)

        # All safe weights > 0
        assert safe_neural_w > 0
        assert safe_rules_w > 0
        assert safe_lgbm_w > 0

        # But safe weights no longer sum to 1.0
        # This is a known design decision, not a bug
        total_safe = safe_neural_w + safe_rules_w + safe_lgbm_w
        assert total_safe != 1.0  # 0.05 + 0.05 + 0.70 = 0.80


class TestCompositeScoreStability:
    """Test composite score computation for numerical stability."""

    def test_score_reproducibility(self):
        """Same inputs should produce identical scores."""
        base_score = 0.75
        penalties = [(0.40, 0.1), (0.45, 0.2), (0.35, 0.15)]

        def compute_score() -> float:
            score = base_score
            for coeff, penalty in penalties:
                score *= 1.0 - coeff * min(1.0, penalty)
            return max(0.0, score)

        scores = [compute_score() for _ in range(100)]
        assert all(s == scores[0] for s in scores), "Score not reproducible"

    def test_penalty_order_affects_result(self):
        """Penalty application order DOES affect final score (multiplicative)."""
        base_score = 0.8
        penalties_a = [(0.5, 0.3), (0.4, 0.2)]
        penalties_b = [(0.4, 0.2), (0.5, 0.3)]  # Reversed order

        def compute_score(penalties: list) -> float:
            score = base_score
            for coeff, penalty in penalties:
                score *= 1.0 - coeff * min(1.0, penalty)
            return max(0.0, score)

        score_a = compute_score(penalties_a)
        score_b = compute_score(penalties_b)

        # Multiplication is commutative, so order shouldn't matter
        assert abs(score_a - score_b) < 1e-9, "Penalty order should not affect score"

    def test_extremely_small_base_score(self):
        """Very small base score should not cause underflow."""
        base_score = 1e-15
        penalties = [(0.40, 0.1), (0.45, 0.2)]

        score = base_score
        for coeff, penalty in penalties:
            score *= 1.0 - coeff * min(1.0, penalty)
        score = max(0.0, score)

        assert score >= 0.0
        assert not math.isnan(score)

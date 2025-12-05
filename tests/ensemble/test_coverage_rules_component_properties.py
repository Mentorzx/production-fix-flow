"""Property tests for coverage and rules component scoring.

Tests the following properties:
(1) Increasing coverage (within healthy range) should never decrease rules_component.
(2) Coverage below coverage_gate should trigger penalty or SymbolicCoverageError.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import pytest


# ============================================================================
# Inline scoring functions matching scripts/optimization/core.py
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
    """Compute a weighted average from (value, weight) pairs."""
    total_weight = 0.0
    total = 0.0
    for value, weight in scores:
        if weight <= 0:
            continue
        total += value * weight
        total_weight += weight
    if total_weight == 0.0:
        return 0.0
    return total / total_weight


def _compute_rules_component(
    avg_confidence: float,
    recall: float,
    coverage: float,
    confidence_weight: float = 0.5,
    recall_weight: float = 0.3,
    coverage_weight: float = 0.2,
    conf_bounds: tuple[float, float] = (0.4, 0.95),
    recall_bounds: tuple[float, float] = (0.05, 0.5),
    coverage_bounds: tuple[float, float] = (0.05, 0.5),
) -> float:
    """Compute rules component from confidence, recall, coverage.
    
    Args:
        avg_confidence: Average rule confidence.
        recall: Rule recall metric.
        coverage: Rule coverage metric.
        confidence_weight: Weight for confidence in blend.
        recall_weight: Weight for recall in blend.
        coverage_weight: Weight for coverage in blend.
        conf_bounds: (low, high) bounds for confidence normalization.
        recall_bounds: (low, high) bounds for recall normalization.
        coverage_bounds: (low, high) bounds for coverage normalization.
        
    Returns:
        Blended rules component score.
    """
    conf_component = _normalize_metric(avg_confidence, low=conf_bounds[0], high=conf_bounds[1])
    recall_component = _normalize_metric(recall, low=recall_bounds[0], high=recall_bounds[1])
    cov_component = _normalize_metric(coverage, low=coverage_bounds[0], high=coverage_bounds[1])
    
    return _blend_scores([
        (conf_component, confidence_weight),
        (recall_component, recall_weight),
        (cov_component, coverage_weight),
    ])


def _compute_coverage_penalty(
    coverage: float,
    coverage_gate: float,
) -> float:
    """Compute coverage penalty when below gate.
    
    Args:
        coverage: Actual coverage value.
        coverage_gate: Minimum required coverage.
        
    Returns:
        Penalty value (0 if coverage >= gate).
    """
    return max(0.0, coverage_gate - coverage)


class SymbolicCoverageError(RuntimeError):
    """Raised when coverage is below threshold."""
    pass


def check_coverage_gate(coverage: float, coverage_gate: float) -> None:
    """Check if coverage meets gate, raise if not."""
    if coverage < coverage_gate:
        raise SymbolicCoverageError(
            f"Symbolic coverage {coverage:.3f} below required target {coverage_gate:.3f}"
        )


class TestRulesComponentCoverageMonotonicity:
    """Test that rules_component increases (or stays same) with higher coverage."""

    @pytest.mark.parametrize("conf,recall", [
        (0.7, 0.2),
        (0.8, 0.3),
        (0.6, 0.15),
        (0.9, 0.4),
    ])
    def test_higher_coverage_never_decreases_rules_component(self, conf: float, recall: float):
        """Property: increasing coverage (fixed conf/recall) should never decrease rules_component."""
        coverages = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        
        scores = []
        for cov in coverages:
            score = _compute_rules_component(conf, recall, cov)
            scores.append((cov, score))
        
        # Scores should be monotonically non-decreasing
        for i in range(1, len(scores)):
            assert scores[i][1] >= scores[i-1][1], (
                f"rules_component decreased when coverage increased: "
                f"cov={scores[i-1][0]}, score={scores[i-1][1]} -> "
                f"cov={scores[i][0]}, score={scores[i][1]}"
            )

    def test_coverage_weight_affects_sensitivity(self):
        """Property: higher coverage_weight makes rules_component more sensitive to coverage."""
        conf, recall = 0.7, 0.2
        cov_low, cov_high = 0.1, 0.4
        
        # With low coverage_weight (0.1), coverage change has small effect
        score_low_cov_lowweight = _compute_rules_component(conf, recall, cov_low, coverage_weight=0.1)
        score_high_cov_lowweight = _compute_rules_component(conf, recall, cov_high, coverage_weight=0.1)
        diff_lowweight = score_high_cov_lowweight - score_low_cov_lowweight
        
        # With high coverage_weight (0.4), coverage change has larger effect
        score_low_cov_highweight = _compute_rules_component(conf, recall, cov_low, coverage_weight=0.4)
        score_high_cov_highweight = _compute_rules_component(conf, recall, cov_high, coverage_weight=0.4)
        diff_highweight = score_high_cov_highweight - score_low_cov_highweight
        
        assert diff_highweight > diff_lowweight, (
            f"Higher coverage_weight should increase sensitivity: "
            f"diff_high={diff_highweight:.4f}, diff_low={diff_lowweight:.4f}"
        )


class TestCoverageGateBehavior:
    """Test coverage gate penalty and error behavior."""

    def test_coverage_below_gate_triggers_error(self):
        """Property: coverage below gate should raise SymbolicCoverageError."""
        coverage = 0.03
        gate = 0.05
        
        with pytest.raises(SymbolicCoverageError) as exc_info:
            check_coverage_gate(coverage, gate)
        
        assert "below required target" in str(exc_info.value).lower()

    def test_coverage_at_gate_no_error(self):
        """Property: coverage at gate should NOT raise."""
        coverage = 0.05
        gate = 0.05
        
        # Should not raise
        check_coverage_gate(coverage, gate)

    def test_coverage_above_gate_no_error(self):
        """Property: coverage above gate should NOT raise."""
        coverage = 0.10
        gate = 0.05
        
        # Should not raise
        check_coverage_gate(coverage, gate)

    @pytest.mark.parametrize("coverage,gate,expected_penalty", [
        (0.03, 0.05, 0.02),   # Below gate
        (0.04, 0.05, 0.01),   # Just below gate
        (0.05, 0.05, 0.0),    # At gate
        (0.10, 0.05, 0.0),    # Above gate
        (0.30, 0.05, 0.0),    # Well above gate
    ])
    def test_coverage_penalty_values(self, coverage: float, gate: float, expected_penalty: float):
        """Property: coverage penalty is max(0, gate - coverage)."""
        penalty = _compute_coverage_penalty(coverage, gate)
        assert abs(penalty - expected_penalty) < 1e-9, (
            f"Expected penalty {expected_penalty}, got {penalty}"
        )


class TestRulesComponentWithSyntheticMetrics:
    """Test rules_component with synthetic AnyBURL metrics."""

    def test_same_precision_different_coverage_no_penalty_for_higher(self):
        """Property: given same precision, higher coverage should not be penalized more."""
        # Two scenarios with same confidence/recall but different coverage
        metrics_low_cov = {"avg_confidence": 0.75, "recall": 0.25, "coverage": 0.15}
        metrics_high_cov = {"avg_confidence": 0.75, "recall": 0.25, "coverage": 0.35}
        
        score_low = _compute_rules_component(**metrics_low_cov)
        score_high = _compute_rules_component(**metrics_high_cov)
        
        assert score_high >= score_low, (
            f"Higher coverage should not decrease score: "
            f"low_cov={score_low:.4f}, high_cov={score_high:.4f}"
        )

    def test_coverage_at_bounds_normalizes_correctly(self):
        """Property: coverage at bounds should normalize to 0 or 1."""
        conf, recall = 0.7, 0.2
        cov_low, cov_high = 0.05, 0.5  # Default bounds
        
        # At low bound
        score_at_low = _compute_rules_component(
            conf, recall, cov_low, coverage_bounds=(cov_low, cov_high)
        )
        # At high bound
        score_at_high = _compute_rules_component(
            conf, recall, cov_high, coverage_bounds=(cov_low, cov_high)
        )
        
        # Coverage contribution at low bound should be 0
        # But conf/recall still contribute, so total > 0
        assert score_at_low >= 0
        
        # Coverage contribution at high bound should be 1
        assert score_at_high > score_at_low

    @pytest.mark.parametrize("conf_w,recall_w,cov_w", [
        (0.5, 0.3, 0.2),   # Default from config
        (0.4, 0.4, 0.2),   # Equal conf/recall
        (0.3, 0.3, 0.4),   # High coverage weight
    ])
    def test_weights_sum_does_not_need_to_be_one(self, conf_w: float, recall_w: float, cov_w: float):
        """Property: blend function handles arbitrary weights (normalized internally)."""
        conf, recall, cov = 0.7, 0.25, 0.3
        
        # Should not raise regardless of weight sum
        score = _compute_rules_component(
            conf, recall, cov,
            confidence_weight=conf_w,
            recall_weight=recall_w,
            coverage_weight=cov_w,
        )
        
        assert 0.0 <= score <= 1.0, f"Score should be in [0,1], got {score}"


class TestRulesComponentEdgeCases:
    """Test edge cases in rules component computation."""

    def test_zero_coverage_gives_minimum_coverage_contribution(self):
        """Property: zero coverage should give 0 for coverage component."""
        score_zero_cov = _compute_rules_component(0.7, 0.25, 0.0)
        score_max_cov = _compute_rules_component(0.7, 0.25, 0.5)
        
        assert score_zero_cov < score_max_cov

    def test_all_metrics_at_max(self):
        """Property: all metrics at max should give score near 1."""
        score = _compute_rules_component(0.95, 0.5, 0.5)
        # All normalized to ~1.0, blend should be ~1.0
        assert score > 0.9, f"All max metrics should give high score, got {score}"

    def test_all_metrics_at_min(self):
        """Property: all metrics at min should give score near 0."""
        score = _compute_rules_component(0.4, 0.05, 0.05)
        # All normalized to ~0.0, blend should be ~0.0
        assert score < 0.1, f"All min metrics should give low score, got {score}"

    def test_coverage_beyond_bounds_clamped(self):
        """Property: coverage beyond normalization bounds should be clamped."""
        conf, recall = 0.7, 0.25
        
        # Coverage way above high bound
        score_above = _compute_rules_component(conf, recall, 1.0)
        score_at_high = _compute_rules_component(conf, recall, 0.5)
        
        # Both should have coverage component = 1.0 (clamped)
        assert score_above == score_at_high, (
            f"Coverage above bound should clamp to same as at bound"
        )

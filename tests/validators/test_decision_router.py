"""
Tests for Decision Router.

This module tests the routing logic that combines symbolic
and neural predictions based on confidence thresholds.

Test Categories:
    - Individual routing decisions
    - Threshold boundary behavior
    - Batch routing and statistics
    - Vectorized routing performance
    - Config loading
"""

import numpy as np
import pytest

from pff.validators.ensembles.hierarchical.config_loader import DecisionRouterConfig
from pff.validators.ensembles.hierarchical.decision_router import (
    DecisionRouter,
    RoutingDecision,
    RoutingResult,
    RoutingStatistics,
)


class TestRoutingDecisions:
    """Tests for individual routing decision logic."""

    def test_high_symbolic_means_symbolic_decides(self):
        """Test that high symbolic confidence leads to SYMBOLIC_DECIDES."""
        router = DecisionRouter(symbolic_threshold=0.85)
        result = router.route(symbolic_score=0.90, neural_score=0.50)
        assert result.decision == RoutingDecision.SYMBOLIC_DECIDES
        assert abs(result.final_score - 0.90) < 0.001

    def test_low_symbolic_high_neural_means_neural_fallback(self):
        """Test NEURAL_FALLBACK when symbolic is low and neural is high."""
        router = DecisionRouter(symbolic_threshold=0.85, symbolic_low_threshold=0.30, neural_threshold=0.70)
        result = router.route(symbolic_score=0.20, neural_score=0.80)
        assert result.decision == RoutingDecision.NEURAL_FALLBACK
        assert abs(result.final_score - 0.80) < 0.001

    def test_medium_scores_means_blend(self):
        """Test BLEND when both scores are medium."""
        router = DecisionRouter(
            symbolic_threshold=0.85,
            neural_threshold=0.70,
            blend_weight_symbolic=0.6,
            blend_weight_neural=0.4,
        )
        result = router.route(symbolic_score=0.60, neural_score=0.50)
        assert result.decision == RoutingDecision.BLEND
        expected = 0.6 * 0.60 + 0.4 * 0.50
        assert abs(result.final_score - expected) < 0.001
        assert result.blend_weights is not None

    def test_symbolic_at_threshold(self):
        """Test behavior exactly at symbolic threshold."""
        router = DecisionRouter(symbolic_threshold=0.85)
        result = router.route(symbolic_score=0.85, neural_score=0.50)
        assert result.decision == RoutingDecision.SYMBOLIC_DECIDES

    def test_symbolic_just_below_threshold(self):
        """Test behavior just below symbolic threshold."""
        router = DecisionRouter(symbolic_threshold=0.85, neural_threshold=0.70, symbolic_low_threshold=0.30)
        result = router.route(symbolic_score=0.84, neural_score=0.50)
        assert result.decision != RoutingDecision.SYMBOLIC_DECIDES

    def test_neural_at_threshold(self):
        """Test behavior exactly at neural threshold."""
        router = DecisionRouter(symbolic_threshold=0.85, neural_threshold=0.70)
        result = router.route(symbolic_score=0.50, neural_score=0.70)
        assert result.decision == RoutingDecision.NEURAL_FALLBACK

    def test_neural_just_below_threshold(self):
        """Test behavior just below neural threshold."""
        router = DecisionRouter(symbolic_threshold=0.85, neural_threshold=0.70)
        result = router.route(symbolic_score=0.50, neural_score=0.69)
        assert result.decision == RoutingDecision.BLEND


class TestBlendWeights:
    """Tests for blend weight behavior."""

    def test_blend_weights_sum_to_one(self):
        """Test that blend weights are normalized."""
        router = DecisionRouter(
            blend_weight_symbolic=3.0,
            blend_weight_neural=7.0,
        )
        result = router.route(symbolic_score=0.50, neural_score=0.40)
        assert result.decision == RoutingDecision.BLEND
        weights = result.blend_weights
        assert weights is not None
        assert abs(weights[0] + weights[1] - 1.0) < 0.001

    def test_custom_blend_weights(self):
        """Test custom blend weight application."""
        router = DecisionRouter(
            symbolic_threshold=0.85,
            neural_threshold=0.70,
            blend_weight_symbolic=0.7,
            blend_weight_neural=0.3,
        )
        result = router.route(symbolic_score=0.60, neural_score=0.40)
        expected = 0.7 * 0.60 + 0.3 * 0.40
        assert abs(result.final_score - expected) < 0.001


class TestRoutingResult:
    """Tests for RoutingResult dataclass."""

    def test_result_contains_input_scores(self):
        """Test that result contains original input scores."""
        router = DecisionRouter()
        result = router.route(symbolic_score=0.75, neural_score=0.65)
        assert result.symbolic_score == 0.75
        assert result.neural_score == 0.65

    def test_result_contains_metadata(self):
        """Test that result contains threshold metadata."""
        router = DecisionRouter(symbolic_threshold=0.90, neural_threshold=0.75)
        result = router.route(symbolic_score=0.50, neural_score=0.50)
        assert result.metadata["symbolic_threshold"] == 0.90
        assert result.metadata["neural_threshold"] == 0.75

    def test_blend_weights_only_for_blend(self):
        """Test that blend_weights is None for non-BLEND decisions."""
        router = DecisionRouter()
        sym_result = router.route(symbolic_score=0.95, neural_score=0.50)
        assert sym_result.blend_weights is None

        neu_result = router.route(symbolic_score=0.50, neural_score=0.90)
        assert neu_result.blend_weights is None


class TestBatchRouting:
    """Tests for batch routing functionality."""

    def test_batch_routing_same_length_arrays(self):
        """Test batch routing with arrays of same length."""
        router = DecisionRouter()
        symbolic = [0.90, 0.50, 0.60]
        neural = [0.60, 0.80, 0.50]

        results, stats = router.route_batch(symbolic, neural)

        assert len(results) == 3
        assert stats.total_decisions == 3
        assert results[0].decision == RoutingDecision.SYMBOLIC_DECIDES
        assert results[1].decision == RoutingDecision.NEURAL_FALLBACK
        assert results[2].decision == RoutingDecision.BLEND

    def test_batch_routing_statistics(self):
        """Test that statistics are computed correctly."""
        router = DecisionRouter()
        symbolic = np.array([0.90, 0.90, 0.50, 0.50, 0.60])
        neural = np.array([0.60, 0.60, 0.80, 0.80, 0.50])

        _, stats = router.route_batch(symbolic, neural)

        assert stats.symbolic_decides_count == 2
        assert stats.neural_fallback_count == 2
        assert stats.blend_count == 1
        assert abs(stats.symbolic_decides_rate - 0.4) < 0.001
        assert abs(stats.neural_fallback_rate - 0.4) < 0.001
        assert abs(stats.blend_rate - 0.2) < 0.001

    def test_batch_mismatched_lengths_raises(self):
        """Test that mismatched array lengths raise error."""
        router = DecisionRouter()
        with pytest.raises(ValueError, match="same length"):
            router.route_batch([0.5, 0.6], [0.7])


class TestVectorizedRouting:
    """Tests for vectorized routing optimization."""

    def test_vectorized_same_results_as_batch(self):
        """Test that vectorized produces same results as batch."""
        router = DecisionRouter()
        symbolic = np.array([0.90, 0.50, 0.60, 0.40])
        neural = np.array([0.60, 0.80, 0.50, 0.30])

        results, _ = router.route_batch(symbolic, neural)
        vec_scores, vec_codes = router.route_vectorized(symbolic, neural)

        for i, result in enumerate(results):
            assert abs(result.final_score - vec_scores[i]) < 0.001
            expected_code = {
                RoutingDecision.SYMBOLIC_DECIDES: 0,
                RoutingDecision.NEURAL_FALLBACK: 1,
                RoutingDecision.BLEND: 2,
            }[result.decision]
            assert vec_codes[i] == expected_code

    def test_vectorized_decision_codes(self):
        """Test decision code mapping."""
        assert DecisionRouter.decision_code_to_enum(0) == RoutingDecision.SYMBOLIC_DECIDES
        assert DecisionRouter.decision_code_to_enum(1) == RoutingDecision.NEURAL_FALLBACK
        assert DecisionRouter.decision_code_to_enum(2) == RoutingDecision.BLEND

    def test_compute_statistics_from_codes(self):
        """Test statistics computation from vectorized results."""
        router = DecisionRouter()
        symbolic = np.array([0.90, 0.50, 0.60])
        neural = np.array([0.60, 0.80, 0.50])

        scores, codes = router.route_vectorized(symbolic, neural)
        stats = router.compute_statistics_from_codes(codes, scores, symbolic, neural)

        assert stats.total_decisions == 3
        assert stats.symbolic_decides_count == 1
        assert stats.neural_fallback_count == 1
        assert stats.blend_count == 1


class TestRoutingStatistics:
    """Tests for RoutingStatistics dataclass."""

    def test_rates_with_zero_decisions(self):
        """Test that rates are 0 when no decisions made."""
        stats = RoutingStatistics(total_decisions=0)
        assert stats.symbolic_decides_rate == 0.0
        assert stats.neural_fallback_rate == 0.0
        assert stats.blend_rate == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = RoutingStatistics(
            total_decisions=10,
            symbolic_decides_count=5,
            neural_fallback_count=3,
            blend_count=2,
            avg_final_score=0.75,
        )
        d = stats.to_dict()
        assert d["total_decisions"] == 10
        assert d["symbolic_decides_rate"] == 0.5
        assert d["neural_fallback_rate"] == 0.3
        assert d["blend_rate"] == 0.2


class TestFromConfig:
    """Tests for configuration loading."""

    def test_from_config_with_config_object(self):
        """Test creating router from DecisionRouterConfig."""
        config = DecisionRouterConfig(
            symbolic_confidence_threshold=0.90,
            neural_confidence_threshold=0.75,
            blend_weight_symbolic=0.7,
            blend_weight_neural=0.3,
        )
        router = DecisionRouter.from_config(config)
        assert router.symbolic_threshold == 0.90
        assert router.neural_threshold == 0.75

    def test_from_config_without_config(self):
        """Test creating router without config (uses defaults from YAML)."""
        router = DecisionRouter.from_config()
        # Defaults now: symbolic_confidence=0.70, neural_confidence=0.50
        assert router.symbolic_threshold == 0.70
        assert router.neural_threshold == 0.50


class TestEdgeCases:
    """Edge case tests for decision routing."""

    def test_zero_scores(self):
        """Test handling of zero scores."""
        router = DecisionRouter()
        result = router.route(symbolic_score=0.0, neural_score=0.0)
        assert result.decision == RoutingDecision.BLEND
        assert result.final_score == 0.0

    def test_perfect_scores(self):
        """Test handling of perfect scores (1.0)."""
        router = DecisionRouter()
        result = router.route(symbolic_score=1.0, neural_score=1.0)
        assert result.decision == RoutingDecision.SYMBOLIC_DECIDES
        assert result.final_score == 1.0

    def test_both_high_prefers_symbolic(self):
        """Test that when both are high, symbolic is preferred."""
        router = DecisionRouter(symbolic_threshold=0.85)
        result = router.route(symbolic_score=0.90, neural_score=0.95)
        assert result.decision == RoutingDecision.SYMBOLIC_DECIDES
        assert abs(result.final_score - 0.90) < 0.001

    def test_large_batch(self):
        """Test routing with large batch for performance."""
        router = DecisionRouter()
        n = 10000
        symbolic = np.random.uniform(0, 1, n)
        neural = np.random.uniform(0, 1, n)

        scores, codes = router.route_vectorized(symbolic, neural)

        assert len(scores) == n
        assert len(codes) == n
        assert all(0 <= s <= 1 for s in scores)

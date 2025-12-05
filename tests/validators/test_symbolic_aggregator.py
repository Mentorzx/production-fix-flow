"""
Tests for Symbolic Aggregator.

This module tests the symbolic aggregation strategies,
with focus on Noisy-OR as the default strategy.

Test Categories:
    - Individual strategy correctness
    - Noisy-OR mathematical properties
    - Factory pattern behavior
    - Batch aggregation
    - Edge cases (empty, single rule, many rules)
"""

import numpy as np
import pytest

from pff.validators.ensembles.hierarchical.symbolic_aggregator import (
    AggregationResult,
    AggregationStrategy,
    MaxConfidenceStrategy,
    MeanStrategy,
    NoisyOrStrategy,
    SymbolicAggregator,
    SymbolicAggregatorFactory,
    VotingStrategy,
    WeightedSumStrategy,
)


class TestNoisyOrStrategy:
    """Tests for Noisy-OR aggregation strategy."""

    def test_single_rule_returns_confidence(self):
        """Test that single rule returns its confidence."""
        strategy = NoisyOrStrategy()
        result = strategy.aggregate(np.array([0.8]))
        assert abs(result - 0.8) < 0.01

    def test_two_rules_accumulate_evidence(self):
        """Test that two rules accumulate evidence correctly."""
        strategy = NoisyOrStrategy()
        result = strategy.aggregate(np.array([0.5, 0.5]))
        expected = 1 - (0.5 * 0.5)
        assert abs(result - expected) < 0.001

    def test_three_rules_example_from_docs(self):
        """Test the example from docstring: 3 rules @ 0.5 → 0.875."""
        strategy = NoisyOrStrategy()
        result = strategy.aggregate(np.array([0.5, 0.5, 0.5]))
        expected = 1 - (0.5 * 0.5 * 0.5)
        assert abs(result - expected) < 0.001

    def test_high_confidence_single_rule(self):
        """Test that single rule with high confidence dominates."""
        strategy = NoisyOrStrategy()
        result = strategy.aggregate(np.array([0.99]))
        assert result > 0.98

    def test_many_weak_rules_accumulate(self):
        """Test that many weak rules accumulate to strong confidence."""
        strategy = NoisyOrStrategy()
        weak_rules = np.array([0.2] * 10)
        result = strategy.aggregate(weak_rules)
        assert result > 0.85

    def test_empty_array_returns_base_confidence(self):
        """Test that empty input returns base confidence."""
        strategy = NoisyOrStrategy(base_confidence=0.01)
        result = strategy.aggregate(np.array([]))
        assert result == 0.01

    def test_perfect_confidence_returns_one(self):
        """Test that single rule with 1.0 returns ~1.0."""
        strategy = NoisyOrStrategy()
        result = strategy.aggregate(np.array([0.999]))
        assert result > 0.99

    def test_monotonicity(self):
        """Test that adding more rules never decreases confidence."""
        strategy = NoisyOrStrategy()
        r1 = strategy.aggregate(np.array([0.5]))
        r2 = strategy.aggregate(np.array([0.5, 0.3]))
        r3 = strategy.aggregate(np.array([0.5, 0.3, 0.2]))
        assert r1 <= r2 <= r3

    def test_order_independence(self):
        """Test that order of rules doesn't matter."""
        strategy = NoisyOrStrategy()
        r1 = strategy.aggregate(np.array([0.8, 0.3, 0.5]))
        r2 = strategy.aggregate(np.array([0.3, 0.5, 0.8]))
        r3 = strategy.aggregate(np.array([0.5, 0.8, 0.3]))
        assert abs(r1 - r2) < 0.001
        assert abs(r2 - r3) < 0.001


class TestMaxConfidenceStrategy:
    """Tests for Max Confidence strategy."""

    def test_returns_maximum(self):
        """Test that max is returned."""
        strategy = MaxConfidenceStrategy()
        result = strategy.aggregate(np.array([0.3, 0.8, 0.5]))
        assert result == 0.8

    def test_single_rule(self):
        """Test single rule case."""
        strategy = MaxConfidenceStrategy()
        result = strategy.aggregate(np.array([0.6]))
        assert result == 0.6

    def test_empty_returns_zero(self):
        """Test empty input returns zero."""
        strategy = MaxConfidenceStrategy()
        result = strategy.aggregate(np.array([]))
        assert result == 0.0


class TestWeightedSumStrategy:
    """Tests for Weighted Sum strategy."""

    def test_uniform_weights_equals_mean(self):
        """Test that uniform weights produce mean-like result."""
        strategy = WeightedSumStrategy(normalize=True)
        result = strategy.aggregate(np.array([0.2, 0.4, 0.6]))
        expected = 0.4
        assert abs(result - expected) < 0.001

    def test_custom_weights(self):
        """Test custom weight application."""
        strategy = WeightedSumStrategy(normalize=True)
        result = strategy.aggregate(
            np.array([0.8, 0.2]),
            weights=np.array([3.0, 1.0]),
        )
        expected = (0.8 * 0.75) + (0.2 * 0.25)
        assert abs(result - expected) < 0.001

    def test_cap_applied(self):
        """Test that cap is applied to result."""
        strategy = WeightedSumStrategy(normalize=False, cap=0.9)
        result = strategy.aggregate(np.array([1.0, 1.0, 1.0]))
        assert result == 0.9


class TestVotingStrategy:
    """Tests for Voting strategy."""

    def test_all_above_threshold(self):
        """Test all rules above threshold → 1.0."""
        strategy = VotingStrategy(threshold=0.3)
        result = strategy.aggregate(np.array([0.5, 0.6, 0.7]))
        assert result == 1.0

    def test_none_above_threshold(self):
        """Test no rules above threshold → 0.0."""
        strategy = VotingStrategy(threshold=0.5)
        result = strategy.aggregate(np.array([0.1, 0.2, 0.3]))
        assert result == 0.0

    def test_partial_votes(self):
        """Test partial voting."""
        strategy = VotingStrategy(threshold=0.5)
        result = strategy.aggregate(np.array([0.3, 0.6, 0.8, 0.4]))
        expected = 2 / 4
        assert result == expected


class TestMeanStrategy:
    """Tests for Mean strategy."""

    def test_arithmetic_mean(self):
        """Test correct mean calculation."""
        strategy = MeanStrategy()
        result = strategy.aggregate(np.array([0.2, 0.4, 0.6]))
        assert abs(result - 0.4) < 0.001


class TestSymbolicAggregatorParams:
    """Tests for SymbolicAggregator parameter handling."""

    def test_params_override_limits(self):
        """max_rules/min_confidence from params should be applied and not forwarded."""
        aggregator = SymbolicAggregator(
            params={
                "base_confidence": 0.02,
                "max_rules": 3,
                "min_confidence": 0.5,
                "cap": 0.95,
            }
        )

        confidences = [0.9, 0.8, 0.4, 0.2]
        result = aggregator.aggregate_single(confidences)

        assert aggregator.max_rules == 3
        assert abs(aggregator.min_confidence - 0.5) < 1e-6
        assert result.num_rules_fired == 2  # 0.9 and 0.8 survive the threshold
        assert result.metadata["capped_by_max_rules"] == 0


class TestSymbolicAggregatorFactory:
    """Tests for the aggregator factory."""

    def test_create_noisy_or(self):
        """Test factory creates Noisy-OR."""
        strategy = SymbolicAggregatorFactory.create("noisy_or")
        assert strategy.name == "noisy_or"
        assert isinstance(strategy, NoisyOrStrategy)

    def test_create_with_enum(self):
        """Test factory works with enum."""
        strategy = SymbolicAggregatorFactory.create(AggregationStrategy.MAX_CONFIDENCE)
        assert strategy.name == "max_confidence"

    def test_create_with_params(self):
        """Test factory passes params."""
        strategy = SymbolicAggregatorFactory.create(
            "noisy_or", params={"base_confidence": 0.05}
        )
        assert strategy.base_confidence == 0.05

    def test_unknown_strategy_raises(self):
        """Test unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            SymbolicAggregatorFactory.create("unknown_strategy")

    def test_available_strategies(self):
        """Test listing available strategies."""
        available = SymbolicAggregatorFactory.available_strategies()
        assert "noisy_or" in available
        assert "max_confidence" in available
        assert "weighted_sum" in available
        assert "voting" in available
        assert "mean" in available
        assert "pc" in available


class TestSymbolicAggregator:
    """Tests for the main SymbolicAggregator class."""

    def test_default_strategy_is_noisy_or(self):
        """Test that default strategy is Noisy-OR."""
        aggregator = SymbolicAggregator()
        assert aggregator.strategy_name == "noisy_or"

    def test_aggregate_single_returns_result(self):
        """Test single aggregation returns AggregationResult."""
        aggregator = SymbolicAggregator()
        result = aggregator.aggregate_single([0.5, 0.6, 0.7])
        assert isinstance(result, AggregationResult)
        assert result.confidence > 0
        assert result.num_rules_fired == 3
        assert result.strategy_used == "noisy_or"

    def test_min_confidence_filter(self):
        """Test that rules below min_confidence are filtered."""
        aggregator = SymbolicAggregator(min_confidence=0.1)
        result = aggregator.aggregate_single([0.05, 0.5, 0.8])
        assert result.num_rules_fired == 2
        assert result.metadata["filtered_by_min_confidence"] == 1

    def test_max_rules_cap(self):
        """Test that max_rules caps the number of rules."""
        aggregator = SymbolicAggregator(max_rules=3)
        result = aggregator.aggregate_single([0.5, 0.6, 0.7, 0.8, 0.9])
        assert result.num_rules_fired == 3
        assert result.metadata["capped_by_max_rules"] == 2

    def test_aggregate_batch(self):
        """Test batch aggregation."""
        aggregator = SymbolicAggregator()
        results = aggregator.aggregate_batch([
            [0.5, 0.6],
            [0.8],
            [0.3, 0.4, 0.5],
        ])
        assert len(results) == 3
        assert all(isinstance(r, AggregationResult) for r in results)

    def test_aggregate_matrix(self):
        """Test matrix aggregation."""
        aggregator = SymbolicAggregator()
        matrix = np.array([
            [0.5, 0.6, 0.0],
            [0.8, 0.0, 0.0],
            [0.3, 0.4, 0.5],
        ])
        results = aggregator.aggregate_matrix(matrix, fill_value=0.0)
        assert len(results) == 3
        assert all(r >= 0 for r in results)


class TestNoisyOrVsOtherStrategies:
    """Comparison tests showing Noisy-OR advantages."""

    def test_noisy_or_accumulates_better_than_max(self):
        """Test that Noisy-OR accumulates evidence better than max."""
        rules = np.array([0.3, 0.3, 0.3, 0.3])

        noisy_or = NoisyOrStrategy()
        max_strat = MaxConfidenceStrategy()

        noisy_result = noisy_or.aggregate(rules)
        max_result = max_strat.aggregate(rules)

        assert noisy_result > max_result
        assert noisy_result > 0.7

    def test_noisy_or_better_than_mean_for_strong_rules(self):
        """Test Noisy-OR doesn't dilute strong evidence like mean."""
        rules = np.array([0.9, 0.9, 0.1])

        noisy_or = NoisyOrStrategy()
        mean_strat = MeanStrategy()

        noisy_result = noisy_or.aggregate(rules)
        mean_result = mean_strat.aggregate(rules)

        assert noisy_result > mean_result
        assert noisy_result > 0.98

    def test_noisy_or_handles_many_weak_rules(self):
        """Test that many weak rules combine to strong prediction."""
        weak_rules = np.array([0.15] * 20)

        noisy_or = NoisyOrStrategy()
        result = noisy_or.aggregate(weak_rules)

        assert result > 0.95


class TestEdgeCases:
    """Edge case tests for symbolic aggregation."""

    def test_empty_input(self):
        """Test empty input handling."""
        aggregator = SymbolicAggregator()
        result = aggregator.aggregate_single([])
        assert result.confidence == 0.01
        assert result.num_rules_fired == 0

    def test_all_zeros_treated_as_no_rules(self):
        """Test zero confidence rules are filtered."""
        aggregator = SymbolicAggregator(min_confidence=0.01)
        result = aggregator.aggregate_single([0.0, 0.0, 0.0])
        assert result.num_rules_fired == 0

    def test_very_small_confidences(self):
        """Test handling of very small confidence values."""
        aggregator = SymbolicAggregator(min_confidence=0.001)
        result = aggregator.aggregate_single([0.001, 0.002, 0.003])
        assert result.confidence > 0
        assert result.num_rules_fired == 3

    def test_mixed_zero_and_nonzero(self):
        """Test mixed zero and non-zero confidences."""
        aggregator = SymbolicAggregator(min_confidence=0.01)
        result = aggregator.aggregate_single([0.0, 0.5, 0.0, 0.6])
        assert result.num_rules_fired == 2

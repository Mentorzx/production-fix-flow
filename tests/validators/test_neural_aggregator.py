"""
Tests for Neural Aggregator.

This module tests the neural aggregation strategies,
with focus on weighted average as the default strategy.

Test Categories:
    - Individual strategy correctness
    - Mathematical properties
    - Factory pattern behavior
    - Batch aggregation
    - Edge cases (empty, single model, many models)
"""

import numpy as np
import pytest

from pff.validators.ensembles.hierarchical.neural_aggregator import (
    GeometricMeanStrategy,
    HarmonicMeanStrategy,
    MaxScoreStrategy,
    NeuralAggregationResult,
    NeuralAggregationStrategy,
    NeuralAggregator,
    NeuralAggregatorFactory,
    SoftmaxStrategy,
    WeightedAverageStrategy,
)


class TestWeightedAverageStrategy:
    """Tests for Weighted Average strategy."""

    def test_single_score_passthrough(self):
        """Test that single score is returned as-is."""
        strategy = WeightedAverageStrategy()
        result = strategy.aggregate(np.array([0.8]))
        assert abs(result - 0.8) < 0.001

    def test_uniform_weights_is_mean(self):
        """Test that uniform weights produce arithmetic mean."""
        strategy = WeightedAverageStrategy()
        result = strategy.aggregate(np.array([0.2, 0.4, 0.6]))
        expected = 0.4
        assert abs(result - expected) < 0.001

    def test_custom_weights(self):
        """Test custom weight application."""
        strategy = WeightedAverageStrategy(normalize_weights=True)
        result = strategy.aggregate(
            np.array([0.8, 0.2]),
            weights=np.array([3.0, 1.0]),
        )
        expected = (0.8 * 0.75) + (0.2 * 0.25)
        assert abs(result - expected) < 0.001

    def test_empty_returns_zero(self):
        """Test empty input returns zero."""
        strategy = WeightedAverageStrategy()
        result = strategy.aggregate(np.array([]))
        assert result == 0.0


class TestMaxScoreStrategy:
    """Tests for Max Score strategy."""

    def test_returns_maximum(self):
        """Test that max is returned."""
        strategy = MaxScoreStrategy()
        result = strategy.aggregate(np.array([0.3, 0.8, 0.5]))
        assert result == 0.8

    def test_single_score(self):
        """Test single score case."""
        strategy = MaxScoreStrategy()
        result = strategy.aggregate(np.array([0.6]))
        assert result == 0.6

    def test_empty_returns_zero(self):
        """Test empty input returns zero."""
        strategy = MaxScoreStrategy()
        result = strategy.aggregate(np.array([]))
        assert result == 0.0


class TestSoftmaxStrategy:
    """Tests for Softmax strategy."""

    def test_equal_scores_uniform(self):
        """Test that equal scores result in mean-like behavior."""
        strategy = SoftmaxStrategy(temperature=1.0)
        result = strategy.aggregate(np.array([0.5, 0.5, 0.5]))
        assert abs(result - 0.5) < 0.001

    def test_temperature_effect(self):
        """Test that lower temperature sharpens weights."""
        scores = np.array([0.2, 0.5, 0.8])

        high_temp = SoftmaxStrategy(temperature=10.0)
        low_temp = SoftmaxStrategy(temperature=0.1)

        high_result = high_temp.aggregate(scores)
        low_result = low_temp.aggregate(scores)

        assert low_result > high_result

    def test_empty_returns_zero(self):
        """Test empty input returns zero."""
        strategy = SoftmaxStrategy()
        result = strategy.aggregate(np.array([]))
        assert result == 0.0


class TestGeometricMeanStrategy:
    """Tests for Geometric Mean strategy."""

    def test_geometric_mean_calculation(self):
        """Test correct geometric mean."""
        strategy = GeometricMeanStrategy()
        result = strategy.aggregate(np.array([0.5, 0.5]))
        assert abs(result - 0.5) < 0.001

    def test_penalizes_low_scores(self):
        """Test that low scores pull down the result."""
        strategy = GeometricMeanStrategy()
        geo = strategy.aggregate(np.array([0.9, 0.1]))
        arith = np.mean([0.9, 0.1])
        assert geo < arith

    def test_single_score_passthrough(self):
        """Test single score is returned."""
        strategy = GeometricMeanStrategy()
        result = strategy.aggregate(np.array([0.7]))
        assert abs(result - 0.7) < 0.001


class TestHarmonicMeanStrategy:
    """Tests for Harmonic Mean strategy."""

    def test_harmonic_mean_calculation(self):
        """Test correct harmonic mean."""
        strategy = HarmonicMeanStrategy()
        result = strategy.aggregate(np.array([0.5, 0.5]))
        assert abs(result - 0.5) < 0.001

    def test_penalizes_low_scores_severely(self):
        """Test that low scores severely impact result."""
        strategy = HarmonicMeanStrategy()
        harm = strategy.aggregate(np.array([0.9, 0.1]))
        geo = GeometricMeanStrategy().aggregate(np.array([0.9, 0.1]))
        assert harm < geo


class TestNeuralAggregatorFactory:
    """Tests for the aggregator factory."""

    def test_create_weighted_average(self):
        """Test factory creates weighted average."""
        strategy = NeuralAggregatorFactory.create("weighted_average")
        assert strategy.name == "weighted_average"
        assert isinstance(strategy, WeightedAverageStrategy)

    def test_create_with_enum(self):
        """Test factory works with enum."""
        strategy = NeuralAggregatorFactory.create(NeuralAggregationStrategy.MAX_SCORE)
        assert strategy.name == "max_score"

    def test_create_with_params(self):
        """Test factory passes params."""
        strategy = NeuralAggregatorFactory.create(
            "softmax", params={"temperature": 0.5}
        )
        assert strategy.temperature == 0.5

    def test_unknown_strategy_raises(self):
        """Test unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown neural aggregation strategy"):
            NeuralAggregatorFactory.create("unknown_strategy")

    def test_available_strategies(self):
        """Test listing available strategies."""
        available = NeuralAggregatorFactory.available_strategies()
        assert "weighted_average" in available
        assert "max_score" in available
        assert "softmax" in available
        assert "geometric_mean" in available
        assert "harmonic_mean" in available


class TestNeuralAggregatorParams:
    """Tests for NeuralAggregator parameter handling."""

    def test_unknown_params_are_filtered(self):
        """Unexpected params must be ignored instead of breaking strategy init."""
        aggregator = NeuralAggregator(
            strategy="weighted_average",
            params={"temperature": 2.0},
        )

        result = aggregator.aggregate_single([0.4, 0.6])

        assert aggregator.strategy.name == NeuralAggregationStrategy.WEIGHTED_AVERAGE.value
        assert abs(result.score - 0.5) < 1e-6


class TestNeuralAggregator:
    """Tests for the main NeuralAggregator class."""

    def test_default_strategy_is_weighted_average(self):
        """Test that default strategy is weighted average."""
        aggregator = NeuralAggregator()
        assert aggregator.strategy_name == "weighted_average"

    def test_aggregate_single_returns_result(self):
        """Test single aggregation returns NeuralAggregationResult."""
        aggregator = NeuralAggregator()
        result = aggregator.aggregate_single([0.5, 0.6, 0.7])
        assert isinstance(result, NeuralAggregationResult)
        assert result.score > 0
        assert result.num_models == 3
        assert result.strategy_used == "weighted_average"

    def test_aggregate_single_float(self):
        """Test aggregation of single float value."""
        aggregator = NeuralAggregator()
        result = aggregator.aggregate_single(0.75)
        assert abs(result.score - 0.75) < 0.001
        assert result.num_models == 1

    def test_min_score_filter(self):
        """Test that scores below min_score are filtered."""
        aggregator = NeuralAggregator(min_score=0.1)
        result = aggregator.aggregate_single([0.05, 0.5, 0.8])
        assert result.num_models == 2
        assert result.metadata["filtered_by_min_score"] == 1

    def test_aggregate_batch(self):
        """Test batch aggregation."""
        aggregator = NeuralAggregator()
        results = aggregator.aggregate_batch([
            [0.5, 0.6],
            [0.8],
            [0.3, 0.4, 0.5],
        ])
        assert len(results) == 3
        assert all(isinstance(r, NeuralAggregationResult) for r in results)

    def test_aggregate_array_1d(self):
        """Test 1D array aggregation (passthrough)."""
        aggregator = NeuralAggregator()
        scores = np.array([0.5, 0.6, 0.7])
        results = aggregator.aggregate_array(scores)
        np.testing.assert_array_equal(results, scores)

    def test_aggregate_array_2d(self):
        """Test 2D array aggregation (across models)."""
        aggregator = NeuralAggregator()
        scores = np.array([
            [0.5, 0.6],
            [0.8, 0.9],
            [0.3, 0.7],
        ])
        results = aggregator.aggregate_array(scores)
        assert len(results) == 3
        assert all(r >= 0 for r in results)


class TestEdgeCases:
    """Edge case tests for neural aggregation."""

    def test_empty_input(self):
        """Test empty input handling."""
        aggregator = NeuralAggregator()
        result = aggregator.aggregate_single([])
        assert result.score == 0.0
        assert result.num_models == 0

    def test_all_zeros(self):
        """Test all zero scores."""
        aggregator = NeuralAggregator(min_score=0.0)
        result = aggregator.aggregate_single([0.0, 0.0, 0.0])
        assert result.score == 0.0

    def test_very_small_scores(self):
        """Test handling of very small score values."""
        aggregator = NeuralAggregator()
        result = aggregator.aggregate_single([0.001, 0.002, 0.003])
        assert result.score > 0
        assert result.num_models == 3

    def test_perfect_scores(self):
        """Test perfect scores (1.0)."""
        aggregator = NeuralAggregator()
        result = aggregator.aggregate_single([1.0, 1.0])
        assert abs(result.score - 1.0) < 0.001


class TestSingleNeuralFeature:
    """Tests for common case: single hybrid_score from RotatE."""

    def test_single_feature_passthrough(self):
        """Test that single neural feature passes through unchanged."""
        aggregator = NeuralAggregator()
        result = aggregator.aggregate_single(0.72)
        assert abs(result.score - 0.72) < 0.001

    def test_single_feature_batch(self):
        """Test batch of single features."""
        aggregator = NeuralAggregator()
        single_scores = [0.5, 0.6, 0.7, 0.8]
        results = aggregator.aggregate_batch(single_scores)
        for score, result in zip(single_scores, results, strict=False):
            assert abs(result.score - score) < 0.001

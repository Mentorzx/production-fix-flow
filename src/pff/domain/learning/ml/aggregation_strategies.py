"""
Aggregation Strategies for Probabilistic Confidence Scores.

This module provides aggregation strategies for combining multiple
confidence scores into a single prediction. Used primarily by the
Probabilistic Circuits (PC) component.

Design Patterns Applied:
    - **Strategy Pattern:** Interchangeable aggregation algorithms.
    - **Factory Pattern:** Create aggregators from config strings.

Default Strategy: Noisy-OR
    The Noisy-OR aggregation correctly accumulates evidence from
    multiple sources:

    P(triple) = 1 - ∏(1 - confidence_i)

    Example: 3 sources each with 0.5 confidence
    - Max: 0.5 (ignores evidence accumulation)
    - Mean: 0.5 (dilutes strong evidence)
    - Noisy-OR: 1 - (0.5 * 0.5 * 0.5) = 0.875 (correct accumulation)

Reference:
    - SAFRAN: An Interpretable, Rule-Based Link Prediction Method (2021)
"""

from __future__ import annotations
from typing import Any, cast

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class AggregationStrategy(str, Enum):
    """Available aggregation strategies."""

    NOISY_OR = "noisy_or"
    MAX_CONFIDENCE = "max_confidence"
    WEIGHTED_SUM = "weighted_sum"
    MEAN = "mean"


class BaseAggregatorStrategy(ABC):
    """Abstract base class for aggregation strategies.

    All strategies must implement the aggregate method that combines
    multiple confidence scores into a single prediction.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        ...

    @abstractmethod
    def aggregate(
        self,
        confidences: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Aggregate confidences into a single score.

        Args:
            confidences: Array of confidence scores [0, 1].
            weights: Optional weights for each source (normalized).

        Returns:
            float: Aggregated confidence score [0, 1].
        """
        ...


class NoisyOrStrategy(BaseAggregatorStrategy):
    """Noisy-OR aggregation (SAFRAN default).

    Treats each source as an independent probabilistic cause.
    Correctly accumulates evidence from multiple sources.

    Formula: P = 1 - ∏(1 - c_i)

    Where c_i is the confidence of source i.

    Properties:
        - More sources with positive confidence → higher final score
        - Single source with 1.0 confidence → final score 1.0
        - Handles the "multiple weak sources" case correctly
    """

    def __init__(self, base_confidence: float = 0.01):
        """Initialize Noisy-OR strategy.

        Args:
            base_confidence: Minimum confidence floor for numerical stability.
        """
        self.base_confidence = base_confidence

    @property
    def name(self) -> str:
        return AggregationStrategy.NOISY_OR.value

    def aggregate(
        self,
        confidences: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Apply Noisy-OR aggregation.

        Args:
            confidences: Confidence scores.
            weights: Ignored for Noisy-OR (all sources treated equally).

        Returns:
            float: Aggregated confidence [0, 1].
        """
        if len(confidences) == 0:
            return self.base_confidence

        clipped = np.clip(confidences, self.base_confidence, 1.0 - 1e-9)
        complement_product = np.prod(1.0 - clipped)

        return float(1.0 - complement_product)


class MaxConfidenceStrategy(BaseAggregatorStrategy):
    """Max confidence aggregation.

    Returns the highest confidence among all sources.
    Simple but ignores evidence accumulation.
    """

    def __init__(self, default: float = 0.0):
        """Initialize Max strategy.

        Args:
            default: Default value when no confidences provided.
        """
        self.default = default

    @property
    def name(self) -> str:
        return AggregationStrategy.MAX_CONFIDENCE.value

    def aggregate(
        self,
        confidences: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Return maximum confidence.

        Args:
            confidences: Confidence scores.
            weights: Ignored for max aggregation.

        Returns:
            float: Maximum confidence or default if empty.
        """
        if len(confidences) == 0:
            return self.default
        return float(np.max(confidences))


class MeanStrategy(BaseAggregatorStrategy):
    """Mean aggregation.

    Returns the arithmetic mean of all confidences.
    """

    def __init__(self, default: float = 0.0):
        """Initialize Mean strategy.

        Args:
            default: Default value when no confidences provided.
        """
        self.default = default

    @property
    def name(self) -> str:
        return AggregationStrategy.MEAN.value

    def aggregate(
        self,
        confidences: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Return mean confidence.

        Args:
            confidences: Confidence scores.
            weights: If provided, computes weighted mean.

        Returns:
            float: Mean confidence or default if empty.
        """
        if len(confidences) == 0:
            return self.default
        if weights is not None and len(weights) == len(confidences):
            return float(np.average(confidences, weights=weights))
        return float(np.mean(confidences))


class WeightedSumStrategy(BaseAggregatorStrategy):
    """Weighted sum aggregation.

    Returns the weighted sum of confidences, clipped to [0, 1].
    """

    def __init__(self, default: float = 0.0):
        """Initialize Weighted Sum strategy.

        Args:
            default: Default value when no confidences provided.
        """
        self.default = default

    @property
    def name(self) -> str:
        return AggregationStrategy.WEIGHTED_SUM.value

    def aggregate(
        self,
        confidences: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Return weighted sum of confidences.

        Args:
            confidences: Confidence scores.
            weights: Weights for each confidence.

        Returns:
            float: Weighted sum clipped to [0, 1].
        """
        if len(confidences) == 0:
            return self.default
        if weights is None:
            weights = np.ones_like(confidences) / len(confidences)
        result = np.sum(confidences * weights)
        return float(np.clip(result, 0.0, 1.0))


def get_aggregation_strategy(
    strategy_name: str,
    **kwargs,
) -> BaseAggregatorStrategy:
    """Factory function to create aggregation strategies.

    Args:
        strategy_name: Name of the strategy (noisy_or, max_confidence, mean, weighted_sum).
        **kwargs: Additional arguments passed to strategy constructor.

    Returns:
        BaseAggregatorStrategy: Configured strategy instance.

    Raises:
        ValueError: If strategy_name is not recognized.
    """
    strategies = {
        "noisy_or": NoisyOrStrategy,
        "max_confidence": MaxConfidenceStrategy,
        "max": MaxConfidenceStrategy,
        "mean": MeanStrategy,
        "average": MeanStrategy,
        "weighted_sum": WeightedSumStrategy,
        "weighted": WeightedSumStrategy,
    }

    strategy_class = strategies.get(strategy_name.lower())
    if strategy_class is None:
        raise ValueError(
            f"Unknown aggregation strategy: {strategy_name}. Available: {list(strategies.keys())}"
        )

    return cast(Any, strategy_class)(**kwargs)

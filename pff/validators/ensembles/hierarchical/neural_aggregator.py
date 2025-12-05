"""
Neural Aggregator for Hierarchical Ensemble.

This module implements aggregation strategies for combining neural
embedding scores from RotatE or other KGE models.

Design Patterns Applied:
    - **Strategy Pattern:** Interchangeable aggregation algorithms.
    - **Factory Pattern:** Create aggregators from config strings.
    - **Template Method:** Common pre/post processing in base class.

Default Strategy: Weighted Average
    The weighted average is a stable default that balances multiple
    neural predictions (e.g., from ensemble of RotatE models or
    different relation-specific embeddings).

Neural Signal Properties:
    - Scores are typically in [0, 1] after sigmoid normalization
    - Single feature (hybrid_score) from RotatE in current pipeline
    - Future: multiple neural features from different KGE models
    
Confidence Calculation:
    - Entropy-based confidence: 1 - normalized_entropy of prediction
    - High confidence = low entropy (certain predictions)
    - Low confidence = high entropy (uncertain predictions)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import inspect
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pff.utils.logger import logger


def compute_entropy_confidence(
    score: float,
    epsilon: float = 1e-10,
) -> float:
    """Compute confidence from prediction probability using entropy.
    
    Confidence is defined as 1 - normalized_entropy.
    For binary predictions, entropy is maximal at p=0.5 (confidence=0)
    and minimal at p=0 or p=1 (confidence=1).
    
    Args:
        score: Prediction probability [0, 1].
        epsilon: Small value to avoid log(0).
        
    Returns:
        float: Confidence value [0, 1] where 1 is most confident.
    """
    p = np.clip(score, epsilon, 1.0 - epsilon)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    # Normalize: max binary entropy is 1.0 (at p=0.5)
    normalized_entropy = entropy  # Already normalized for binary
    confidence = 1.0 - normalized_entropy
    return float(confidence)


def compute_entropy_confidence_batch(
    scores: NDArray[np.float64],
    epsilon: float = 1e-10,
) -> NDArray[np.float64]:
    """Compute entropy-based confidence for a batch of scores.
    
    Args:
        scores: Array of prediction probabilities [0, 1].
        epsilon: Small value to avoid log(0).
        
    Returns:
        NDArray[np.float64]: Confidence values [0, 1].
    """
    p = np.clip(scores, epsilon, 1.0 - epsilon)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    confidence = 1.0 - entropy
    return confidence


class NeuralAggregationStrategy(str, Enum):
    """Available neural aggregation strategies."""

    WEIGHTED_AVERAGE = "weighted_average"
    MAX_SCORE = "max_score"
    SOFTMAX = "softmax"
    GEOMETRIC_MEAN = "geometric_mean"
    HARMONIC_MEAN = "harmonic_mean"


@dataclass
class NeuralAggregationResult:
    """Result of neural aggregation for a single triple.

    Attributes:
        score: Final aggregated neural score [0, 1].
        confidence: Confidence in the prediction [0, 1] (entropy-based).
        num_models: Number of neural models/features that contributed.
        strategy_used: Name of the aggregation strategy.
        individual_scores: Scores from each neural source (for debugging).
        metadata: Additional strategy-specific metadata.
    """

    score: float
    confidence: float  # Distinct from score - based on entropy
    num_models: int
    strategy_used: str
    individual_scores: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class NeuralAggregatorStrategy(ABC):
    """Abstract base class for neural aggregation strategies.

    All strategies must implement the aggregate method that combines
    multiple neural scores into a single prediction.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        ...

    @abstractmethod
    def aggregate(
        self,
        scores: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Aggregate neural scores into a single score.

        Args:
            scores: Array of neural model scores [0, 1].
            weights: Optional weights for each model (normalized).

        Returns:
            float: Aggregated score [0, 1].
        """
        ...


class WeightedAverageStrategy(NeuralAggregatorStrategy):
    """Weighted average aggregation (default).

    Simple and stable aggregation that combines multiple neural
    predictions with optional weighting.

    Properties:
        - Preserves calibration of individual models
        - Smooth blending of predictions
        - Works well with single or multiple neural sources
    """

    def __init__(self, normalize_weights: bool = True):
        """Initialize weighted average strategy.

        Args:
            normalize_weights: If True, normalize weights to sum to 1.
        """
        self.normalize_weights = normalize_weights

    @property
    def name(self) -> str:
        return NeuralAggregationStrategy.WEIGHTED_AVERAGE.value

    def aggregate(
        self,
        scores: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Compute weighted average of scores.

        Args:
            scores: Neural model scores.
            weights: Optional weights. Defaults to uniform.

        Returns:
            float: Weighted average score.
        """
        if len(scores) == 0:
            return 0.0

        if weights is None:
            weights = np.ones_like(scores)

        if self.normalize_weights and np.sum(weights) > 0:
            weights = weights / np.sum(weights)

        return float(np.dot(scores, weights))


class MaxScoreStrategy(NeuralAggregatorStrategy):
    """Maximum score aggregation.

    Returns the highest score among all neural models.
    Useful when any single high-confidence prediction is sufficient.
    """

    @property
    def name(self) -> str:
        return NeuralAggregationStrategy.MAX_SCORE.value

    def aggregate(
        self,
        scores: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Return maximum score.

        Args:
            scores: Neural model scores.
            weights: Ignored for max strategy.

        Returns:
            float: Maximum score.
        """
        if len(scores) == 0:
            return 0.0
        return float(np.max(scores))


class SoftmaxStrategy(NeuralAggregatorStrategy):
    """Softmax-weighted aggregation.

    Applies softmax to scores to create dynamic weights,
    then computes weighted sum. Temperature controls sharpness.

    Properties:
        - High-confidence predictions get more weight
        - Temperature=1.0: standard softmax
        - Temperature→0: approaches max
        - Temperature→∞: approaches uniform
    """

    def __init__(self, temperature: float = 1.0):
        """Initialize softmax strategy.

        Args:
            temperature: Softmax temperature. Lower = sharper.
        """
        self.temperature = max(0.01, temperature)

    @property
    def name(self) -> str:
        return NeuralAggregationStrategy.SOFTMAX.value

    def aggregate(
        self,
        scores: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Compute softmax-weighted aggregation.

        Args:
            scores: Neural model scores.
            weights: Optional base weights (combined with softmax).

        Returns:
            float: Softmax-weighted score.
        """
        if len(scores) == 0:
            return 0.0

        scaled = scores / self.temperature
        scaled = scaled - np.max(scaled)
        exp_scores = np.exp(scaled)
        softmax_weights = exp_scores / np.sum(exp_scores)

        if weights is not None:
            softmax_weights = softmax_weights * weights
            softmax_weights = softmax_weights / np.sum(softmax_weights)

        return float(np.dot(scores, softmax_weights))


class GeometricMeanStrategy(NeuralAggregatorStrategy):
    """Geometric mean aggregation.

    Computes nth root of product of scores. Penalizes low scores
    more strongly than arithmetic mean.

    Properties:
        - Requires all models to agree
        - Single low score heavily impacts result
        - Good for consensus-based aggregation
    """

    def __init__(self, epsilon: float = 1e-10):
        """Initialize geometric mean strategy.

        Args:
            epsilon: Small value to avoid log(0).
        """
        self.epsilon = epsilon

    @property
    def name(self) -> str:
        return NeuralAggregationStrategy.GEOMETRIC_MEAN.value

    def aggregate(
        self,
        scores: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Compute geometric mean.

        Args:
            scores: Neural model scores.
            weights: Optional weights for weighted geometric mean.

        Returns:
            float: Geometric mean score.
        """
        if len(scores) == 0:
            return 0.0

        clipped = np.clip(scores, self.epsilon, 1.0)

        if weights is not None:
            weights = weights / np.sum(weights)
            log_mean = np.sum(weights * np.log(clipped))
        else:
            log_mean = np.mean(np.log(clipped))

        return float(np.exp(log_mean))


class HarmonicMeanStrategy(NeuralAggregatorStrategy):
    """Harmonic mean aggregation.

    Penalizes low scores even more strongly than geometric mean.
    Useful when all neural sources must contribute.

    Properties:
        - Very sensitive to low scores
        - Good for "weakest link" scenarios
        - Bounded by minimum score
    """

    def __init__(self, epsilon: float = 1e-10):
        """Initialize harmonic mean strategy.

        Args:
            epsilon: Small value to avoid division by zero.
        """
        self.epsilon = epsilon

    @property
    def name(self) -> str:
        return NeuralAggregationStrategy.HARMONIC_MEAN.value

    def aggregate(
        self,
        scores: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Compute harmonic mean.

        Args:
            scores: Neural model scores.
            weights: Ignored for harmonic mean.

        Returns:
            float: Harmonic mean score.
        """
        if len(scores) == 0:
            return 0.0

        clipped = np.clip(scores, self.epsilon, 1.0)
        return float(len(clipped) / np.sum(1.0 / clipped))


class NeuralAggregatorFactory:
    """Factory for creating neural aggregator strategies.

    Centralizes strategy instantiation and parameter validation.

    Usage:
        strategy = NeuralAggregatorFactory.create("weighted_average")
        result = strategy.aggregate(scores)
    """

    _REGISTRY: dict[str, type[NeuralAggregatorStrategy]] = {
        NeuralAggregationStrategy.WEIGHTED_AVERAGE.value: WeightedAverageStrategy,
        NeuralAggregationStrategy.MAX_SCORE.value: MaxScoreStrategy,
        NeuralAggregationStrategy.SOFTMAX.value: SoftmaxStrategy,
        NeuralAggregationStrategy.GEOMETRIC_MEAN.value: GeometricMeanStrategy,
        NeuralAggregationStrategy.HARMONIC_MEAN.value: HarmonicMeanStrategy,
    }

    @classmethod
    def create(
        cls,
        strategy: str | NeuralAggregationStrategy = NeuralAggregationStrategy.WEIGHTED_AVERAGE,
        params: dict[str, Any] | None = None,
    ) -> NeuralAggregatorStrategy:
        """Create a neural aggregation strategy instance.

        Args:
            strategy: Strategy name or enum value.
            params: Strategy-specific parameters.

        Returns:
            NeuralAggregatorStrategy: Configured strategy instance.

        Raises:
            ValueError: If strategy is not recognized.
        """
        if isinstance(strategy, NeuralAggregationStrategy):
            strategy_name = strategy.value
        else:
            strategy_name = strategy.lower()

        if strategy_name not in cls._REGISTRY:
            available = list(cls._REGISTRY.keys())
            raise ValueError(
                f"Unknown neural aggregation strategy: {strategy_name}. "
                f"Available: {available}"
            )

        strategy_class = cls._REGISTRY[strategy_name]
        params = params or {}

        return strategy_class(**params)

    @classmethod
    def available_strategies(cls) -> list[str]:
        """Return list of available strategy names."""
        return list(cls._REGISTRY.keys())


class NeuralAggregator:
    """Main interface for neural score aggregation.

    Aggregates scores from multiple neural models for a set of triples.
    Supports batch processing for efficiency.
    
    Key Feature: Entropy-Based Confidence
        When entropy_based_confidence=True (default), the aggregator
        computes a separate confidence value based on prediction entropy.
        This is used by the DecisionRouter to make routing decisions.
        - score: The actual prediction (probability)
        - confidence: How certain the model is (1 - entropy)

    Usage:
        aggregator = NeuralAggregator(strategy="weighted_average")
        results = aggregator.aggregate_batch(neural_scores_matrix)

    Attributes:
        strategy: The aggregation strategy being used.
        min_score: Minimum score threshold to include a model.
        entropy_based_confidence: Use entropy for confidence calculation.
    """

    def __init__(
        self,
        strategy: str | NeuralAggregationStrategy = NeuralAggregationStrategy.WEIGHTED_AVERAGE,
        params: dict[str, Any] | None = None,
        min_score: float = 0.0,
        entropy_based_confidence: bool = True,
    ):
        """Initialize the neural aggregator.

        Args:
            strategy: Aggregation strategy name.
            params: Strategy-specific parameters.
            min_score: Minimum score to include a model.
            entropy_based_confidence: If True, compute confidence via entropy.
        """
        strategy_name = (
            strategy.value if isinstance(strategy, NeuralAggregationStrategy) else str(strategy)
        )
        allowed_keys: set[str] = set()
        if strategy_name == NeuralAggregationStrategy.SOFTMAX.value:
            allowed_keys.update({"temperature"})
        elif strategy_name == NeuralAggregationStrategy.WEIGHTED_AVERAGE.value:
            allowed_keys.update({"normalize_weights"})
        elif strategy_name == NeuralAggregationStrategy.GEOMETRIC_MEAN.value:
            allowed_keys.update({"epsilon"})

        # Fallback: derive allowed keys from the strategy signature to drop stray params
        strategy_class = NeuralAggregatorFactory._REGISTRY.get(strategy_name)
        if strategy_class and not allowed_keys:
            signature_keys = {
                name
                for name, param in inspect.signature(strategy_class.__init__).parameters.items()
                if name not in {"self", "args", "kwargs"}
            }
            allowed_keys.update(signature_keys)

        safe_params = {k: v for k, v in (params or {}).items() if (not allowed_keys or k in allowed_keys)}
        if params and len(safe_params) != len(params):
            logger.debug(
                f"Neural strategy params filtered; strategy={strategy_name}, "
                f"allowed={allowed_keys or 'none'}, received={list(params.keys())}"
            )

        self.strategy = NeuralAggregatorFactory.create(strategy, safe_params)
        self.min_score = min_score
        self.entropy_based_confidence = entropy_based_confidence

        logger.debug(
            f"NeuralAggregator initialized with strategy={self.strategy.name}, "
            f"min_score={min_score}, entropy_based_confidence={entropy_based_confidence}"
        )

    def aggregate_single(
        self,
        neural_scores: NDArray[np.float64] | list[float] | float,
        model_weights: NDArray[np.float64] | list[float] | None = None,
    ) -> NeuralAggregationResult:
        """Aggregate neural scores for a single triple.

        Args:
            neural_scores: Scores from neural models. Can be single float.
            model_weights: Optional weights for each model.

        Returns:
            NeuralAggregationResult: Aggregated result with score and confidence.
        """
        if isinstance(neural_scores, (int, float)):
            neural_scores = [neural_scores]

        scores = np.asarray(neural_scores, dtype=np.float64)

        mask = scores >= self.min_score
        filtered = scores[mask]

        weights = None
        if model_weights is not None:
            weights = np.asarray(model_weights, dtype=np.float64)[mask]

        if len(filtered) == 0:
            aggregated = 0.0
        else:
            aggregated = self.strategy.aggregate(filtered, weights)

        # Compute confidence based on entropy or use score directly
        if self.entropy_based_confidence:
            confidence = compute_entropy_confidence(aggregated)
        else:
            confidence = aggregated

        return NeuralAggregationResult(
            score=aggregated,
            confidence=confidence,
            num_models=len(filtered),
            strategy_used=self.strategy.name,
            individual_scores=filtered.tolist(),
            metadata={
                "original_num_models": len(scores),
                "filtered_by_min_score": int(np.sum(~mask)),
                "entropy_based": self.entropy_based_confidence,
            },
        )

    def aggregate_batch(
        self,
        scores_per_triple: list[NDArray[np.float64] | list[float] | float],
        weights_per_triple: list[NDArray[np.float64] | list[float] | None] | None = None,
    ) -> list[NeuralAggregationResult]:
        """Aggregate neural scores for multiple triples.

        Args:
            scores_per_triple: List of score arrays, one per triple.
            weights_per_triple: Optional list of weight arrays.

        Returns:
            list[NeuralAggregationResult]: Aggregation results for each triple.
        """
        if weights_per_triple is None:
            weights_per_triple = [None] * len(scores_per_triple)

        results = []
        for scores, weights in zip(
            scores_per_triple, weights_per_triple, strict=False
        ):
            results.append(self.aggregate_single(scores, weights))

        return results

    def aggregate_array(
        self,
        scores: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Aggregate a 1D array of scores (one per triple).

        For the common case where there's a single neural feature
        (hybrid_score from RotatE), this is a passthrough.

        For 2D arrays (triples x models), aggregates across models.

        Args:
            scores: 1D or 2D array of neural scores.

        Returns:
            NDArray[np.float64]: Aggregated score per triple.
        """
        if scores.ndim == 1:
            return scores

        n_triples = scores.shape[0]
        results = np.zeros(n_triples, dtype=np.float64)

        for i in range(n_triples):
            results[i] = self.strategy.aggregate(scores[i])

        return results

    @property
    def strategy_name(self) -> str:
        """Return the name of the current strategy."""
        return self.strategy.name

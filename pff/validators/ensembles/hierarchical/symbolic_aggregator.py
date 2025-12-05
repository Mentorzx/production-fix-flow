"""
Symbolic Aggregator for Hierarchical Ensemble.

This module implements aggregation strategies for combining rule-based
confidence scores from symbolic validators (AnyBURL, PyClause).

Design Patterns Applied:
    - **Strategy Pattern:** Interchangeable aggregation algorithms.
    - **Factory Pattern:** Create aggregators from config strings.
    - **Template Method:** Common pre/post processing in base class.

Default Strategy: Noisy-OR
    The Noisy-OR aggregation (SAFRAN/AnyBURL literature) correctly
    accumulates evidence from multiple rules:
    
    P(triple) = 1 - ∏(1 - confidence_i)
    
    Example: 3 rules each with 0.5 confidence
    - Max: 0.5 (ignores evidence accumulation)
    - Mean: 0.5 (dilutes strong evidence)
    - Noisy-OR: 1 - (0.5 * 0.5 * 0.5) = 0.875 (correct accumulation)

Reference:
    - SAFRAN: An Interpretable, Rule-Based Link Prediction Method (2021)
    - AnyBURL: Anytime Bottom-Up Rule Learning (2019)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import inspect

import numpy as np
from numpy.typing import NDArray

from pff.config import PC_CONFIG_PATH
from pff.utils.file_manager import FileManager
from pff.utils.logger import logger


class AggregationStrategy(str, Enum):
    """Available symbolic aggregation strategies."""

    NOISY_OR = "noisy_or"
    MAX_CONFIDENCE = "max_confidence"
    WEIGHTED_SUM = "weighted_sum"
    VOTING = "voting"
    MEAN = "mean"
    PC = "pc"


@dataclass
class AggregationResult:
    """Result of symbolic aggregation for a single triple.

    Attributes:
        confidence: Final aggregated confidence score [0, 1].
        num_rules_fired: Number of rules that contributed.
        strategy_used: Name of the aggregation strategy.
        rule_confidences: Individual rule confidence scores (for debugging).
        metadata: Additional strategy-specific metadata.
    """

    confidence: float
    num_rules_fired: int
    strategy_used: str
    rule_confidences: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SymbolicAggregatorStrategy(ABC):
    """Abstract base class for symbolic aggregation strategies.

    All strategies must implement the aggregate method that combines
    multiple rule confidence scores into a single prediction.
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
        """Aggregate rule confidences into a single score.

        Args:
            confidences: Array of rule confidence scores [0, 1].
            weights: Optional weights for each rule (normalized).

        Returns:
            float: Aggregated confidence score [0, 1].
        """
        ...


class NoisyOrStrategy(SymbolicAggregatorStrategy):
    """Noisy-OR aggregation (SAFRAN/AnyBURL default).

    Treats each rule as an independent probabilistic cause.
    Correctly accumulates evidence from multiple rules.

    Formula: P = 1 - ∏(1 - c_i)

    Where c_i is the confidence of rule i.

    Properties:
        - More rules with positive confidence → higher final score
        - Single rule with 1.0 confidence → final score 1.0
        - Handles the "multiple weak rules" case correctly
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
            confidences: Rule confidence scores.
            weights: Ignored for Noisy-OR (all rules treated equally).

        Returns:
            float: Aggregated confidence [0, 1].
        """
        if len(confidences) == 0:
            return self.base_confidence

        clipped = np.clip(confidences, self.base_confidence, 1.0 - 1e-9)

        complement_product = np.prod(1.0 - clipped)

        return float(1.0 - complement_product)


class MaxConfidenceStrategy(SymbolicAggregatorStrategy):
    """Max confidence aggregation.

    Returns the highest confidence among all rules.
    Simple but ignores evidence accumulation.

    Use case: When only the strongest rule should decide.
    """

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
            confidences: Rule confidence scores.
            weights: Ignored for max strategy.

        Returns:
            float: Maximum confidence value.
        """
        if len(confidences) == 0:
            return 0.0
        return float(np.max(confidences))


class WeightedSumStrategy(SymbolicAggregatorStrategy):
    """Weighted sum aggregation with optional normalization.

    Combines rule confidences using weights, with optional
    capping to ensure output stays in [0, 1].
    """

    def __init__(self, normalize: bool = True, cap: float = 1.0):
        """Initialize weighted sum strategy.

        Args:
            normalize: If True, normalize weights to sum to 1.
            cap: Maximum value for aggregated score.
        """
        self.normalize = normalize
        self.cap = cap

    @property
    def name(self) -> str:
        return AggregationStrategy.WEIGHTED_SUM.value

    def aggregate(
        self,
        confidences: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Compute weighted sum of confidences.

        Args:
            confidences: Rule confidence scores.
            weights: Weights for each rule. Defaults to uniform.

        Returns:
            float: Weighted sum, capped at self.cap.
        """
        if len(confidences) == 0:
            return 0.0

        if weights is None:
            weights = np.ones_like(confidences)

        if self.normalize and np.sum(weights) > 0:
            weights = weights / np.sum(weights)

        result = float(np.dot(confidences, weights))
        return min(result, self.cap)


class VotingStrategy(SymbolicAggregatorStrategy):
    """Voting-based aggregation.

    Returns the fraction of rules that exceed a confidence threshold.
    Useful when you want a "democratic" decision among rules.
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize voting strategy.

        Args:
            threshold: Minimum confidence for a rule to "vote yes".
        """
        self.threshold = threshold

    @property
    def name(self) -> str:
        return AggregationStrategy.VOTING.value

    def aggregate(
        self,
        confidences: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Compute fraction of rules voting yes.

        Args:
            confidences: Rule confidence scores.
            weights: Ignored for voting strategy.

        Returns:
            float: Fraction of rules exceeding threshold [0, 1].
        """
        if len(confidences) == 0:
            return 0.0

        votes = np.sum(confidences >= self.threshold)
        return float(votes / len(confidences))


class MeanStrategy(SymbolicAggregatorStrategy):
    """Simple arithmetic mean of confidences.

    Use case: Baseline comparison or when rules are expected
    to be independent and equally informative.
    """

    @property
    def name(self) -> str:
        return AggregationStrategy.MEAN.value

    def aggregate(
        self,
        confidences: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Compute arithmetic mean.

        Args:
            confidences: Rule confidence scores.
            weights: Ignored for mean strategy.

        Returns:
            float: Mean confidence.
        """
        if len(confidences) == 0:
            return 0.0
        return float(np.mean(confidences))


# Late import to avoid circular dependency during module initialization
from pff.validators.pc.strategy import ProbabilisticCircuitStrategy, build_pc_params_from_config  # noqa: E402


class SymbolicAggregatorFactory:
    """Factory for creating symbolic aggregator strategies.

    Centralizes strategy instantiation and parameter validation.

    Usage:
        strategy = SymbolicAggregatorFactory.create("noisy_or")
        result = strategy.aggregate(confidences)
    """

    _REGISTRY: dict[str, type[SymbolicAggregatorStrategy]] = {
        AggregationStrategy.NOISY_OR.value: NoisyOrStrategy,
        AggregationStrategy.MAX_CONFIDENCE.value: MaxConfidenceStrategy,
        AggregationStrategy.WEIGHTED_SUM.value: WeightedSumStrategy,
        AggregationStrategy.VOTING.value: VotingStrategy,
        AggregationStrategy.MEAN.value: MeanStrategy,
        AggregationStrategy.PC.value: ProbabilisticCircuitStrategy,
    }

    @classmethod
    def create(
        cls,
        strategy: str | AggregationStrategy = AggregationStrategy.NOISY_OR,
        params: dict[str, Any] | None = None,
    ) -> SymbolicAggregatorStrategy:
        """Create an aggregation strategy instance.

        Args:
            strategy: Strategy name or enum value.
            params: Strategy-specific parameters.

        Returns:
            SymbolicAggregatorStrategy: Configured strategy instance.

        Raises:
            ValueError: If strategy is not recognized.
        """
        if isinstance(strategy, AggregationStrategy):
            strategy_name = strategy.value
        else:
            strategy_name = strategy.lower()

        if strategy_name not in cls._REGISTRY:
            available = list(cls._REGISTRY.keys())
            raise ValueError(
                f"Unknown aggregation strategy: {strategy_name}. "
                f"Available: {available}"
            )

        strategy_class = cls._REGISTRY[strategy_name]
        params = params or {}

        return strategy_class(**params)

    @classmethod
    def available_strategies(cls) -> list[str]:
        """Return list of available strategy names."""
        return list(cls._REGISTRY.keys())


class SymbolicAggregator:
    """Main interface for symbolic rule aggregation.

    Aggregates confidence scores from multiple rules for a set of triples.
    Supports batch processing for efficiency.

    Usage:
        aggregator = SymbolicAggregator(strategy="noisy_or")
        results = aggregator.aggregate_batch(rule_scores_matrix)

    Attributes:
        strategy: The aggregation strategy being used.
        max_rules: Maximum number of rules to consider per triple.
        min_confidence: Minimum confidence threshold to include a rule.
    """

    def __init__(
        self,
        strategy: str | AggregationStrategy = AggregationStrategy.NOISY_OR,
        params: dict[str, Any] | None = None,
        max_rules: int = 50,
        min_confidence: float = 0.01,
    ):
        """Initialize the symbolic aggregator.

        Args:
            strategy: Aggregation strategy name.
            params: Strategy-specific parameters.
            max_rules: Maximum rules to consider per triple.
            min_confidence: Minimum confidence to include a rule.
        """
        effective_params = dict(params or {})

        # Allow max_rules/min_confidence in params for config convenience
        max_rules = int(effective_params.pop("max_rules", max_rules))
        min_confidence = float(effective_params.pop("min_confidence", min_confidence))

        strategy_name = strategy.value if isinstance(strategy, AggregationStrategy) else str(strategy)

        if strategy_name == AggregationStrategy.PC.value:
            try:
                pc_defaults = build_pc_params_from_config(FileManager.read(PC_CONFIG_PATH) or {})
                effective_params = {**pc_defaults, **effective_params}
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    f"Failed to load PC config from {PC_CONFIG_PATH}: {exc}; using inline parameters"
                )

        strategy_class = SymbolicAggregatorFactory._REGISTRY.get(strategy_name)
        if strategy_class is not None:
            allowed_keys = {
                name
                for name, param in inspect.signature(strategy_class.__init__).parameters.items()
                if name not in {"self", "args", "kwargs"}
            }
        else:
            allowed_keys = set()

        safe_params = {k: v for k, v in effective_params.items() if k in allowed_keys}
        if params and len(safe_params) != len(effective_params):
            logger.debug(
                f"Strategy params filtered; allowed={allowed_keys}, received={list(params.keys())}"
            )

        self.strategy = SymbolicAggregatorFactory.create(strategy, safe_params)
        self.max_rules = max_rules
        self.min_confidence = min_confidence

        logger.debug(
            f"SymbolicAggregator initialized with strategy={self.strategy.name}, "
            f"max_rules={max_rules}, min_confidence={min_confidence}"
        )

    def aggregate_single(
        self,
        rule_confidences: NDArray[np.float64] | list[float],
        rule_weights: NDArray[np.float64] | list[float] | None = None,
    ) -> AggregationResult:
        """Aggregate rule confidences for a single triple.

        Args:
            rule_confidences: Confidence scores from matching rules.
            rule_weights: Optional weights for each rule.

        Returns:
            AggregationResult: Aggregated result with metadata.
        """
        confidences = np.asarray(rule_confidences, dtype=np.float64)

        mask = confidences >= self.min_confidence
        filtered = confidences[mask]

        if len(filtered) > self.max_rules:
            # SAFRAN-style top-h: ordena por confiança antes de limitar
            top_indices = np.argsort(-filtered)[: self.max_rules]
            filtered = filtered[top_indices]

        weights = None
        if rule_weights is not None:
            weights = np.asarray(rule_weights, dtype=np.float64)[mask]
            if len(filtered) < len(weights):
                weights = weights[np.argsort(confidences[mask])[-self.max_rules:]]

        aggregated = self.strategy.aggregate(filtered, weights)

        return AggregationResult(
            confidence=aggregated,
            num_rules_fired=len(filtered),
            strategy_used=self.strategy.name,
            rule_confidences=filtered.tolist(),
            metadata={
                "original_num_rules": len(confidences),
                "filtered_by_min_confidence": int(np.sum(~mask)),
                "capped_by_max_rules": max(0, len(confidences[mask]) - self.max_rules),
            },
        )

    def aggregate_batch(
        self,
        rule_scores_per_triple: list[NDArray[np.float64] | list[float]],
        rule_weights_per_triple: list[NDArray[np.float64] | list[float] | None] | None = None,
    ) -> list[AggregationResult]:
        """Aggregate rule confidences for multiple triples.

        Args:
            rule_scores_per_triple: List of confidence arrays, one per triple.
            rule_weights_per_triple: Optional list of weight arrays.

        Returns:
            list[AggregationResult]: Aggregation results for each triple.
        """
        if rule_weights_per_triple is None:
            rule_weights_per_triple = [None] * len(rule_scores_per_triple)

        results = []
        for confidences, weights in zip(
            rule_scores_per_triple, rule_weights_per_triple, strict=False
        ):
            results.append(self.aggregate_single(confidences, weights))

        return results

    def aggregate_matrix(
        self,
        rule_matrix: NDArray[np.float64],
        fill_value: float = 0.0,
    ) -> NDArray[np.float64]:
        """Aggregate a matrix of rule scores (triples x rules).

        Optimized for dense matrices where each row is a triple
        and each column is a rule's confidence for that triple.

        Args:
            rule_matrix: 2D array of shape (n_triples, n_rules).
            fill_value: Value indicating no rule fired (usually 0).

        Returns:
            NDArray[np.float64]: Aggregated confidence per triple.
        """
        n_triples = rule_matrix.shape[0]
        results = np.zeros(n_triples, dtype=np.float64)

        for i in range(n_triples):
            row = rule_matrix[i]
            valid_mask = row > fill_value
            confidences = row[valid_mask]

            if len(confidences) == 0:
                results[i] = 0.0
            else:
                results[i] = self.strategy.aggregate(
                    np.clip(confidences, self.min_confidence, 1.0)[:self.max_rules]
                )

        return results

    @property
    def strategy_name(self) -> str:
        """Return the name of the current strategy."""
        return self.strategy.name

"""
Hierarchical Ensemble Pipeline.

This module provides a hierarchical alternative to the flat stacking ensemble.
It separates symbolic and neural aggregation, then routes decisions based
on confidence thresholds.

Design Patterns Applied:
    - **Strategy Pattern:** Flat vs Hierarchical pathway selection.
    - **Facade Pattern:** Simple interface hiding complex aggregation logic.
    - **Template Method:** Common prediction flow with customizable routing.

Usage:
    from pff.validators.ensembles.hierarchical import HierarchicalPipeline

    pipeline = HierarchicalPipeline.from_config()
    if pipeline.is_enabled:
        predictions = pipeline.predict(X, symbolic_features, neural_features)
    else:
        # Fall back to flat ensemble
        ...

Integration with AdvancedEnsembleTrainer:
    The trainer checks `load_hierarchical_config().is_hierarchical` and
    uses this pipeline instead of the standard stacking when enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pff.utils.logger import logger
from pff.validators.ensembles.hierarchical.config_loader import (
    HierarchicalConfig,
    load_hierarchical_config,
)
from pff.validators.ensembles.hierarchical.decision_router import (
    DecisionRouter,
    RoutingStatistics,
)
from pff.validators.ensembles.hierarchical.neural_aggregator import (
    NeuralAggregationStrategy,
    NeuralAggregator,
)
from pff.validators.ensembles.hierarchical.symbolic_aggregator import (
    SymbolicAggregator,
)


@dataclass
class HierarchicalPredictionResult:
    """Complete result from hierarchical prediction.

    Attributes:
        final_scores: Array of final prediction scores [0, 1].
        routing_stats: Aggregate routing statistics.
        symbolic_aggregated: Symbolic scores after aggregation.
        neural_aggregated: Neural scores after aggregation.
        metadata: Additional metadata for debugging/logging.
    """

    final_scores: NDArray[np.float64]
    routing_stats: RoutingStatistics
    symbolic_aggregated: NDArray[np.float64]
    neural_aggregated: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)


class HierarchicalPipeline:
    """Hierarchical neuro-symbolic ensemble pipeline.

    Processes predictions through three stages:
    1. Symbolic Aggregation: Combine rule confidences (Noisy-OR default)
    2. Neural Aggregation: Combine embedding scores
    3. Decision Routing: Route to SYMBOLIC_DECIDES, NEURAL_FALLBACK, or BLEND

    Attributes:
        config: Full hierarchical configuration.
        symbolic_aggregator: Strategy for combining rule confidences.
        neural_aggregator: Strategy for combining neural scores.
        decision_router: Routing logic based on thresholds.
        is_enabled: Whether hierarchical mode is active.
    """

    def __init__(
        self,
        config: HierarchicalConfig,
        symbolic_aggregator: SymbolicAggregator | None = None,
        neural_aggregator: NeuralAggregator | None = None,
        decision_router: DecisionRouter | None = None,
    ):
        """Initialize the hierarchical pipeline.

        Args:
            config: Hierarchical ensemble configuration.
            symbolic_aggregator: Custom symbolic aggregator (uses config if None).
            neural_aggregator: Custom neural aggregator (uses config if None).
            decision_router: Custom decision router (uses config if None).
        """
        self.config = config

        # Separate strategy params from aggregator params
        symbolic_strategy_params = {
            k: v for k, v in config.symbolic_aggregator.params.items()
            if k in ("base_confidence", "threshold", "normalize", "cap")  # Strategy-level params
        }
        symbolic_aggregator_params = {
            k: v for k, v in config.symbolic_aggregator.params.items()
            if k in ("max_rules", "min_confidence")  # Aggregator-level params
        }

        self.symbolic_aggregator = symbolic_aggregator or SymbolicAggregator(
            strategy=config.symbolic_aggregator.strategy,
            params=symbolic_strategy_params,
            max_rules=symbolic_aggregator_params.get("max_rules", 50),
            min_confidence=symbolic_aggregator_params.get("min_confidence", 0.01),
        )

        # Neural aggregator params are simpler
        # Filter params by strategy to avoid unexpected kwargs
        allowed_neural_keys = set()
        if config.neural_aggregator.strategy == NeuralAggregationStrategy.SOFTMAX.value:
            allowed_neural_keys.update({"temperature"})
        elif config.neural_aggregator.strategy == NeuralAggregationStrategy.WEIGHTED_AVERAGE.value:
            allowed_neural_keys.update({"normalize_weights"})
        elif config.neural_aggregator.strategy == NeuralAggregationStrategy.GEOMETRIC_MEAN.value:
            allowed_neural_keys.update({"epsilon"})

        neural_strategy_params = {
            k: v for k, v in config.neural_aggregator.params.items() if k in allowed_neural_keys
        }

        self.neural_aggregator = neural_aggregator or NeuralAggregator(
            strategy=config.neural_aggregator.strategy,
            params=neural_strategy_params,
        )

        self.decision_router = decision_router or DecisionRouter.from_config(
            config.decision_router
        )

        self._last_stats: RoutingStatistics | None = None

        logger.debug(
            f"HierarchicalPipeline initialized: "
            f"symbolic={self.symbolic_aggregator.strategy_name}, "
            f"neural={self.neural_aggregator.strategy_name}, "
            f"thresholds=(sym={self.decision_router.symbolic_threshold}, "
            f"neu={self.decision_router.neural_threshold})"
        )

    @classmethod
    def from_config(cls, config: HierarchicalConfig | None = None) -> HierarchicalPipeline:
        """Create pipeline from configuration.

        Args:
            config: Configuration. Loads from file if None.

        Returns:
            HierarchicalPipeline: Configured pipeline instance.
        """
        if config is None:
            config = load_hierarchical_config()
        return cls(config)

    @property
    def is_enabled(self) -> bool:
        """Check if hierarchical mode is enabled."""
        return self.config.is_hierarchical

    @property
    def last_routing_stats(self) -> RoutingStatistics | None:
        """Return statistics from the last prediction."""
        return self._last_stats

    def predict(
        self,
        symbolic_features: NDArray[np.float64],
        neural_features: NDArray[np.float64] | float,
    ) -> HierarchicalPredictionResult:
        """Generate predictions using hierarchical routing.

        Args:
            symbolic_features: Matrix of shape (n_samples, n_rules) with rule
                confidences for each sample. Each row contains the confidence
                scores from rules that matched that triple.
            neural_features: Neural scores. Either:
                - 1D array of shape (n_samples,) with single neural score per sample
                - 2D array of shape (n_samples, n_models) for multi-model aggregation
                - Single float applied to all samples

        Returns:
            HierarchicalPredictionResult: Complete prediction results with
                final scores, routing statistics, and intermediate values.
        """
        n_samples = symbolic_features.shape[0]

        symbolic_aggregated = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            result = self.symbolic_aggregator.aggregate_single(symbolic_features[i])
            symbolic_aggregated[i] = result.confidence

        if isinstance(neural_features, (int, float)):
            neural_aggregated = np.full(n_samples, neural_features, dtype=np.float64)
        elif neural_features.ndim == 1:
            neural_aggregated = neural_features.astype(np.float64)
        else:
            neural_aggregated = self.neural_aggregator.aggregate_array(neural_features)

        final_scores, decision_codes = self.decision_router.route_vectorized(
            symbolic_aggregated, neural_aggregated
        )

        stats = self.decision_router.compute_statistics_from_codes(
            decision_codes, final_scores, symbolic_aggregated, neural_aggregated
        )
        self._last_stats = stats

        return HierarchicalPredictionResult(
            final_scores=final_scores,
            routing_stats=stats,
            symbolic_aggregated=symbolic_aggregated,
            neural_aggregated=neural_aggregated,
            metadata={
                "symbolic_strategy": self.symbolic_aggregator.strategy_name,
                "neural_strategy": self.neural_aggregator.strategy_name,
                "config_type": self.config.architecture_type,
            },
        )

    def predict_from_flat_features(
        self,
        flat_features: NDArray[np.float64],
        neural_feature_index: int = 0,
    ) -> HierarchicalPredictionResult:
        """Predict from flat feature matrix (compatibility with existing pipeline).

        When the existing ensemble produces a flat feature matrix where:
        - Column 0 (or neural_feature_index) is the neural/hybrid score
        - Remaining columns are symbolic features (binary activations or confidences)

        This method extracts and routes them through the hierarchical pipeline.

        Args:
            flat_features: Feature matrix of shape (n_samples, n_features).
            neural_feature_index: Column index containing neural score.

        Returns:
            HierarchicalPredictionResult: Hierarchical prediction results.
        """
        if flat_features.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {flat_features.shape}")

        neural_features = flat_features[:, neural_feature_index]

        symbolic_mask = np.ones(flat_features.shape[1], dtype=bool)
        symbolic_mask[neural_feature_index] = False
        symbolic_features = flat_features[:, symbolic_mask]

        return self.predict(symbolic_features, neural_features)

    def should_apply_symbolic_dominance_penalty(self) -> bool:
        """Check if symbolic dominance penalty should be applied.

        This is a key integration point: in hierarchical mode, the
        penalty should typically be disabled since the architecture
        already handles modality separation.

        Returns:
            bool: True if penalty should be applied.
        """
        return self.config.should_apply_symbolic_dominance_penalty

    def log_routing_summary(self) -> None:
        """Log a summary of the last routing decision distribution."""
        if self._last_stats is None:
            logger.debug("No routing statistics available yet")
            return

        stats = self._last_stats
        logger.info(
            f"Roteamento hierarquico: "
            f"SYMBOLIC_DECIDES={stats.symbolic_decides_rate:.1%}, "
            f"NEURAL_FALLBACK={stats.neural_fallback_rate:.1%}, "
            f"BLEND={stats.blend_rate:.1%}"
        )
        logger.info(
            f"Scores medios: "
            f"symbolic={stats.avg_symbolic_score:.3f}, "
            f"neural={stats.avg_neural_score:.3f}, "
            f"final={stats.avg_final_score:.3f}"
        )


def get_hierarchical_pipeline_if_enabled() -> HierarchicalPipeline | None:
    """Get hierarchical pipeline if enabled in config.

    Convenience function for integration with existing code.

    Returns:
        HierarchicalPipeline if hierarchical mode is enabled, None otherwise.
    """
    config = load_hierarchical_config()
    if config.is_hierarchical:
        return HierarchicalPipeline(config)
    return None

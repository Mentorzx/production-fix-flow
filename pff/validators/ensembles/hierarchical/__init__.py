"""
Hierarchical Ensemble Package.

This package implements a hierarchical neuro-symbolic ensemble architecture
that processes neural and symbolic signals through separate aggregation
pathways before combining them via a decision router.

Design Patterns Applied:
    - **Strategy Pattern:** Different aggregation strategies (Noisy-OR, Max, Voting)
      for symbolic and neural signals.
    - **Factory Pattern:** Creation of aggregators and decision routers from config.
    - **Template Method:** Base aggregation workflow with customizable combination logic.
    - **Observer Pattern:** Event-driven metrics collection during inference.

Architecture:
    1. SymbolicAggregator: Combines rule-based predictions (default: Noisy-OR)
    2. NeuralAggregator: Combines embedding-based predictions
    3. DecisionRouter: Routes to SYMBOLIC_DECIDES, NEURAL_FALLBACK, or BLEND

Reference:
    - SAFRAN: Symbolic aggregation via Noisy-OR (AnyBURL literature)
    - Hierarchical ensemble architecture per SOTA 2024-2025 neuro-symbolic research

Module Structure:
    - config_loader: Configuration loading and defaults
    - symbolic_aggregator: Rule confidence aggregation
    - neural_aggregator: Embedding score aggregation
    - decision_router: Routing logic based on thresholds
"""

from pff.validators.ensembles.hierarchical.config_loader import (
    AggregatorConfig,
    DecisionRouterConfig,
    HierarchicalConfig,
    NeuralAggregatorConfig,
    load_hierarchical_config,
)
from pff.validators.ensembles.hierarchical.decision_router import (
    DecisionRouter,
    RoutingDecision,
    RoutingResult,
    RoutingStatistics,
)
from pff.validators.ensembles.hierarchical.neural_aggregator import (
    NeuralAggregationResult,
    NeuralAggregationStrategy,
    NeuralAggregator,
    NeuralAggregatorFactory,
    compute_entropy_confidence,
    compute_entropy_confidence_batch,
)
from pff.validators.ensembles.hierarchical.pipeline import (
    HierarchicalPipeline,
    HierarchicalPredictionResult,
    get_hierarchical_pipeline_if_enabled,
)
from pff.validators.ensembles.hierarchical.symbolic_aggregator import (
    AggregationResult,
    AggregationStrategy,
    SymbolicAggregator,
    SymbolicAggregatorFactory,
)

__all__ = [
    # Config
    "AggregatorConfig",
    "DecisionRouterConfig",
    "HierarchicalConfig",
    "NeuralAggregatorConfig",
    "load_hierarchical_config",
    # Symbolic
    "SymbolicAggregator",
    "SymbolicAggregatorFactory",
    "AggregationStrategy",
    "AggregationResult",
    # Neural
    "NeuralAggregator",
    "NeuralAggregatorFactory",
    "NeuralAggregationStrategy",
    "NeuralAggregationResult",
    "compute_entropy_confidence",
    "compute_entropy_confidence_batch",
    # Router
    "DecisionRouter",
    "RoutingDecision",
    "RoutingResult",
    "RoutingStatistics",
    # Pipeline
    "HierarchicalPipeline",
    "HierarchicalPredictionResult",
    "get_hierarchical_pipeline_if_enabled",
]

"""
Ensemble Wrappers - sklearn-compatible wrappers for ensemble methods

Refactored in Sprint 4 into specialized modules for better maintainability.

This package provides wrapper classes that adapt RotatE, Symbolic (AnyBURL),
and Hybrid models to work seamlessly with scikit-learn's ensemble methods.
"""

from .base_wrapper import (
    BaseWrapper,
    _coerce_mapping_df,
    evaluate_and_save_metrics,
    get_shared_cache,
)
from .model_wrappers import HybridWrapper, RotatEWrapper, TransEWrapper
from .transformers import (
    GraphStructuralFeatureExtractor,
    ProbaTransformer,
    SymbolicFeatureExtractor,
)

__all__ = [
    # Base classes and utilities
    "BaseWrapper",
    "get_shared_cache",
    "evaluate_and_save_metrics",
    "_coerce_mapping_df",
    # Model wrappers
    "RotatEWrapper",
    "TransEWrapper",  # Alias for backward compatibility
    "HybridWrapper",
    # Transformers
    "ProbaTransformer",
    "SymbolicFeatureExtractor",
    "GraphStructuralFeatureExtractor",
]

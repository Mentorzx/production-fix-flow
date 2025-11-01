"""
Refactored Transformers - sklearn-compatible transformers with improved design.

This module contains refactored transformers using modern design patterns:
- ProbaTransformer (extracts probabilities as features)
- SymbolicFeatureExtractorV2 (rule-based feature extraction with Strategy pattern)

Design Patterns Used:
- Strategy: Different processing strategies for symbolic features
- Factory: Creation of appropriate processors
- Builder: Configuration management
- Command: Debug and validation operations
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .processors import (
    ProcessorConfig,
    ProcessorConfigBuilder,
    ProcessorFactory,
    SymbolicFeatureProcessorV2,
)
from .processors.factory import default_factory


class ProbaTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that wraps a classifier and extracts the probability of the
    positive class as a feature.
    """

    def __init__(self, model: Any):
        self.model = model

    def fit(self, X: Any, y: Any = None) -> ProbaTransformer:
        """Fit the underlying model."""
        self._is_fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        """
        Run predict_proba and return the probability of the positive class,
        reshaped for FeatureUnion.
        """
        check_is_fitted(self, "_is_fitted")
        proba = self.model.predict_proba(X)
        return proba[:, 1].reshape(-1, 1)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the name of the output feature."""
        model_name = type(getattr(self, "model", object)).__name__
        return [f"{model_name}_proba"]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_is_fitted"] = getattr(self, "_is_fitted", False)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "_is_fitted"):
            self._is_fitted = (
                hasattr(self, "model")
                and self.model is not None
                and hasattr(self.model, "classes_")
            )


class SymbolicFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that converts samples of triples into binary
    feature vectors based on symbolic rule violations.

    This is a compatibility wrapper around the new SymbolicFeatureProcessorV2
    that maintains the original API while using the improved internal implementation.
    """

    def __init__(
        self,
        rules_path: str,
        min_confidence_threshold: float = 0.0,
        enable_grouping: bool = False,
        n_groups: int = 50,
        boost_factor: float = 10.0,
        enable_rule_indexing: bool = True,
        enable_numba: bool = True,
    ):
        # Build configuration using the Builder pattern
        config = (
            ProcessorConfigBuilder(rules_path)
            .with_confidence_threshold(min_confidence_threshold)
            .with_feature_grouping(enable_grouping, n_groups, boost_factor)
            .with_rule_indexing(enable_rule_indexing)
            .with_numba(enable_numba)
            .build()
        )

        self.config = config
        self._processor = None

        # Legacy attributes for backward compatibility
        self.rules_ = []
        self.concurrency_manager = None
        self.group_indices_ = None
        self.rule_index_ = None
        self.numba_accelerator_ = None

    def fit(self, X: list[Any], y: Any = None) -> SymbolicFeatureExtractor:
        """Fit the transformer to the data."""
        # Create processor on first fit to handle configuration-dependent initialization
        if self._processor is None:
            data_size = len(X) if X else 0
            self._processor = default_factory.create_optimized_processor(
                self.config, data_size
            )

        # Fit the processor
        self._processor.fit(X, y)

        # Update legacy attributes for backward compatibility
        self._update_legacy_attributes()

        return self

    def transform(self, X: list[list[Any]]) -> np.ndarray:
        """Transform input samples into binary feature vectors."""
        if self._processor is None:
            raise ValueError("Transformer must be fitted before transform")

        result = self._processor.transform(X)

        # Ensure the result is a numpy array
        if not isinstance(result, np.ndarray):
            result = np.array(result)

        return result

    def fit_transform(self, X: list[Any], y: Any = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names."""
        if self._processor is None:
            raise ValueError("Transformer must be fitted before getting feature names")

        return self._processor.get_feature_names_out(input_features)

    def _update_legacy_attributes(self) -> None:
        """Update legacy attributes for backward compatibility."""
        if self._processor:
            # Copy attributes from processor to maintain backward compatibility
            self.rules_ = getattr(self._processor, 'rules_', [])
            self.group_indices_ = getattr(self._processor, 'group_indices_', None)
            self.rule_index_ = getattr(self._processor, 'rule_index_', None)
            self.numba_accelerator_ = getattr(self._processor, 'numba_accelerator_', None)

            # For concurrency_manager, we'll create a dummy one if needed
            if self.concurrency_manager is None:
                from pff.utils import ConcurrencyManager
                self.concurrency_manager = ConcurrencyManager()

    def __getstate__(self) -> dict:
        """Get state for pickling."""
        state = self.__dict__.copy()
        # Don't pickle the processor directly
        state['_processor'] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Set state from pickling."""
        self.__dict__.update(state)
        # Processor will be recreated on next use


# Convenience functions for creating transformers with common configurations
def create_high_performance_extractor(
    rules_path: str,
    **kwargs,
) -> SymbolicFeatureExtractor:
    """Create a high-performance symbolic feature extractor."""
    config = ProcessorConfigBuilder(rules_path).with_feature_grouping(
        enabled=True, n_groups=100
    ).with_numba(enabled=True).with_performance(
        parallel_threshold=50, batch_size=2000
    ).build()

    extractor = SymbolicFeatureExtractor.__new__(SymbolicFeatureExtractor)
    extractor.config = config
    extractor._processor = None
    extractor.rules_ = []
    extractor.concurrency_manager = None
    extractor.group_indices_ = None
    extractor.rule_index_ = None
    extractor.numba_accelerator_ = None

    # Set additional kwargs if provided
    for key, value in kwargs.items():
        if hasattr(extractor, key):
            setattr(extractor, key, value)

    return extractor


def create_debug_extractor(
    rules_path: str,
    debug_dir: str = "debug",
    **kwargs,
) -> SymbolicFeatureExtractor:
    """Create a symbolic feature extractor optimized for debugging."""
    config = ProcessorConfigBuilder(rules_path).with_debug(
        enabled=True, output_dir=debug_dir
    ).with_performance(
        parallel_threshold=1000
    ).build()

    extractor = SymbolicFeatureExtractor.__new__(SymbolicFeatureExtractor)
    extractor.config = config
    extractor._processor = None
    extractor.rules_ = []
    extractor.concurrency_manager = None
    extractor.group_indices_ = None
    extractor.rule_index_ = None
    extractor.numba_accelerator_ = None

    # Set additional kwargs if provided
    for key, value in kwargs.items():
        if hasattr(extractor, key):
            setattr(extractor, key, value)

    return extractor


# Export the main classes
__all__ = [
    "ProbaTransformer",
    "SymbolicFeatureExtractor",
    "SymbolicFeatureProcessorV2",
    "ProcessorConfig",
    "ProcessorConfigBuilder",
    "ProcessorFactory",
    "create_high_performance_extractor",
    "create_debug_extractor",
]
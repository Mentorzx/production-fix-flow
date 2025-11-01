"""
Refactored Symbolic Feature Processor using design patterns.

This module provides a clean, maintainable implementation of the
SymbolicFeatureExtractor using Strategy, Factory, and Command patterns.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from .base import FeatureProcessor, ProcessingResult, ProcessingStrategy
from .config import ProcessorConfig
from .debug import DebugManager
from .validation import ValidationManager


class SymbolicFeatureProcessorV2(BaseEstimator, TransformerMixin, FeatureProcessor):
    """
    Refactored symbolic feature processor using design patterns.

    This class replaces the original SymbolicFeatureExtractor with a cleaner,
    more maintainable implementation using:
    - Strategy Pattern: Different processing approaches
    - Command Pattern: Debug and validation operations
    - Dependency Injection: Configurable components
    """

    def __init__(self, config: ProcessorConfig):
        """Initialize the processor with configuration."""
        super().__init__(config.__dict__)
        self.config = config
        self._strategies: list[ProcessingStrategy] = []
        self._debug_manager = DebugManager(config.enable_debug)
        self._validation_manager = ValidationManager()

        # Internal state
        self.rules_ = []
        self.rule_index_ = None
        self.numba_accelerator_ = None
        self.group_indices_ = None
        self._is_fitted = False

    def set_strategies(self, strategies: list[ProcessingStrategy]) -> None:
        """Set the processing strategies to use."""
        self._strategies = strategies

    def fit(self, X: list[Any], y: Any = None) -> SymbolicFeatureProcessorV2:
        """Fit the processor to the data."""
        start_time = time.time()

        try:
            # Validate input
            validation_context = {"config": self.config.__dict__}
            if not self._validation_manager.validate("samples", X, validation_context):
                raise ValueError("Invalid input samples")

            # Load rules
            self._load_rules()

            # Build rule index if enabled
            if self.config.enable_rule_indexing:
                self._build_rule_index()

            # Initialize Numba accelerator if enabled
            if self.config.enable_numba:
                self._initialize_numba_accelerator()

            # Initialize feature grouping if enabled
            if self.config.enable_grouping:
                self._initialize_feature_grouping()

            self._is_fitted = True
            fit_time = time.time() - start_time

            self._debug_manager.execute_command(
                "log_performance",
                {
                    "strategy": "fit",
                    "processing_time": fit_time,
                    "samples_processed": len(X) if X else 0,
                    "metadata": {"n_rules": len(self.rules_), "fitted": True},
                }
            )

            return self

        except Exception as e:
            # Debug information for failed fit
            self._debug_manager.execute_command(
                "save_debug",
                {
                    "data": X,
                    "error": e,
                    "config": self.config.__dict__,
                    "strategy": "fit",
                    "processing_time": time.time() - start_time,
                }
            )
            raise

    def transform(self, X: list[Any]) -> NDArray[np.int8]:
        """Transform input samples into binary feature vectors."""
        if not self._is_fitted:
            raise ValueError("Processor must be fitted before transform")

        start_time = time.time()

        try:
            # Validate input
            validation_context = {"config": self.config.__dict__}
            if not self._validation_manager.validate("samples", X, validation_context):
                raise ValueError("Invalid input samples")

            # Try each strategy in order
            for strategy in self._strategies:
                if strategy.can_process(X, self._get_processing_context()):
                    result = strategy.process(X, self._get_processing_context())

                    if result.is_valid():
                        # Apply feature grouping if enabled
                        if self.config.enable_grouping:
                            result.data = self._apply_feature_grouping(result.data)

                        # Log success
                        self._debug_manager.execute_command(
                            "log_performance",
                            {
                                "strategy": strategy.get_name(),
                                "processing_time": result.processing_time,
                                "samples_processed": len(X),
                                "metadata": result.metadata,
                            }
                        )

                        return result.data if isinstance(result.data, np.ndarray) else np.vstack(result.data)

            # If no strategy succeeded, raise an error
            raise RuntimeError("No processing strategy succeeded")

        except Exception as e:
            # Debug information for failed transform
            self._debug_manager.execute_command(
                "save_debug",
                {
                    "data": X,
                    "error": e,
                    "config": self.config.__dict__,
                    "strategy": "transform",
                    "processing_time": time.time() - start_time,
                }
            )
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        return {
            "n_rules": len(self.rules_),
            "has_rule_index": self.rule_index_ is not None,
            "has_numba_accelerator": self.numba_accelerator_ is not None,
            "has_feature_grouping": self.group_indices_ is not None,
            "strategies_available": [s.get_name() for s in self._strategies],
            "is_fitted": self._is_fitted,
            "config": self.config.__dict__,
        }

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names."""
        if not self._is_fitted:
            raise ValueError("Processor must be fitted before getting feature names")

        if self.config.enable_grouping and self.group_indices_ is not None:
            # Grouped features
            feature_names = []
            for i, group_indices in enumerate(self.group_indices_):
                feature_names.extend([
                    f"group_{i}_proportion",
                    f"group_{i}_any_active",
                    f"group_{i}_count_normalized",
                ])
            # Add global features
            feature_names.extend(["global_proportion", "global_count_normalized"])
            return feature_names
        else:
            # Individual rule features
            return [f"rule_{i}" for i in range(len(self.rules_))]

    def _load_rules(self) -> None:
        """Load rules from the specified path."""
        from pff.utils import FileManager

        try:
            if self.config.rules_path.endswith('.json'):
                rules_data = FileManager.read_json(self.config.rules_path)
                if isinstance(rules_data, dict) and "rules" in rules_data:
                    self.rules_ = rules_data["rules"]
                else:
                    self.rules_ = rules_data
            else:
                # Assume TSV format
                df = FileManager.read_tsv(self.config.rules_path)
                self.rules_ = df.to_dict("records")

            # Filter rules by confidence
            if self.config.min_confidence_threshold > 0:
                self.rules_ = [
                    rule for rule in self.rules_
                    if rule.get("confidence", 0) >= self.config.min_confidence_threshold
                ]

        except Exception as e:
            raise RuntimeError(f"Failed to load rules from {self.config.rules_path}: {e}")

    def _build_rule_index(self) -> None:
        """Build an index mapping predicates to rule indices for efficient processing."""
        self.rule_index_ = {}
        for i, rule in enumerate(self.rules_):
            # Extract predicate from rule head
            head = rule.get("head", {})
            predicate = head.get("predicate", "")
            if predicate:
                if predicate not in self.rule_index_:
                    self.rule_index_[predicate] = []
                self.rule_index_[predicate].append(i)

    def _initialize_numba_accelerator(self) -> None:
        """Initialize the Numba accelerator if available."""
        try:
            from pff.utils import SymbolicRuleAccelerator

            self.numba_accelerator_ = SymbolicRuleAccelerator(
                self.rules_,
                enable_numba=True,
            )

            # Configure Numba strategy with the accelerator
            for strategy in self._strategies:
                if hasattr(strategy, 'set_accelerator'):
                    strategy.set_accelerator(self.numba_accelerator_)

        except Exception as e:
            from pff.utils.core.logger import logger
            logger.warning(f"Failed to initialize Numba accelerator: {e}")
            self.numba_accelerator_ = None

    def _initialize_feature_grouping(self) -> None:
        """Initialize feature grouping for dimensionality reduction."""
        n_features = len(self.rules_)
        if n_features > 0:
            self.group_indices_ = self._create_feature_groups(n_features)

    def _create_feature_groups(self, n_features: int) -> list[list[int]]:
        """Create feature groups for dimensionality reduction."""
        if n_features <= self.config.n_groups:
            # If we have fewer features than groups, one feature per group
            return [[i] for i in range(n_features)]
        else:
            # Distribute features evenly across groups
            base_size = n_features // self.config.n_groups
            remainder = n_features % self.config.n_groups

            groups = []
            start_idx = 0
            for i in range(self.config.n_groups):
                group_size = base_size + (1 if i < remainder else 0)
                end_idx = start_idx + group_size
                groups.append(list(range(start_idx, end_idx)))
                start_idx = end_idx

            return groups

    def _apply_feature_grouping(self, data: NDArray[np.int8] | list[NDArray[np.int8]]) -> NDArray[np.int8]:
        """Apply feature grouping to reduce dimensionality."""
        if not self.group_indices_:
            return data if isinstance(data, np.ndarray) else np.vstack(data)

        # Convert to numpy array if needed
        if isinstance(data, list):
            binary_features = np.vstack(data)
        else:
            binary_features = data

        grouped_features = []
        for group_indices in self.group_indices_:
            group_data = binary_features[:, group_indices]
            proportion = np.mean(group_data, axis=1, keepdims=True)
            any_active = np.any(group_data, axis=1, keepdims=True).astype(float)
            count_normalized = np.sum(group_data, axis=1, keepdims=True) / len(group_indices)

            grouped_features.extend([
                proportion * self.config.boost_factor,
                any_active * self.config.boost_factor,
                count_normalized * self.config.boost_factor,
            ])

        # Add global features
        global_features = [
            np.mean(binary_features, axis=1, keepdims=True) * self.config.boost_factor,
            np.sum(binary_features, axis=1, keepdims=True) / binary_features.shape[1] * self.config.boost_factor,
        ]
        grouped_features.extend(global_features)

        return np.hstack(grouped_features).astype(np.int8)

    def _get_processing_context(self) -> dict[str, Any]:
        """Get the processing context for strategies."""
        return {
            "config": self.config.__dict__,
            "rules": self.rules_,
            "rule_index": self.rule_index_,
            "n_rules": len(self.rules_),
            "enable_grouping": self.config.enable_grouping,
            "boost_factor": self.config.boost_factor,
            "group_indices": self.group_indices_,
        }

    # Compatibility methods for the original interface
    def __getstate__(self) -> dict:
        """Get state for pickling."""
        state = self.__dict__.copy()
        state["_debug_manager"] = None  # Don't pickle debug manager
        return state

    def __setstate__(self, state: dict) -> None:
        """Set state from pickling."""
        self.__dict__.update(state)
        # Recreate debug manager
        if hasattr(self, 'config'):
            self._debug_manager = DebugManager(self.config.enable_debug)
        else:
            self._debug_manager = DebugManager(False)
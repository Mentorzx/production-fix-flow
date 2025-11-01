"""
Processing strategies for symbolic feature extraction.

This module implements the Strategy pattern for different approaches
to processing symbolic rules and violations.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import ProcessingResult, ProcessingStrategy

# Import context variables from the original transformers module
try:
    from ...transformers import _ensemble_violations_context, _ensemble_all_rules_context
except ImportError:
    # Fallback for when imported from v2
    from ..transformers import _ensemble_violations_context, _ensemble_all_rules_context


class NumbaProcessingStrategy(ProcessingStrategy):
    """Strategy using Numba-accelerated processing."""

    def __init__(self):
        self.accelerator = None

    def can_process(self, data: list[Any], config: dict[str, Any]) -> bool:
        """Check if Numba processing is available and suitable."""
        return (
            config.get("enable_numba", False)
            and len(data) > 0
            and self.accelerator is not None
        )

    def process(self, data: list[Any], config: dict[str, Any]) -> ProcessingResult:
        """Process data using Numba acceleration."""
        start_time = time.time()

        try:
            # Normalize data for Numba
            normalized_data = self._normalize_for_numba(data)

            # Use larger batch size for better performance with many samples
            batch_size = 2000 if len(normalized_data) > 5000 else 1000

            violations_list = self.accelerator.check_violations_batch(
                normalized_data,
                use_parallel=True,
                batch_size=batch_size,
            )

            # Validate results
            if self._validate_violations_list(violations_list, config):
                processing_time = time.time() - start_time
                return ProcessingResult(
                    data=[np.asarray(v).astype(np.int8).ravel() for v in violations_list],
                    success=True,
                    metadata={
                        "strategy": "numba",
                        "samples_processed": len(data),
                        "batch_size": batch_size,
                    },
                    processing_time=processing_time,
                )
            else:
                raise ValueError("Invalid violations list returned")

        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                data=[],
                success=False,
                metadata={"strategy": "numba", "error": str(e)},
                processing_time=processing_time,
                error_message=str(e),
            )

    def get_name(self) -> str:
        """Get strategy name."""
        return "Numba"

    def set_accelerator(self, accelerator) -> None:
        """Set the Numba accelerator."""
        self.accelerator = accelerator

    def _normalize_for_numba(self, data: list[Any]) -> list[Any]:
        """Normalize data for Numba processing."""
        normalized = []
        for sample in data:
            if isinstance(sample, np.ndarray):
                seq = sample.tolist()
            else:
                seq = sample

            normalized_sample = []
            for t in seq:
                if isinstance(t, np.ndarray):
                    inner = t.tolist()
                else:
                    inner = t

                if isinstance(inner, (list, tuple)):
                    tup = tuple("" if x is None else str(x) for x in inner)
                else:
                    tup = (str(inner),)

                normalized_sample.append(tup)
            normalized.append(normalized_sample)
        return normalized

    def _validate_violations_list(self, violations_list: list, config: dict[str, Any]) -> bool:
        """Validate violations list format."""
        try:
            if not isinstance(violations_list, (list, tuple)):
                return False

            n_rules = config.get("n_rules", 0)
            if n_rules == 0:
                return True

            for v in violations_list:
                arr = np.asarray(v)
                if arr.ndim != 1 or arr.shape[0] != n_rules:
                    return False

            return True
        except Exception:
            return False


class ParallelProcessingStrategy(ProcessingStrategy):
    """Strategy using parallel processing with ConcurrencyManager."""

    def __init__(self):
        from pff.utils import ConcurrencyManager
        self.concurrency_manager = ConcurrencyManager()

    def can_process(self, data: list[Any], config: dict[str, Any]) -> bool:
        """Check if parallel processing is suitable."""
        return len(data) >= config.get("parallel_threshold", 100)

    def process(self, data: list[Any], config: dict[str, Any]) -> ProcessingResult:
        """Process data using parallel execution."""
        start_time = time.time()

        try:
            # Prepare data for parallel processing
            sample_data = [(sample, config.get("rules", []), config.get("rule_index", {})) for sample in data]

            # Execute parallel processing
            results = self.concurrency_manager.execute_sync(
                self._transform_single_sample_indexed,
                sample_data,
                desc="Processando Regras Simbólicas (Parallel)",
                task_type="process",
            )

            processing_time = time.time() - start_time
            return ProcessingResult(
                data=np.array(results, dtype=np.int8),
                success=True,
                metadata={
                    "strategy": "parallel",
                    "samples_processed": len(data),
                    "workers_used": self.concurrency_manager._pool._max_workers if hasattr(self.concurrency_manager, '_pool') else "unknown",
                },
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                data=[],
                success=False,
                metadata={"strategy": "parallel", "error": str(e)},
                processing_time=processing_time,
                error_message=str(e),
            )

    def get_name(self) -> str:
        """Get strategy name."""
        return "Parallel"

    @staticmethod
    def _transform_single_sample_indexed(args: tuple) -> NDArray[np.int8]:
        """Transform a single sample using indexed rule processing."""
        sample_triples_list, rules, rule_index = args

        available_triples_set = {tuple(map(str, t)) for t in sample_triples_list}
        sample_feature_vector = np.zeros(len(rules), dtype=np.int8)

        # Use rule index for efficient processing
        if rule_index:
            for predicate, triples in sample_triples_list:
                predicate = str(predicate)
                if predicate in rule_index:
                    # Only check relevant rules for this predicate
                    for rule_idx in rule_index[predicate]:
                        if rule_idx < len(rules):
                            rule = rules[rule_idx]
                            if ParallelProcessingStrategy._rule_is_violated(rule, available_triples_set):
                                sample_feature_vector[rule_idx] = 1
        else:
            # Fallback to checking all rules
            for i, rule in enumerate(rules):
                if ParallelProcessingStrategy._rule_is_violated(rule, available_triples_set):
                    sample_feature_vector[i] = 1

        return sample_feature_vector

    @staticmethod
    def _rule_is_violated(rule: dict, available_triples: set) -> bool:
        """Check if a rule is violated by the available triples."""
        try:
            # This is a simplified implementation
            # In practice, this would implement proper rule matching logic
            body_atoms = rule.get("body", [])
            return any(
                tuple(map(str, atom.values())) in available_triples
                for atom in body_atoms
                if isinstance(atom, dict) and "subject" in atom and "predicate" in atom and "object" in atom
            )
        except Exception:
            return False


class IndexedProcessingStrategy(ProcessingStrategy):
    """Strategy using indexed rule processing for efficiency."""

    def can_process(self, data: list[Any], config: dict[str, Any]) -> bool:
        """Check if indexed processing is available."""
        return config.get("rule_index") is not None and config.get("enable_rule_indexing", False)

    def process(self, data: list[Any], config: dict[str, Any]) -> ProcessingResult:
        """Process data using indexed rule processing."""
        # Similar to ParallelProcessingStrategy but optimized for indexed access
        # Implementation would be similar to the parallel strategy but with better indexing
        pass  # Placeholder - would implement similar to parallel but with index optimization

    def get_name(self) -> str:
        """Get strategy name."""
        return "Indexed"


class SequentialProcessingStrategy(ProcessingStrategy):
    """Strategy using sequential processing (fallback)."""

    def can_process(self, data: list[Any], config: dict[str, Any]) -> bool:
        """Sequential processing can always handle data."""
        return True

    def process(self, data: list[Any], config: dict[str, Any]) -> ProcessingResult:
        """Process data sequentially."""
        start_time = time.time()

        try:
            rules = config.get("rules", [])
            results = []

            for sample in data:
                # Simple sequential processing
                sample_result = self._process_sample(sample, rules)
                results.append(sample_result)

            processing_time = time.time() - start_time
            return ProcessingResult(
                data=results,
                success=True,
                metadata={
                    "strategy": "sequential",
                    "samples_processed": len(data),
                },
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                data=[],
                success=False,
                metadata={"strategy": "sequential", "error": str(e)},
                processing_time=processing_time,
                error_message=str(e),
            )

    def get_name(self) -> str:
        """Get strategy name."""
        return "Sequential"

    def _process_sample(self, sample: Any, rules: list) -> NDArray[np.int8]:
        """Process a single sample sequentially."""
        # Simple implementation - would need proper rule matching logic
        return np.zeros(len(rules), dtype=np.int8)


class ContextBasedStrategy(ProcessingStrategy):
    """Strategy that tries to use pre-calculated violations from context."""

    def can_process(self, data: list[Any], config: dict[str, Any]) -> bool:
        """Check if context violations are available."""
        if not config.get("use_context_violations", True):
            return False

        try:
            violations = _ensemble_violations_context.get()
            all_rules = _ensemble_all_rules_context.get()
            return (
                violations is not None
                and all_rules is not None
                and len(violations) > 0
                and len(all_rules) > 0
                and len(violations) == len(data)
            )
        except Exception:
            return False

    def process(self, data: list[Any], config: dict[str, Any]) -> ProcessingResult:
        """Process data using pre-calculated violations from context."""
        start_time = time.time()

        try:
            violations = _ensemble_violations_context.get()
            all_rules = _ensemble_all_rules_context.get()

            binary_features = self._violations_to_binary_features(
                violations, all_rules, len(data), config
            )

            processing_time = time.time() - start_time
            return ProcessingResult(
                data=binary_features,
                success=True,
                metadata={
                    "strategy": "context",
                    "samples_processed": len(data),
                    "violations_from_context": True,
                },
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                data=[],
                success=False,
                metadata={"strategy": "context", "error": str(e)},
                processing_time=processing_time,
                error_message=str(e),
            )

    def get_name(self) -> str:
        """Get strategy name."""
        return "Context"

    def _violations_to_binary_features(
        self,
        violations: list,
        all_rules: list,
        n_samples: int,
        config: dict[str, Any],
    ) -> NDArray[np.int8]:
        """Convert violations list to binary feature matrix."""
        # This would implement the conversion logic from the original transformer
        # For now, return a simple implementation
        return np.zeros((n_samples, len(all_rules)), dtype=np.int8)
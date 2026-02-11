"""
Generic Loop Accelerator - Automatic optimization of Python loops using multiple strategies.

This module provides a unified interface for accelerating Python loops using:
- NumPy vectorization (2-10× speedup)
- Parallel execution via concurrency.py
- Pure Python baseline

Design Patterns Used:
- Strategy Pattern: Different acceleration strategies (Vectorized, Parallel, Pure Python)
- Factory Pattern: Creates appropriate accelerator based on configuration
- Template Method: Standard prepare → execute → postprocess flow
- Adapter Pattern: Adapts user functions to accelerated implementations
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar, cast

import numpy as np

from ..core.logging import logger
from .concurrency import ConcurrencyManager

T = TypeVar("T")
R = TypeVar("R")


class AcceleratorBackend(Enum):
    """Available acceleration backends."""

    VECTORIZED = "vectorized"
    PARALLEL = "parallel"
    PYTHON = "python"


@dataclass
class AcceleratorConfig:
    """Configuration for loop accelerator."""

    backend: AcceleratorBackend = AcceleratorBackend.PARALLEL
    parallel: bool = True
    cache: bool = True

    chunk_size: int = 1000
    max_workers: int | None = None

    profile: bool = False
    verbose: bool = False


class AcceleratorStrategy(ABC, Generic[T, R]):
    """Abstract base class for acceleration strategies."""

    def __init__(self, config: AcceleratorConfig):
        self.config = config
        self.stats = {"calls": 0, "total_time": 0.0, "items_processed": 0}

    @abstractmethod
    def execute(self, func: Callable[[T], R], items: list[T], **kwargs) -> list[R]:
        """Execute function on all items using this strategy."""
        pass

    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {"calls": 0, "total_time": 0.0, "items_processed": 0}

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()
        if stats["calls"] > 0:
            stats["avg_time_per_call"] = stats["total_time"] / stats["calls"]
            stats["items_per_second"] = (
                stats["items_processed"] / stats["total_time"]
                if stats["total_time"] > 0
                else 0
            )
        return stats


class VectorizedStrategy(AcceleratorStrategy[T, R]):
    """Acceleration strategy using NumPy vectorization."""

    def execute(self, func: Callable[[T], R], items: list[T], **kwargs) -> list[R]:
        """Execute using NumPy vectorization."""
        start_time = time.time()

        try:
            items_array = np.asarray(items)
            results_array = func(items_array, **kwargs)  # type: ignore[arg-type]
            results_array = np.asarray(results_array)  # type: ignore[assignment]
            if results_array.shape != items_array.shape:  # type: ignore[attr-defined]
                raise ValueError("Vectorized function returned non-elementwise output")
            results = results_array.tolist()  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(
                f" Vectorization failed: {e}, falling back to list comprehension"
            )
            results = [func(item, **kwargs) for item in items]

        elapsed = time.time() - start_time
        self.stats["calls"] += 1
        self.stats["total_time"] += elapsed
        self.stats["items_processed"] += len(items)

        if self.config.verbose:
            logger.debug(
                f" Vectorized executed {len(items)} items in {elapsed:.3f}s "
                f"({len(items) / elapsed:.1f} items/s)"
            )

        return results  # type: ignore[no-any-return]


class ParallelStrategy(AcceleratorStrategy[T, R]):
    """Acceleration strategy using CPU parallelism."""

    def __init__(self, config: AcceleratorConfig):
        super().__init__(config)
        self.concurrency = ConcurrencyManager()

    @staticmethod
    def _worker_func(item: T, kwargs: dict[str, Any], func: Callable[[T], R]) -> R:
        """
        Worker function that unpacks kwargs and calls the original function.

        This must be a static method to be pickleable for ProcessPoolExecutor.
        """
        return func(item, **kwargs)

    def execute(self, func: Callable[[T], R], items: list[T], **kwargs) -> list[R]:
        """Execute using CPU parallelism."""
        start_time = time.time()

        task_data = [(item, kwargs, func) for item in items]

        results = self.concurrency.execute_sync(
            self._worker_func,
            task_data,
            desc="Loop Accelerator (Parallel)",
            task_type="process",
        )

        elapsed = time.time() - start_time
        self.stats["calls"] += 1
        self.stats["total_time"] += elapsed
        self.stats["items_processed"] += len(items)

        if self.config.verbose:
            logger.debug(
                f" Parallel executed {len(items)} items in {elapsed:.3f}s "
                f"({len(items) / elapsed:.1f} items/s)"
            )

        return results


class PythonStrategy(AcceleratorStrategy[T, R]):
    """Baseline strategy using pure Python."""

    def execute(self, func: Callable[[T], R], items: list[T], **kwargs) -> list[R]:
        """Execute using pure Python."""
        start_time = time.time()

        results = [func(item, **kwargs) for item in items]

        elapsed = time.time() - start_time
        self.stats["calls"] += 1
        self.stats["total_time"] += elapsed
        self.stats["items_processed"] += len(items)

        if self.config.verbose:
            logger.debug(
                f" Python executed {len(items)} items in {elapsed:.3f}s "
                f"({len(items) / elapsed:.1f} items/s)"
            )

        return results


class LoopAcceleratorFactory:
    """Factory for creating appropriate acceleration strategies."""

    @staticmethod
    def create(config: AcceleratorConfig) -> AcceleratorStrategy:
        """Create acceleration strategy based on config and availability."""
        if config.backend == AcceleratorBackend.VECTORIZED:
            return VectorizedStrategy(config)

        if config.backend == AcceleratorBackend.PARALLEL:
            return ParallelStrategy(config)

        return PythonStrategy(config)


class LoopAccelerator(Generic[T, R]):
    """
    Generic loop accelerator with automatic optimization selection.

    Example:
        >>> def check_rule(triple):
        ...     return triple[0] == triple[2]
        >>>
        >>> triples = [(1, 2, 1), (3, 4, 5), (6, 7, 6)]
        >>> accelerator = LoopAccelerator()
        >>> results = accelerator.map(check_rule, triples)
        >>> results
        [True, False, True]
    """

    def __init__(
        self, config: AcceleratorConfig | None = None, encoder: Any | None = None
    ):
        """
        Initialize loop accelerator.

        Args:
            config: Acceleration configuration (defaults to Rust with parallel)
            encoder: Optional encoder for converting complex types to Rust-compatible types
        """
        self.config = config or AcceleratorConfig()
        self.encoder = encoder
        self.strategy = LoopAcceleratorFactory.create(self.config)

    def map(self, func: Callable[[T], R], items: list[T], **kwargs) -> list[R]:
        """
        Apply function to all items with automatic acceleration.

        Args:
            func: Function to apply to each item
            items: List of items to process
            **kwargs: Additional keyword arguments passed to func

        Returns:
            List of results
        """
        if not items:
            return []

        start_time = time.time()

        results = self.strategy.execute(func, items, **kwargs)

        if self.config.profile:
            elapsed = time.time() - start_time
            logger.debug(
                f"LoopAccelerator processed {len(items)} items in {elapsed:.3f}s "
                f"({len(items) / elapsed:.1f} items/s) using {self.config.backend.value}"
            )

        return results

    def map_batch(
        self,
        func: Callable[[list[T]], list[R]],
        items: list[T],
        batch_size: int = 1000,
        **kwargs,
    ) -> list[R]:
        """
        Apply function to batches of items.

        Args:
            func: Function that processes a batch and returns a batch of results
            items: List of items to process
            batch_size: Number of items per batch
            **kwargs: Additional keyword arguments passed to func

        Returns:
            Flattened list of results
        """
        if not items:
            return []

        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

        batch_results = cast(
            list[list[R]], self.map(cast(Any, func), cast(Any, batches), **kwargs)
        )

        results = []
        for batch_result in batch_results:
            results.extend(batch_result)

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics from current strategy."""
        return self.strategy.get_stats()

    def reset_stats(self):
        """Reset performance statistics."""
        self.strategy.reset_stats()


def accelerate_loop(
    func: Callable[[T], R],
    items: list[T],
    backend: AcceleratorBackend = AcceleratorBackend.PARALLEL,
    parallel: bool = True,
    **kwargs,
) -> list[R]:
    """
    Convenience function to accelerate a loop with minimal setup.

    Args:
        func: Function to apply to each item
        items: List of items to process
        backend: Acceleration backend to use
        parallel: Enable parallel execution
        **kwargs: Additional keyword arguments passed to func

    Returns:
        List of results
    """
    config = AcceleratorConfig(backend=backend, parallel=parallel)
    accelerator: LoopAccelerator[T, R] = LoopAccelerator(config=config)
    return accelerator.map(func, items, **kwargs)

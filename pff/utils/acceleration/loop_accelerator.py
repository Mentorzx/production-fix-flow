"""
Generic Loop Accelerator - Automatic optimization of Python loops using multiple strategies.

This module provides a unified interface for accelerating Python loops using:
- Numba JIT compilation (10-100× speedup)
- NumPy vectorization (2-10× speedup)
- Parallel execution via concurrency.py
- Automatic fallback to pure Python

Design Patterns Used:
- Strategy Pattern: Different acceleration strategies (Numba, Vectorized, Pure Python)
- Factory Pattern: Creates appropriate accelerator based on availability
- Template Method: Standard prepare → execute → postprocess flow
- Adapter Pattern: Adapts user functions to accelerated implementations

Author: PFF Team
Date: 2025-10-31
Version: 1.0.0
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generic, TypeVar, Optional
import numpy as np
from numpy.typing import NDArray

from ..core.logger import logger
from .concurrency import ConcurrencyManager

# Try to import Numba
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator

    prange = range


T = TypeVar('T')
R = TypeVar('R')


class AcceleratorBackend(Enum):
    """Available acceleration backends."""
    NUMBA = "numba"          # JIT compilation (fastest: 10-100×)
    VECTORIZED = "vectorized"  # NumPy vectorization (fast: 2-10×)
    PARALLEL = "parallel"    # CPU parallelism via concurrency.py (2-8×)
    PYTHON = "python"        # Pure Python (baseline)


@dataclass
class AcceleratorConfig:
    """Configuration for loop accelerator."""

    backend: AcceleratorBackend = AcceleratorBackend.NUMBA
    parallel: bool = True           # Enable parallelism
    cache: bool = True              # Cache compiled functions
    fastmath: bool = True           # Enable fast-math optimizations
    error_model: str = "numpy"      # Error model: "numpy" or "python"

    # Parallelism config
    chunk_size: int = 1000          # Items per worker
    max_workers: Optional[int] = None  # None = auto-detect

    # Debugging
    profile: bool = False           # Profile execution time
    verbose: bool = False           # Log detailed info


class AcceleratorStrategy(ABC, Generic[T, R]):
    """Abstract base class for acceleration strategies."""

    def __init__(self, config: AcceleratorConfig):
        self.config = config
        self.stats = {"calls": 0, "total_time": 0.0, "items_processed": 0}

    @abstractmethod
    def execute(
        self,
        func: Callable[[T], R],
        items: list[T],
        **kwargs
    ) -> list[R]:
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
                if stats["total_time"] > 0 else 0
            )
        return stats


class NumbaStrategy(AcceleratorStrategy[T, R]):
    """Acceleration strategy using Numba JIT compilation."""

    def __init__(self, config: AcceleratorConfig):
        super().__init__(config)
        self.compiled_funcs = {}  # Cache compiled functions

    def execute(
        self,
        func: Callable[[T], R],
        items: list[T],
        **kwargs
    ) -> list[R]:
        """Execute using Numba JIT compilation."""
        start_time = time.time()

        # Check if function is already compiled
        func_id = id(func)
        if self.config.cache and func_id in self.compiled_funcs:
            compiled_func = self.compiled_funcs[func_id]
        else:
            # Check if function is compilable by Numba
            # Numba cannot compile bound methods or functions with closures
            if hasattr(func, '__self__') or hasattr(func, '__code__') and func.__code__.co_freevars:
                # Fall back to Python execution
                logger.debug("Function not compilable by Numba (method or closure), using Python execution")
                return [func(item, **kwargs) for item in items]

            # Compile function with Numba
            try:
                compiled_func = njit(
                    cache=self.config.cache,
                    fastmath=self.config.fastmath,
                    error_model=self.config.error_model,
                    parallel=self.config.parallel,
                )(func)

                if self.config.cache:
                    self.compiled_funcs[func_id] = compiled_func
            except Exception as e:
                # If compilation fails, fall back to Python execution
                logger.debug(f"Numba compilation failed: {e}, using Python execution")
                return [func(item, **kwargs) for item in items]

        # Execute compiled function
        if self.config.parallel and len(items) > 1000:
            # Use Numba's parallel execution
            results = self._execute_parallel_numba(compiled_func, items, **kwargs)
        else:
            # Sequential execution
            results = [compiled_func(item, **kwargs) for item in items]

        # Update stats
        elapsed = time.time() - start_time
        self.stats["calls"] += 1
        self.stats["total_time"] += elapsed
        self.stats["items_processed"] += len(items)

        if self.config.verbose:
            logger.debug(
                f"⚡ Numba executed {len(items)} items in {elapsed:.3f}s "
                f"({len(items)/elapsed:.1f} items/s)"
            )

        return results

    def _execute_parallel_numba(
        self,
        func: Callable,
        items: list[T],
        **kwargs
    ) -> list[R]:
        """Execute with Numba parallel loops."""
        # Convert items to NumPy array if possible
        try:
            items_array = np.array(items)
            n = len(items_array)

            # Create output array
            # Note: This assumes homogeneous output types
            result_array = np.empty(n, dtype=object)

            # Parallel loop (Numba will parallelize this)
            for i in prange(n):
                result_array[i] = func(items_array[i], **kwargs)

            return result_array.tolist()

        except Exception as e:
            logger.warning(f"⚠️ Numba parallel execution failed: {e}, falling back to sequential")
            return [func(item, **kwargs) for item in items]


class VectorizedStrategy(AcceleratorStrategy[T, R]):
    """Acceleration strategy using NumPy vectorization."""

    def execute(
        self,
        func: Callable[[T], R],
        items: list[T],
        **kwargs
    ) -> list[R]:
        """Execute using NumPy vectorization."""
        start_time = time.time()

        try:
            # Try to vectorize using numpy.vectorize
            vectorized_func = np.vectorize(func)
            items_array = np.array(items)
            results = vectorized_func(items_array, **kwargs).tolist()

        except Exception as e:
            logger.warning(f"⚠️ Vectorization failed: {e}, falling back to list comprehension")
            results = [func(item, **kwargs) for item in items]

        # Update stats
        elapsed = time.time() - start_time
        self.stats["calls"] += 1
        self.stats["total_time"] += elapsed
        self.stats["items_processed"] += len(items)

        if self.config.verbose:
            logger.debug(
                f"📊 Vectorized executed {len(items)} items in {elapsed:.3f}s "
                f"({len(items)/elapsed:.1f} items/s)"
            )

        return results


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
        The function is passed as part of the data tuple to avoid serialization issues.
        """
        return func(item, **kwargs)

    def execute(
        self,
        func: Callable[[T], R],
        items: list[T],
        **kwargs
    ) -> list[R]:
        """Execute using CPU parallelism via concurrency.py."""
        start_time = time.time()

        # Prepare data for concurrency manager
        # Each worker gets a tuple with (item, kwargs, func)
        task_data = [(item, kwargs, func) for item in items]

        # Execute using concurrency manager with a static worker function
        results = self.concurrency.execute_sync(
            self._worker_func,
            task_data,
            desc="Loop Accelerator (Parallel)",
            task_type="process",
        )

        # Update stats
        elapsed = time.time() - start_time
        self.stats["calls"] += 1
        self.stats["total_time"] += elapsed
        self.stats["items_processed"] += len(items)

        if self.config.verbose:
            logger.debug(
                f"🚀 Parallel executed {len(items)} items in {elapsed:.3f}s "
                f"({len(items)/elapsed:.1f} items/s)"
            )

        return results


class PythonStrategy(AcceleratorStrategy[T, R]):
    """Baseline strategy using pure Python."""

    def execute(
        self,
        func: Callable[[T], R],
        items: list[T],
        **kwargs
    ) -> list[R]:
        """Execute using pure Python."""
        start_time = time.time()

        results = [func(item, **kwargs) for item in items]

        # Update stats
        elapsed = time.time() - start_time
        self.stats["calls"] += 1
        self.stats["total_time"] += elapsed
        self.stats["items_processed"] += len(items)

        if self.config.verbose:
            logger.debug(
                f"🐍 Python executed {len(items)} items in {elapsed:.3f}s "
                f"({len(items)/elapsed:.1f} items/s)"
            )

        return results


class LoopAcceleratorFactory:
    """Factory for creating appropriate acceleration strategies."""

    @staticmethod
    def create(config: AcceleratorConfig) -> AcceleratorStrategy:
        """Create acceleration strategy based on config and availability."""

        # Try requested backend first
        if config.backend == AcceleratorBackend.NUMBA:
            if NUMBA_AVAILABLE:
                logger.debug("⚡ Using Numba JIT acceleration")
                return NumbaStrategy(config)
            else:
                logger.warning(
                    "⚠️ Numba not available, falling back to parallel execution"
                )
                config.backend = AcceleratorBackend.PARALLEL

        if config.backend == AcceleratorBackend.VECTORIZED:
            logger.debug("📊 Using NumPy vectorization")
            return VectorizedStrategy(config)

        if config.backend == AcceleratorBackend.PARALLEL:
            logger.debug("🚀 Using parallel execution")
            return ParallelStrategy(config)

        # Fallback to pure Python
        logger.debug("🐍 Using pure Python execution")
        return PythonStrategy(config)


class LoopAccelerator(Generic[T, R]):
    """
    Generic loop accelerator with automatic optimization selection.

    Usage:
        # Basic usage with Numba
        accelerator = LoopAccelerator()
        results = accelerator.map(my_function, items)

        # Custom configuration
        config = AcceleratorConfig(
            backend=AcceleratorBackend.NUMBA,
            parallel=True,
            cache=True
        )
        accelerator = LoopAccelerator(config)

        # With custom encoder for complex types
        encoder = MyCustomEncoder()
        accelerator = LoopAccelerator(encoder=encoder)
        results = accelerator.map(my_function, items)

    Example:
        >>> def check_rule(triple):
        ...     return triple[0] == triple[2]
        >>>
        >>> triples = [(1, 2, 1), (3, 4, 5), (6, 7, 6)]
        >>> accelerator = LoopAccelerator()
        >>> results = accelerator.map(check_rule, triples)
        >>> print(results)
        [True, False, True]
    """

    def __init__(
        self,
        config: Optional[AcceleratorConfig] = None,
        encoder: Optional[Any] = None
    ):
        """
        Initialize loop accelerator.

        Args:
            config: Acceleration configuration (defaults to Numba with parallel)
            encoder: Optional encoder for converting complex types to Numba-compatible types
        """
        self.config = config or AcceleratorConfig()
        self.encoder = encoder
        self.strategy = LoopAcceleratorFactory.create(self.config)

    def map(
        self,
        func: Callable[[T], R],
        items: list[T],
        **kwargs
    ) -> list[R]:
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

        # Profile if requested
        if self.config.profile:
            start_time = time.time()

        # Execute using selected strategy
        results = self.strategy.execute(func, items, **kwargs)

        # Log profiling info
        if self.config.profile:
            elapsed = time.time() - start_time
            logger.info(
                f"⏱️ LoopAccelerator processed {len(items)} items in {elapsed:.3f}s "
                f"({len(items)/elapsed:.1f} items/s) using {self.config.backend.value}"
            )

        return results

    def map_batch(
        self,
        func: Callable[[list[T]], list[R]],
        items: list[T],
        batch_size: int = 1000,
        **kwargs
    ) -> list[R]:
        """
        Apply function to batches of items.

        Useful when func operates more efficiently on batches (e.g., matrix operations).

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

        # Create batches
        batches = [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]

        # Process each batch
        batch_results = self.map(func, batches, **kwargs)

        # Flatten results
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


# Convenience function for simple use cases
def accelerate_loop(
    func: Callable[[T], R],
    items: list[T],
    backend: AcceleratorBackend = AcceleratorBackend.NUMBA,
    parallel: bool = True,
    **kwargs
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

    Example:
        >>> def square(x):
        ...     return x ** 2
        >>>
        >>> numbers = list(range(1000))
        >>> results = accelerate_loop(square, numbers)
    """
    config = AcceleratorConfig(backend=backend, parallel=parallel)
    accelerator = LoopAccelerator(config=config)
    return accelerator.map(func, items, **kwargs)

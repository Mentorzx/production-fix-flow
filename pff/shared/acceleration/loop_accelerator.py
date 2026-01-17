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
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar, cast
from collections.abc import Callable
import numpy as np

from ..core.logger import logger
from .concurrency import ConcurrencyManager

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if args and callable(args[0]) else decorator

    prange = range


T = TypeVar("T")
R = TypeVar("R")


class AcceleratorBackend(Enum):
    """Available acceleration backends."""

    NUMBA = "numba"
    VECTORIZED = "vectorized"
    PARALLEL = "parallel"
    PYTHON = "python"


@dataclass
class AcceleratorConfig:
    """Configuration for loop accelerator."""

    backend: AcceleratorBackend = AcceleratorBackend.NUMBA
    parallel: bool = True
    cache: bool = True
    fastmath: bool = True
    error_model: str = "numpy"

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
                stats["items_processed"] / stats["total_time"] if stats["total_time"] > 0 else 0
            )
        return stats


class NumbaStrategy(AcceleratorStrategy[T, R]):
    """Acceleration strategy using Numba JIT compilation."""

    def __init__(self, config: AcceleratorConfig):
        super().__init__(config)
        self.compiled_funcs = {}
        self.compiled_batch_kernels = {}
        self.output_dtype_cache = {}

    def execute(self, func: Callable[[T], R], items: list[T], **kwargs) -> list[R]:
        """Execute using Numba JIT compilation."""
        start_time = time.time()

        compiled_func = self._get_or_compile(func)
        if compiled_func is None:
            return [func(item, **kwargs) for item in items]

        results = None
        if self.config.parallel and len(items) > 1000:
            results = self._execute_parallel_numba(compiled_func, items, **kwargs)

        if results is None:
            results = self._execute_sequential(compiled_func, func, items, **kwargs)

        elapsed = time.time() - start_time
        self.stats["calls"] += 1
        self.stats["total_time"] += elapsed
        self.stats["items_processed"] += len(items)

        if self.config.verbose:
            logger.debug(
                f" Numba executed {len(items)} items in {elapsed:.3f}s "
                f"({len(items) / elapsed:.1f} items/s)"
            )

        return results

    def _get_or_compile(self, func: Callable[[T], R]) -> Callable[[T], R] | None:
        """Compile or return cached Numba function; return None on fallback."""
        func_id = id(func)
        if self.config.cache and func_id in self.compiled_funcs:
            return self.compiled_funcs[func_id]

        if hasattr(func, "__self__") or (hasattr(func, "__code__") and func.__code__.co_freevars):
            logger.debug(
                "Function not compilable by Numba (method or closure), using Python execution"
            )
            return None

        try:
            if NUMBA_AVAILABLE:
                from numba.core.registry import CPUDispatcher

                if isinstance(func, CPUDispatcher):
                    compiled_func = func
                else:
                    compiled_func = njit(
                        cache=self.config.cache,
                        fastmath=self.config.fastmath,
                        error_model=self.config.error_model,
                        parallel=self.config.parallel,
                    )(func)
            else:
                compiled_func = njit(
                    cache=self.config.cache,
                    fastmath=self.config.fastmath,
                    error_model=self.config.error_model,
                    parallel=self.config.parallel,
                )(func)

            if self.config.cache:
                self.compiled_funcs[func_id] = compiled_func
            return compiled_func
        except Exception as e:
            logger.debug(f"Numba compilation failed: {e}, using Python execution")
            return None

    def _execute_sequential(
        self,
        compiled_func: Callable[[T], R],
        func: Callable[[T], R],
        items: list[T],
        **kwargs,
    ) -> list[R]:
        """Execute compiled function sequentially with safe fallback."""
        try:
            return [compiled_func(item, **kwargs) for item in items]
        except Exception as e:
            logger.debug(f"Numba execution failed: {e}, using Python execution")
            return [func(item, **kwargs) for item in items]

    def _execute_parallel_numba(
        self, compiled_func: Callable, items: list[T], **kwargs
    ) -> list[R] | None:
        """Execute with Numba parallel loops when inputs are numeric and kwargs are empty."""
        if kwargs:
            return None

        try:
            items_array = np.asarray(items)
            if items_array.ndim != 1:
                return None
            if items_array.dtype == object:
                return None
            if not (np.issubdtype(items_array.dtype, np.number) or items_array.dtype == np.bool_):
                return None
            if items_array.size == 0:
                return []

            items_array = np.ascontiguousarray(items_array)

            output_dtype = self._infer_output_dtype(compiled_func, items_array)
            if output_dtype is None:
                return None

            batch_func = self._get_or_build_batch_kernel(
                compiled_func, items_array.dtype, output_dtype
            )

            result_array = batch_func(items_array)
            return result_array.tolist()
        except Exception as e:
            logger.warning(f" Numba parallel execution failed: {e}, falling back to sequential")
            return None

    def _infer_output_dtype(
        self, compiled_func: Callable, items_array: np.ndarray
    ) -> np.dtype | None:
        """Infer output dtype for a scalar numeric function."""
        cache_key = (id(compiled_func), str(items_array.dtype))
        cached_dtype = self.output_dtype_cache.get(cache_key)
        if cached_dtype is not None:
            return cached_dtype

        try:
            sample = compiled_func(items_array[0])
            sample_array = np.asarray(sample)
            if sample_array.shape != ():
                return None
            if not (np.issubdtype(sample_array.dtype, np.number) or sample_array.dtype == np.bool_):
                return None
            output_dtype = sample_array.dtype
            self.output_dtype_cache[cache_key] = output_dtype
            return output_dtype
        except Exception:
            return None

    def _get_or_build_batch_kernel(
        self,
        compiled_func: Callable,
        input_dtype: np.dtype,
        output_dtype: np.dtype,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Compile or return cached parallel batch kernel."""
        cache_key = (id(compiled_func), str(input_dtype), str(output_dtype))
        cached = self.compiled_batch_kernels.get(cache_key)
        if cached is not None:
            return cached

        output_dtype = np.dtype(output_dtype)

        @njit(
            cache=self.config.cache,
            fastmath=self.config.fastmath,
            error_model=self.config.error_model,
            parallel=True,
        )
        def _batch_kernel(items_array: np.ndarray) -> np.ndarray:
            n = items_array.shape[0]
            results = np.empty(n, dtype=output_dtype)
            for i in prange(n):
                results[i] = compiled_func(items_array[i])
            return results

        self.compiled_batch_kernels[cache_key] = _batch_kernel
        return _batch_kernel


class VectorizedStrategy(AcceleratorStrategy[T, R]):
    """Acceleration strategy using NumPy vectorization."""

    def execute(self, func: Callable[[T], R], items: list[T], **kwargs) -> list[R]:
        """Execute using NumPy vectorization."""
        start_time = time.time()

        try:
            items_array = np.asarray(items)
            results_array = func(items_array, **kwargs)  # type: ignore
            results_array = np.asarray(results_array)
            if results_array.shape != items_array.shape:
                raise ValueError("Vectorized function returned non-elementwise output")
            results = results_array.tolist()
        except Exception as e:
            logger.warning(f" Vectorization failed: {e}, falling back to list comprehension")
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

        if config.backend == AcceleratorBackend.NUMBA:
            if NUMBA_AVAILABLE:
                return NumbaStrategy(config)
            logger.warning("Numba not available, falling back to parallel")
            config.backend = AcceleratorBackend.PARALLEL

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
        >>> print(results)
        [True, False, True]
    """

    def __init__(self, config: AcceleratorConfig | None = None, encoder: Any | None = None):
        """
        Initialize loop accelerator.

        Args:
            config: Acceleration configuration (defaults to Numba with parallel)
            encoder: Optional encoder for converting complex types to Numba-compatible types
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
            logger.info(
                f"LoopAccelerator processou {len(items)} itens em {elapsed:.3f}s "
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

        batch_results = cast(list[list[R]], self.map(cast(Any, func), cast(Any, batches), **kwargs))

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
    backend: AcceleratorBackend = AcceleratorBackend.NUMBA,
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
    accelerator = LoopAccelerator(config=config)
    return accelerator.map(func, items, **kwargs)

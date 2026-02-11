"""Concurrency manager and factory for executor creation."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .executors.dask import DaskExecutor
from .executors.joblib import JoblibExecutor
from .executors.process import ProcessExecutor
from .executors.ray import RayExecutor
from .executors.thread import ThreadExecutor
from .hardware import HardwareManager
from .protocols import BaseExecutor
from .strategies import (
    IoAsyncioStrategy,
    IoThreadingStrategy,
)
from .utils import _require_psutil, _try_import_pynvml


class ExecutorFactory:
    """Factory for creating executor instances."""

    @staticmethod
    def create(
        kind: str, max_workers: int | None = None, **backend_kwargs: Any
    ) -> BaseExecutor:
        k = kind.lower()
        if k == "thread":
            return ThreadExecutor(max_workers=max_workers)
        if k == "process":
            return ProcessExecutor(max_workers=max_workers)
        if k == "dask":
            return DaskExecutor(address=backend_kwargs.get("address"), **backend_kwargs)
        if k == "ray":
            return RayExecutor(**backend_kwargs)
        if k == "joblib":
            return JoblibExecutor(n_jobs=max_workers)
        raise ValueError(f"Executor desconhecido: {kind}")


class DurableRayTrainer:
    """Trainer providing fault-tolerance and node affinity for Ray tasks."""

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        max_retries: int = 3,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.max_retries = max_retries
        self._ray = None

    def _require_ray(self) -> Any:
        if self._ray is None:
            try:
                import ray

                self._ray = ray  # type: ignore[assignment]
            except ImportError:
                raise RuntimeError("ray não disponível")
        return self._ray

    def create_durable_trainable(self, train_fn: Callable[..., Any]) -> Any:
        """Wrap training function with fault tolerance."""
        ray = self._require_ray()

        @ray.remote(max_retries=self.max_retries)
        @functools.wraps(train_fn)
        def durable_wrapper(*args, **kwargs):
            return train_fn(*args, **kwargs)

        return durable_wrapper

    def create_node_affinity_executor(self, fn: Callable[..., Any]) -> Any:
        """Create executor with node affinity affinity."""
        ray = self._require_ray()

        @ray.remote
        @functools.wraps(fn)
        def affinity_wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return affinity_wrapper


def get_durable_trainer(
    checkpoint_dir: str | Path | None = None,
    max_retries: int = 3,
) -> DurableRayTrainer:
    """Factory for DurableRayTrainer."""
    return DurableRayTrainer(checkpoint_dir=checkpoint_dir, max_retries=max_retries)


class ConcurrencyManager:
    """Manager for concurrent execution with hardware detection and safety checks."""

    def __init__(self, memory_threshold_pct: float | None = None):
        """
        Initialize ConcurrencyManager with hardware detection and memory monitoring.

        Args:
            memory_threshold_pct: Maximum RAM usage percentage before raising error.
                                 Default: read from env PFF_MEMORY_THRESHOLD_PCT or 85.0
        """
        from ...ops.global_interrupt_manager import (
            PRIORITY_HIGH,
            get_interrupt_manager,
            should_stop,
        )

        if memory_threshold_pct is None:
            memory_threshold_pct = float(os.getenv("PFF_MEMORY_THRESHOLD_PCT", "85.0"))

        self.hardware = HardwareManager()
        self._memory_threshold_pct = memory_threshold_pct
        self._should_stop = should_stop
        get_interrupt_manager().register_callback_once(
            self._shutdown_workers,
            priority=PRIORITY_HIGH,
            label="concurrency_manager_shutdown",
        )

    def _check_memory_safety(self) -> None:
        """
        Verifies if there is sufficient RAM before starting workers.

        Raises:
            MemoryError: If RAM usage exceeds threshold, preventing OOM.
        """
        psutil = _require_psutil()
        proc = psutil.Process()
        with proc.oneshot():
            mem = psutil.virtual_memory()
            rss = proc.memory_info().rss / (1024**3)

        from ...core.logging import logger

        logger.debug(f"memory_check rss_gb={rss:.2f} committed_pct={mem.percent:.1f}")
        if mem.percent > self._memory_threshold_pct:
            available_gb = mem.available / (1024**3)
            total_gb = mem.total / (1024**3)
            raise MemoryError(
                f"  RAM usage {mem.percent:.1f}% exceeds safety threshold "
                f"({self._memory_threshold_pct}%). "
                f"Available: {available_gb:.1f} GB / {total_gb:.1f} GB total. "
                f"Recomendação: Fechar aplicações ou reduzir max_workers."
            )

        if self.hardware.gpus:
            pynvml = _try_import_pynvml()
            if pynvml is None:
                return
            gpu_alerts = []
            for gpu in self.hardware.gpus:
                try:
                    handle = self.hardware.get_handle(gpu)
                    if handle is None:
                        continue
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    usage_pct = mem_info.used / mem_info.total * 100
                    logger.debug(
                        f"GPU {gpu.name} utilizacao: {usage_pct:.1f}% memoria (utilizacao compute {util.gpu:.0f}%)"
                    )
                    if usage_pct > 92:
                        gpu_alerts.append((gpu.name, usage_pct))
                except pynvml.NVMLError:
                    continue
            if gpu_alerts:
                alerts = ", ".join(f"{name} {pct:.1f}%" for name, pct in gpu_alerts)
                logger.warning(
                    f"GPUs near memory limit: {alerts}. Consider reducing batch sizes."
                )

    def _shutdown_workers(self) -> None:
        from ...core.logging import logger

        logger.info("ConcurrencyManager: iniciando shutdown de workers")

    def shutdown(self) -> None:
        """Explicitly shut down all managed workers and executors."""
        self._shutdown_workers()

    def execute_sync(
        self,
        fn: Callable[..., Any],
        args_list: list[tuple],
        *,
        task_type: str = "auto",
        max_workers: int | None = None,
        desc: str | None = None,
        shared_data: Any = None,
        backend_kwargs: dict | None = None,
    ) -> list[Any]:
        """
        Synchronously executes a function in parallel over a list of arguments.

        Args:
            fn: The function to be parallelized.
            args_list: A list of argument tuples for the function.
            task_type: The execution strategy. Supported: 'auto', 'io_thread',
                       'process', 'joblib', 'dask', 'ray', 'cpu'.
                       'io_async' is not supported in the sync version.
            max_workers: The maximum number of worker processes or threads.
            desc: A description for the progress bar.
            shared_data: Data to be shared across workers (for Joblib/Dask).
            backend_kwargs: Backend-specific arguments.

        Returns:
            A list of results in the same order as `args_list`.
        """
        from ...core.logging import logger

        if self._should_stop():
            logger.warning("ConcurrencyManager: execution cancelled due to interrupt")
            raise KeyboardInterrupt("Concurrent execution interrupted")

        t = task_type.lower()
        backend_kwargs = backend_kwargs or {}

        if t == "auto":
            return self._auto_execute_sync(fn, args_list, max_workers, desc, shared_data)  # type: ignore[no-any-return]
        elif t in ("io_thread", "thread"):
            logger.debug(
                "Executing tasks in thread pool",
                task_count=len(args_list),
                workers=max_workers,
                strategy="io_thread",
            )
            strategy = IoThreadingStrategy(self.hardware, max_workers)
            try:
                return strategy.exec.map(fn, args_list, desc=desc)
            finally:
                if hasattr(strategy, "shutdown"):
                    strategy.shutdown()
        elif t in ("io_async", "asyncio"):
            raise ValueError(
                "task_type 'io_async' is not supported in execute_sync. Use execute instead."
            )
        elif t in ("dask", "process", "joblib", "ray", "cpu"):
            if t == "cpu":
                t = "process"
            logger.debug(
                "Executing tasks in process pool",
                task_count=len(args_list),
                workers=max_workers,
                strategy=t,
            )
            executor = None
            try:
                executor = ExecutorFactory.create(t, max_workers, **backend_kwargs)
                return executor.map(fn, args_list, desc=desc, shared_data=shared_data)
            finally:
                if executor:
                    executor.shutdown()
        elif t == "polars":
            raise ValueError(
                "task_type 'polars' is not supported in execute_sync. Use execute instead."
            )
        else:
            raise ValueError(f"Tipo de tarefa desconhecido: {task_type!r}")

    async def execute(
        self,
        fn: Callable[..., Any],
        args_list: list[tuple],
        *,
        task_type: str = "auto",
        max_workers: int | None = None,
        desc: str | None = None,
        shared_data: Any = None,
        backend_kwargs: dict | None = None,
    ) -> list[Any]:
        """
        Args:
            fn: function to be parallelized.
            args_list: list of argument tuples.
            task_type:
            - 'auto'      → heuristic (interactive convenience).
            - 'io_thread' → IoThreadingStrategy.
            - 'io_async'  → IoAsyncioStrategy.
            - 'process'   → ProcessExecutor.
            - 'joblib'    → JoblibExecutor.
            - 'dask'      → DaskExecutor.
            max_workers: maximum number of workers.
            desc: text for the progress bar.
            shared_data: data to be shared (Joblib/Dask).
        Returns:
            list of results, in the order of args_list.
        """
        from ...core.logging import logger

        if self._should_stop():
            logger.warning("ConcurrencyManager: execution cancelled due to interrupt")
            raise KeyboardInterrupt("Concurrent execution interrupted")

        self._check_memory_safety()

        t = task_type.lower()
        backend_kwargs = backend_kwargs or {}

        if t == "auto":
            return await self._auto_execute(fn, args_list, max_workers, desc, shared_data)  # type: ignore[no-any-return]
        elif t in ("io_thread", "thread"):
            logger.debug(
                "Executing tasks in thread pool (async wrapper)",
                task_count=len(args_list),
                workers=max_workers,
            )
            strategy = IoThreadingStrategy(self.hardware, max_workers)
            try:
                return await strategy.execute(fn, args_list, desc=desc)  # type: ignore[no-any-return]
            finally:
                if hasattr(strategy, "shutdown"):
                    strategy.shutdown()
        elif t in ("io_async", "asyncio"):
            logger.debug(
                "Executing tasks in asyncio loop",
                task_count=len(args_list),
                workers=max_workers,
            )
            strategy = IoAsyncioStrategy(self.hardware, max_workers)  # type: ignore[assignment]
            try:
                return await strategy.execute(fn, args_list, desc=desc)  # type: ignore[no-any-return]
            finally:
                if hasattr(strategy, "shutdown"):
                    strategy.shutdown()
        elif t in ("dask", "process", "joblib", "ray", "cpu"):
            if t == "cpu":
                t = "process"
            logger.debug(
                "Executing tasks in process pool",
                task_count=len(args_list),
                workers=max_workers,
                strategy=t,
            )
            executor = None

            try:
                executor = ExecutorFactory.create(t, max_workers, **backend_kwargs)
                return executor.map(fn, args_list, desc=desc, shared_data=shared_data)
            finally:
                if executor:
                    executor.shutdown()
        elif t == "polars":
            from .strategies import GpuCudfStrategy

            strategy = GpuCudfStrategy(self.hardware)  # type: ignore[assignment]
            return await strategy.execute(fn, args_list, **backend_kwargs)  # type: ignore[no-any-return]
        else:
            raise ValueError(f"Tipo de tarefa desconhecido: {task_type!r}")

    async def _auto_execute(self, fn, args_list, max_workers, desc, shared_data):
        if len(args_list) < 5:
            return await self.execute(
                fn, args_list, task_type="asyncio", max_workers=max_workers, desc=desc
            )
        return await self.execute(
            fn,
            args_list,
            task_type="process",
            max_workers=max_workers,
            desc=desc,
            shared_data=shared_data,
        )

    def _auto_execute_sync(self, fn, args_list, max_workers, desc, shared_data):
        if len(args_list) < 5:
            return self.execute_sync(
                fn, args_list, task_type="thread", max_workers=max_workers, desc=desc
            )
        return self.execute_sync(
            fn,
            args_list,
            task_type="process",
            max_workers=max_workers,
            desc=desc,
            shared_data=shared_data,
        )

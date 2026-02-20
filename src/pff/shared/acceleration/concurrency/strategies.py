"""Execution strategies for different workload types."""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from .executors.process import ProcessExecutor
from .executors.thread import ThreadExecutor
from .hardware import HardwareManager
from .protocols import Args
from .utils import _require_dask, _require_polars, progress_bar


class ExecutionStrategy(ABC):
    """Abstract base class for execution strategies."""

    @abstractmethod
    async def execute(
        self, fn: Callable[..., Any], args_list: Sequence[Args], **kwargs: Any
    ) -> Sequence[Any]:
        """Execute execute.

        Args:
            fn: Input value used by this callable.
            args_list: Input value used by this callable.
            **kwargs: Input value used by this callable.
        """
        ...


class CpuMultiprocessingStrategy(ExecutionStrategy):
    """CPU-bound workload strategy using process-based parallelism."""

    def __init__(self, hardware: HardwareManager, max_workers: int | None = None):
        """Execute init.



        Args:

            hardware: Input value used by this callable.

            max_workers: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        workers = max_workers or max(1, hardware.physical_cores - 1)
        self.exec = ProcessExecutor(max_workers=workers)

    async def execute(self, fn, args_list, **kwargs):
        """Execute execute.



        Args:

            fn: Input value used by this callable.

            args_list: Input value used by this callable.

            **kwargs: Additional keyword arguments.



        Returns:

            Return value produced by the callable.

        """

        return self.exec.map(fn, args_list, desc=kwargs.get("desc"))

    def shutdown(self):
        """Execute shutdown."""

        self.exec.shutdown()


class IoThreadingStrategy(ExecutionStrategy):
    """I/O-bound workload strategy using thread-based parallelism."""

    def __init__(self, hardware: HardwareManager, max_workers: int | None = None):
        """Execute init.



        Args:

            hardware: Input value used by this callable.

            max_workers: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        workers = max_workers or hardware.logical_cores
        self.exec = ThreadExecutor(max_workers=workers)

    async def execute(self, fn, args_list, **kwargs):
        """Execute execute.



        Args:

            fn: Input value used by this callable.

            args_list: Input value used by this callable.

            **kwargs: Additional keyword arguments.



        Returns:

            Return value produced by the callable.

        """

        return self.exec.map(fn, args_list, desc=kwargs.get("desc"))

    def shutdown(self):
        """Execute shutdown."""

        self.exec.shutdown()


class IoAsyncioStrategy(ExecutionStrategy):
    """I/O-bound workload strategy using asyncio."""

    def __init__(self, hardware: HardwareManager, max_workers: int | None = None):
        """Execute init.



        Args:

            hardware: Input value used by this callable.

            max_workers: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.concurrency = max_workers or hardware.logical_cores

    async def execute(self, fn, args_list, **kwargs):
        """Execute execute.



        Args:

            fn: Input value used by this callable.

            args_list: Input value used by this callable.

            **kwargs: Additional keyword arguments.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        desc = kwargs.get("desc")
        sem = asyncio.Semaphore(self.concurrency)
        total_tasks = len(args_list)
        if total_tasks < 100:
            return await self._run_small_batch(fn, args_list, sem, desc=desc)
        return await self._run_bounded_queue(fn, args_list, sem)

    async def _run_small_batch(
        self,
        fn: Callable[..., Any],
        args_list: Sequence[Args],
        sem: asyncio.Semaphore,
        *,
        desc: str | None,
    ) -> list[Any]:
        """Execute run small batch.



        Args:

            fn: Input value used by this callable.

            args_list: Input value used by this callable.

            sem: Input value used by this callable.

            desc: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        tasks = [asyncio.create_task(self._run_one(fn, args, sem)) for args in args_list]
        results: list[Any] = []
        for fut in progress_bar(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
            results.append(await fut)
        return results

    async def _run_bounded_queue(
        self,
        fn: Callable[..., Any],
        args_list: Sequence[Args],
        sem: asyncio.Semaphore,
    ) -> list[Any]:
        """Execute run bounded queue.



        Args:

            fn: Input value used by this callable.

            args_list: Input value used by this callable.

            sem: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        total_tasks = len(args_list)
        queue_size = self.concurrency * 2
        queue: asyncio.Queue[tuple[int, Args]] = asyncio.Queue(maxsize=queue_size)
        results: list[Any] = [None] * total_tasks
        tasks_completed = 0

        async def producer() -> None:
            """Execute producer.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            for idx, args in enumerate(args_list):
                await queue.put((idx, args))

        async def worker() -> None:
            """Execute worker.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            nonlocal tasks_completed
            while True:
                try:
                    idx, args = await asyncio.wait_for(queue.get(), timeout=0.1)
                    result = await self._run_one(fn, args, sem)
                    results[idx] = result
                    tasks_completed += 1
                    queue.task_done()
                    del args
                    del result
                except asyncio.TimeoutError:
                    if tasks_completed >= total_tasks:
                        break

        producer_task = asyncio.create_task(producer())
        worker_tasks = [asyncio.create_task(worker()) for _ in range(self.concurrency)]
        await producer_task
        await asyncio.gather(*worker_tasks)
        return results

    @staticmethod
    async def _run_one(fn: Callable[..., Any], args: Args, sem: asyncio.Semaphore) -> Any:
        """Execute run one.



        Args:

            fn: Input value used by this callable.

            args: Input value used by this callable.

            sem: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        async with sem:
            if inspect.iscoroutinefunction(fn):
                return await fn(*args)
            return fn(*args)

    def shutdown(self) -> None:
        """Execute shutdown."""

        pass


class GpuCudfStrategy(ExecutionStrategy):
    """GPU-accelerated strategy using cuDF/Polars GPU engine."""

    def __init__(self, hardware: HardwareManager):
        """Execute init.



        Args:

            hardware: Input value used by this callable.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if not hardware.gpus:
            raise RuntimeError("Nenhuma GPU NVIDIA detectada.")
        gpu = hardware.gpus[0]
        if gpu.compute_capability[0] < 7:
            raise RuntimeError(f"GPU {gpu.name} não suportada (>=7.0).")

    async def execute(self, fn, args_list, **kwargs):
        """Execute execute.



        Args:

            fn: Input value used by this callable.

            args_list: Input value used by this callable.

            **kwargs: Additional keyword arguments.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        pl = _require_polars()
        results = []
        for args in args_list:
            lazy = fn(*args)
            if not isinstance(lazy, pl.LazyFrame):
                raise RuntimeError("Esperado pl.LazyFrame para GpuCudfStrategy.")
            results.append(lazy.collect(engine="gpu"))
        return results


class DaskRayCompat:
    """Wrapper for compatibility between Ray and Dask APIs"""

    def __init__(self):
        """Execute init.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        DaskClient, _ = _require_dask()
        self.client = DaskClient(processes=True)

    def put(self, data):
        """Execute put.



        Args:

            data: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        return self.client.scatter(data, broadcast=True)

    def get(self, future):
        """Execute get.



        Args:

            future: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if hasattr(future, "result"):
            return future.result()
        return future

    def shutdown(self):
        """Execute shutdown."""

        self.client.close()

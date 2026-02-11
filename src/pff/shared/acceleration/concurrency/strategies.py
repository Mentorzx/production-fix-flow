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
    ) -> Sequence[Any]: ...


class CpuMultiprocessingStrategy(ExecutionStrategy):
    """CPU-bound workload strategy using process-based parallelism."""

    def __init__(self, hardware: HardwareManager, max_workers: int | None = None):
        workers = max_workers or max(1, hardware.physical_cores - 1)
        self.exec = ProcessExecutor(max_workers=workers)

    async def execute(self, fn, args_list, **kwargs):
        return self.exec.map(fn, args_list, desc=kwargs.get("desc"))

    def shutdown(self):
        self.exec.shutdown()


class IoThreadingStrategy(ExecutionStrategy):
    """I/O-bound workload strategy using thread-based parallelism."""

    def __init__(self, hardware: HardwareManager, max_workers: int | None = None):
        workers = max_workers or hardware.logical_cores
        self.exec = ThreadExecutor(max_workers=workers)

    async def execute(self, fn, args_list, **kwargs):
        return self.exec.map(fn, args_list, desc=kwargs.get("desc"))

    def shutdown(self):
        self.exec.shutdown()


class IoAsyncioStrategy(ExecutionStrategy):
    """I/O-bound workload strategy using asyncio."""

    def __init__(self, hardware: HardwareManager, max_workers: int | None = None):
        self.concurrency = max_workers or hardware.logical_cores

    async def execute(self, fn, args_list, **kwargs):
        desc = kwargs.get("desc")

        async def runner():
            sem = asyncio.Semaphore(self.concurrency)

            async def run_one(args):
                async with sem:
                    if inspect.iscoroutinefunction(fn):
                        return await fn(*args)
                    return fn(*args)

            total_tasks = len(args_list)

            if total_tasks < 100:
                tasks = [asyncio.create_task(run_one(args)) for args in args_list]
                results = []
                for fut in progress_bar(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
                    results.append(await fut)
                return results

            queue_size = self.concurrency * 2
            queue = asyncio.Queue(maxsize=queue_size)
            results = [None] * total_tasks
            tasks_completed = 0

            async def producer():
                for idx, args in enumerate(args_list):
                    await queue.put((idx, args))

            async def worker():
                nonlocal tasks_completed
                while True:
                    try:
                        idx, args = await asyncio.wait_for(queue.get(), timeout=0.1)
                        result = await run_one(args)
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

        return await runner()

    def shutdown(self) -> None:
        pass


class GpuCudfStrategy(ExecutionStrategy):
    """GPU-accelerated strategy using cuDF/Polars GPU engine."""

    def __init__(self, hardware: HardwareManager):
        if not hardware.gpus:
            raise RuntimeError("Nenhuma GPU NVIDIA detectada.")
        gpu = hardware.gpus[0]
        if gpu.compute_capability[0] < 7:
            raise RuntimeError(f"GPU {gpu.name} não suportada (>=7.0).")

    async def execute(self, fn, args_list, **kwargs):
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
        DaskClient, _ = _require_dask()
        self.client = DaskClient(processes=True)

    def put(self, data):
        return self.client.scatter(data, broadcast=True)

    def get(self, future):
        if hasattr(future, "result"):
            return future.result()
        return future

    def shutdown(self):
        self.client.close()

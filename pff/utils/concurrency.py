from __future__ import annotations

import asyncio
import functools
import inspect
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Sized
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

import duckdb
import joblib
import numpy as np
import polars as pl
import psutil
import pynvml
import ray
from dask.distributed import Client as DaskClient
from dask.distributed import as_completed as dask_as_completed
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from pff.utils import logger

Args = tuple[Any, ...]
_R = TypeVar("_R")


def progress_bar(
    iterable: Iterable[Any],
    *,
    total: int | None = None,
    desc: str | None = None,
    enabled: bool = True,
) -> Iterator[Any]:
    """
    Iterates over an iterable while displaying a progress bar in the terminal.
    This function provides a visual progress indicator for long-running iterations.
    If enabled and the terminal supports it, it uses the Rich library for a modern progress bar.
    Otherwise, it falls back to a simple text-based progress bar or spinner.
    Progress is displayed on stderr and updates periodically or at the end of the iteration.
    Args:
        iterable (Iterable[Any]): The iterable to process.
        total (int | None, optional): The total number of items. If not provided, tries to infer using len().
        desc (str | None, optional): Description to display alongside the progress bar.
        enabled (bool, optional): If False, disables the progress bar and yields items directly. Defaults to True.
    Yields:
        Any: Items from the input iterable, one by one.
    Notes:
        - If the Rich library is available and the terminal supports it, a Rich progress bar is shown.
        - If not, a fallback text-based progress bar or spinner is used.
        - Progress is only shown if `enabled` is True.
        - Handles both sized and unsized iterables.
        - Displays elapsed time and estimated time remaining (ETA) when possible.
    Examples:
        >>> for item in progress_bar(range(100), desc="Processing"):
        ...     process(item)
    """
    if not enabled:
        yield from iterable
        return
    if total is None and isinstance(iterable, Sized):
        try:
            total = len(iterable)
        except Exception:
            total = None
    if Progress and sys.stderr.isatty():
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
        try:
            with Progress(
                *columns, transient=False, refresh_per_second=4
            ) as progress:  # ✨ refresh mais frequente
                task = progress.add_task(desc or "Processando...", total=total)
                for item in iterable:
                    yield item
                    progress.update(task, advance=1)
                progress.update(
                    task, completed=total if total else progress.tasks[task].completed
                )
            sys.stderr.write("\n")
            sys.stderr.flush()
            return
        except Exception as e:
            logger.debug(f"Falha no Rich progress: {e}, usando fallback")
            pass
    try:
        terminal_width = shutil.get_terminal_size().columns
    except Exception:
        terminal_width = 80
    start_time = time.time()
    last_update = start_time
    items_processed = 0
    for idx, item in enumerate(iterable, start=1):
        yield item
        items_processed = idx
        current_time = time.time()
        if current_time - last_update >= 0.5 or (total and idx == total):
            last_update = current_time
            elapsed = current_time - start_time
            if total and total > 0:
                percentage = (idx / total) * 100
                if idx > 1 and elapsed > 1:  # Precisa de pelo menos 2 items e 1s
                    rate = idx / elapsed
                    if rate > 0:
                        eta_seconds = (total - idx) / rate
                        eta_str = f" ETA: {_format_time(eta_seconds)}"
                    else:
                        eta_str = " ETA: calculando..."
                else:
                    eta_str = " ETA: calculando..."

                bar_width = min(30, terminal_width - 60)  # ✨ Mais espaço para texto
                filled = int((percentage / 100) * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)
                status = (
                    f"\r{desc or 'Progresso'}: {percentage:5.1f}% "
                    f"|{bar}| {idx}/{total} "
                    f"[{_format_time(elapsed)}{eta_str}]"
                )
            else:
                spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                spinner = spinner_chars[idx % len(spinner_chars)]
                status = (
                    f"\r{desc or 'Processando'} {spinner} "
                    f"{idx} items [{_format_time(elapsed)}]"
                )
            clear_line = "\r" + " " * (terminal_width - 1) + "\r"
            sys.stderr.write(clear_line + status)
            sys.stderr.flush()
    if total:
        elapsed = time.time() - start_time
        final_msg = (
            f"\r{desc or 'Concluído'}: 100.0% "
            f"|{'█' * 30}| {total}/{total} "
            f"[{_format_time(elapsed)} total]"
        )
        clear_line = "\r" + " " * (terminal_width - 1) + "\r"
        sys.stderr.write(clear_line + final_msg + "\n")
    else:
        elapsed = time.time() - start_time
        final_msg = (
            f"\r{desc or 'Concluído'}: {items_processed} items "
            f"em {_format_time(elapsed)}"
        )
        clear_line = "\r" + " " * (terminal_width - 1) + "\r"
        sys.stderr.write(clear_line + final_msg + "\n")
    sys.stderr.flush()


def _format_time(seconds: float) -> str:
    """
    Formats a time duration given in seconds into a human-readable string.
    If the duration is negative, returns "--:--".
    If the duration is one hour or more, returns a string in the format "HH:MM:SS".
    If the duration is less than one hour, returns a string in the format "MM:SS".
    Args:
        seconds (float): The time duration in seconds.
    Returns:
        str: The formatted time string.
    """
    if seconds < 0:
        return "--:--"
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


class BaseExecutor(ABC):
    @abstractmethod
    def map(
        self,
        fn: Callable[..., Any],
        args_list: Iterable[Args],
        *,
        desc: str | None = None,
        **kwargs: Any,
    ) -> list[Any]: ...

    @abstractmethod
    def submit(self, fn: Callable[..., Any], *args: Any) -> Any: ...

    @abstractmethod
    def shutdown(self) -> None: ...


class ThreadExecutor(BaseExecutor):
    def __init__(self, max_workers: int | None = None):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def map(
        self, fn: Callable[..., Any], args_list: Iterable[Args], *, desc: str | None = None, **kwargs: Any
    ) -> list[Any]:
        # Convert to list for len() if needed
        args_list_materialized = list(args_list) if not isinstance(args_list, (list, tuple)) else args_list
        futures = [self._pool.submit(fn, *args) for args in args_list_materialized]
        results: list[Any] = []
        for fut in progress_bar(futures, total=len(futures), desc=desc):
            results.append(fut.result())
        return results

    def submit(self, fn, *args):
        return self._pool.submit(fn, *args)

    def shutdown(self):
        self._pool.shutdown(wait=True)


class ProcessExecutor(BaseExecutor):
    def __init__(self, max_workers: int | None = None):
        ctx = mp.get_context("spawn")
        self._pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)

    def map(
        self, fn: Callable[..., Any], args_list: Iterable[Args], *, desc: str | None = None, **kwargs: Any
    ) -> list[Any]:
        """
        Execute function over args_list with bounded memory using lazy task submission.

        Prevents OOM by limiting concurrent futures to avoid memory explosion
        with large task lists (e.g., 100K+ items).
        """
        # Convert to list if needed for indexing and len()
        if not isinstance(args_list, (list, tuple)):
            args_list = list(args_list)

        # 🚀 ADAPTIVE: Use runtime resource detection for max_pending
        # Get current limits from adaptive resource manager
        try:
            from pff.utils.resource_manager import get_resource_manager
            resource_manager = get_resource_manager()

            # Quick calculation for adaptive limits
            max_workers = getattr(self._pool, '_max_workers', None) or os.cpu_count() or 4
            limits = resource_manager.calculate_limits(
                task_count=len(args_list),
                estimated_task_size=5000,  # Assume 5 KB per task
                max_workers=max_workers,
            )
            max_pending = limits.max_pending_futures

            from loguru import logger
            logger.debug(
                f"🚀 Adaptive ProcessExecutor: {max_workers} workers, "
                f"{max_pending} max pending (90% memory safe)"
            )
        except Exception:
            # Fallback to conservative default if adaptive fails
            max_workers = getattr(self._pool, '_max_workers', None) or os.cpu_count() or 4
            max_pending = max(100, max_workers * 10)

        results: list[Any] = [None] * len(args_list)
        pending: dict[Any, int] = {}
        idx = 0
        completed = 0
        total = len(args_list)

        pbar = progress_bar(range(total), total=total, desc=desc, enabled=bool(desc))
        pbar_iter = iter(pbar)

        while completed < total or pending:
            while len(pending) < max_pending and idx < total:
                fut = self._pool.submit(fn, *args_list[idx])
                pending[fut] = idx
                idx += 1

            if not pending:
                break

            done_futs = [f for f in pending.keys() if f.done()]

            for fut in done_futs:
                original_idx = pending.pop(fut)
                results[original_idx] = fut.result()
                completed += 1
                try:
                    next(pbar_iter)
                except StopIteration:
                    pass

            if not done_futs and pending:
                time.sleep(0.001)

        return results

    def submit(self, fn, *args):
        return self._pool.submit(fn, *args)

    def shutdown(self):
        self._pool.shutdown(wait=True)


class DaskExecutor(BaseExecutor):
    def __init__(self, address: str | None = None, **client_kwargs: Any):
        self._client = DaskClient(address=address, **client_kwargs)

    def map(
        self,
        fn: Callable[..., Any],
        args_list: Iterable[tuple],
        *,
        desc: str | None = None,
        shared_data: Any = None,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Applies a function to a list of argument tuples in parallel, optionally sharing data across tasks.

        **FIX (v10.5.0):** Uses lazy submission with bounded queue to prevent OOM with large task lists.

        Args:
            fn (Callable[..., Any]): The function to apply to each set of arguments.
            args_list (Iterable[tuple]): An iterable of argument tuples to pass to the function.
            desc (str | None, optional): Description for the progress bar. Defaults to None.
            shared_data (Any, optional): Data to be shared across all tasks. If provided, it is scattered and passed as the first argument to `fn`. Defaults to None.
            **kwargs (Any): Additional keyword arguments.
        Returns:
            list[Any]: A list of results from applying `fn` to each set of arguments.
        """
        # Convert to list for len() and indexing
        if not isinstance(args_list, (list, tuple)):
            args_list = list(args_list)

        total = len(args_list)

        # Scatter shared data once if provided
        future_shared_data = None
        if shared_data is not None:
            future_shared_data = self._client.scatter(shared_data, broadcast=True)

        # 🔧 FIX: Use lazy submission with bounded queue (prevents 15K+ futures at once)
        max_pending = min(1000, total)  # Max 1000 pending futures
        results: list[Any] = [None] * total
        pending: dict[Any, int] = {}
        idx = 0
        completed = 0

        pbar = progress_bar(range(total), total=total, desc=desc, enabled=bool(desc))
        pbar_iter = iter(pbar)

        while completed < total or pending:
            # Submit new tasks while under limit
            while len(pending) < max_pending and idx < total:
                if future_shared_data is not None:
                    fut = self._client.submit(fn, future_shared_data, *args_list[idx])
                else:
                    fut = self._client.submit(fn, *args_list[idx])
                pending[fut] = idx
                idx += 1

            if not pending:
                break

            # Wait for at least one future to complete
            done_futs = [f for f in pending.keys() if f.done()]

            for fut in done_futs:
                original_idx = pending.pop(fut)
                results[original_idx] = fut.result()
                completed += 1
                try:
                    next(pbar_iter)
                except StopIteration:
                    pass

            # If no futures done yet, sleep briefly
            if not done_futs and pending:
                import time
                time.sleep(0.01)

        return results

    def submit(self, fn, *args):
        return self._client.submit(fn, *args)

    def shutdown(self):
        self._client.close()


class RayExecutor(BaseExecutor):
    def __init__(self, **init_kwargs: Any):
        if sys.platform == "win32":
            logger.warning(
                "Ray no Windows é instável; usando DaskExecutor como fallback"
            )
            # Instead of ProcessExecutor, use DaskExecutor
            self._exec = DaskExecutor(**init_kwargs)
        else:
            if not ray.is_initialized():
                ray.init(**init_kwargs)
            self._exec = None  # signals use of ray

    def map(
        self,
        fn: Callable[..., Any],
        args_list: Iterable[Args],
        *,
        desc: str | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Execute tasks using Ray with adaptive batching for massive parallelism.

        For 100K+ tasks, uses automatic batching to reduce Ray overhead while
        maintaining SOTA performance.
        """
        if self._exec:
            return self._exec.map(fn, args_list, desc=desc)

        args_list = list(args_list)
        total_tasks = len(args_list)

        if total_tasks > 50000:
            batch_size = max(100, total_tasks // 1000)
            return self._map_batched(fn, args_list, batch_size, desc)

        remote_fn = ray.remote(fn)
        max_inflight = min(10000, total_tasks)

        results = [None] * total_tasks
        pending = {}
        idx = 0

        pbar = progress_bar(range(total_tasks), total=total_tasks, desc=desc, enabled=bool(desc))
        pbar_iter = iter(pbar)

        while idx < total_tasks or pending:
            while len(pending) < max_inflight and idx < total_tasks:
                ref = remote_fn.remote(*args_list[idx])
                pending[ref] = idx
                idx += 1

            if not pending:
                break

            ready, _ = ray.wait(list(pending.keys()), num_returns=min(100, len(pending)), timeout=0.01)

            for ref in ready:
                original_idx = pending.pop(ref)
                results[original_idx] = ray.get(ref)
                try:
                    next(pbar_iter)
                except StopIteration:
                    pass

        return results

    def _map_batched(
        self, fn: Callable, args_list: list, batch_size: int, desc: str | None
    ) -> list[Any]:
        """Execute in batches to reduce Ray task overhead for 100K+ tasks."""

        @ray.remote
        def batch_worker(batch_args):
            return [fn(*args) for args in batch_args]

        batches = [
            args_list[i:i + batch_size] for i in range(0, len(args_list), batch_size)
        ]

        batch_refs = [batch_worker.remote(batch) for batch in batches]
        batch_results = []

        for ref in progress_bar(batch_refs, desc=desc):
            batch_results.extend(ray.get(ref))

        return batch_results

    def submit(self, fn, *args):
        if self._exec:
            return self._exec.submit(fn, *args)
        remote_fn = ray.remote(fn)
        return remote_fn.remote(*args)

    def shutdown(self):
        pass


class JoblibExecutor(BaseExecutor):
    """
    Executor baseado em Joblib, usando memmapping para grandes numpy.ndarray
    """

    def __init__(self, n_jobs: int | None = None, mmap_threshold: int = 1 << 26):
        # mmap_threshold em bytes; default ~64MB
        self.n_jobs = n_jobs or joblib.cpu_count()
        self.mmap_thresh = mmap_threshold

    def map(
        self,
        fn: Callable[..., Any],
        args_list: Iterable[tuple],
        *,
        desc: str | None = None,
        shared_data: np.ndarray | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        mmap_path = None
        if (
            isinstance(shared_data, np.ndarray)
            and shared_data.nbytes >= self.mmap_thresh
        ):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mmap")
            tmp.close()
            joblib.dump(shared_data, tmp.name, compress=False)
            shared_mm = np.load(tmp.name, mmap_mode="r")
            mmap_path = tmp.name
        else:
            shared_mm = shared_data

        target_fn = functools.partial(fn, shared_mm) if shared_mm is not None else fn

        def _wrapper(args):
            return target_fn(*args)

        results = list(
            joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(_wrapper)(args)
                for args in progress_bar(args_list, desc=desc)
            )
        )

        if mmap_path:
            try:
                os.remove(mmap_path)
            except OSError:
                pass

        return results

    def submit(self, fn: Callable[..., Any], *args: Any) -> Any:
        """
        Note: Joblib does not have asynchronous submit.
        This method executes fn(*args) synchronously.
        For asynchronous behavior, use DaskExecutor or ThreadExecutor.
        """
        raise NotImplementedError(
            "JoblibExecutor does not support asynchronous 'submit'."
        )

    def shutdown(self):
        pass


class ExecutorFactory:
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


@dataclass
class GPUInfo:
    id: int
    name: str
    memory_total: int
    compute_capability: tuple[int, int]
    uuid: str


class HardwareManager:
    def __init__(self):
        self.physical_cores = psutil.cpu_count(logical=False) or 1
        self.logical_cores = psutil.cpu_count(logical=True) or 1
        try:
            pynvml.nvmlInit()
            cnt = pynvml.nvmlDeviceGetCount()
            self.gpus: list[GPUInfo] = []
            for i in range(cnt):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                nm = pynvml.nvmlDeviceGetName(h)
                name = nm.decode() if isinstance(nm, (bytes, bytearray)) else nm
                mem = int(pynvml.nvmlDeviceGetMemoryInfo(h).total)
                cc = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                uid = pynvml.nvmlDeviceGetUUID(h)
                uuid = uid.decode() if isinstance(uid, (bytes, bytearray)) else uid
                self.gpus.append(GPUInfo(i, name, mem, cc, uuid))
        except pynvml.NVMLError:
            self.gpus = []

    def shutdown(self):
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


class ExecutionStrategy(ABC):
    @abstractmethod
    async def execute(
        self, fn: Callable[..., Any], args_list: Sequence[Args], **kwargs: Any
    ) -> Sequence[Any]: ...


class CpuMultiprocessingStrategy(ExecutionStrategy):
    def __init__(self, hardware: HardwareManager, max_workers: int | None = None):
        workers = max_workers or max(1, hardware.physical_cores - 1)
        self.exec = ProcessExecutor(max_workers=workers)

    async def execute(self, fn, args_list, **kwargs):
        return self.exec.map(fn, args_list, desc=kwargs.get("desc"))

    def shutdown(self):
        self.exec.shutdown()


class IoThreadingStrategy(ExecutionStrategy):
    def __init__(self, hardware: HardwareManager, max_workers: int | None = None):
        workers = max_workers or hardware.logical_cores
        self.exec = ThreadExecutor(max_workers=workers)

    async def execute(self, fn, args_list, **kwargs):
        return self.exec.map(fn, args_list, desc=kwargs.get("desc"))

    def shutdown(self):
        self.exec.shutdown()


class IoAsyncioStrategy(ExecutionStrategy):
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

            # Lazy task creation with bounded queue (prevents OOM)
            total_tasks = len(args_list)

            # For small task lists (<100), use original simpler method
            if total_tasks < 100:
                tasks = [asyncio.create_task(run_one(args)) for args in args_list]
                results = []
                for fut in progress_bar(
                    asyncio.as_completed(tasks), total=len(tasks), desc=desc
                ):
                    results.append(await fut)
                return results

            # For large task lists (>=100), use bounded queue
            # Queue size = 2× concurrency (backpressure to prevent OOM)
            queue_size = self.concurrency * 2
            queue = asyncio.Queue(maxsize=queue_size)
            results = [None] * total_tasks  # Pre-allocated to maintain order
            tasks_completed = 0

            async def producer():
                """Enqueues tasks gradually (lazy)."""
                for idx, args in enumerate(args_list):
                    await queue.put((idx, args))

            async def worker():
                """Processes tasks from queue."""
                nonlocal tasks_completed
                while True:
                    try:
                        idx, args = await asyncio.wait_for(queue.get(), timeout=0.1)
                        result = await run_one(args)
                        results[idx] = result
                        tasks_completed += 1
                        queue.task_done()
                    except asyncio.TimeoutError:
                        # Queue empty and producer finished
                        if tasks_completed >= total_tasks:
                            break

            # Start producer and workers
            producer_task = asyncio.create_task(producer())
            worker_tasks = [asyncio.create_task(worker()) for _ in range(self.concurrency)]

            # Wait for completion
            await producer_task
            await asyncio.gather(*worker_tasks)

            return results

        return await runner()

    def shutdown(self) -> None:
        pass


class GpuCudfStrategy(ExecutionStrategy):
    def __init__(self, hardware: HardwareManager):
        if not hardware.gpus:
            raise RuntimeError("Nenhuma GPU NVIDIA detectada.")
        gpu = hardware.gpus[0]
        if gpu.compute_capability[0] < 7:
            raise RuntimeError(f"GPU {gpu.name} não suportada (>=7.0).")

    async def execute(self, fn, args_list, **kwargs):
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
        from dask.distributed import Client

        self.client = Client(processes=True)  # Uses processes like Ray

    def put(self, data):
        """Equivalent to ray.put() - puts data on the Dask cluster"""
        return self.client.scatter(data, broadcast=True)

    def get(self, future):
        """Equivalent to ray.get() - retrieves data from the Dask cluster"""
        if hasattr(future, "result"):
            return future.result()
        return future

    def shutdown(self):
        """Closes the Dask client"""
        self.client.close()


class ConcurrencyManager:
    def __init__(self, memory_threshold_pct: float = 85.0):
        """
        Initialize ConcurrencyManager with hardware detection and memory monitoring.

        Args:
            memory_threshold_pct: Maximum RAM usage percentage before raising error.
                                 Default: 85% (safe for most systems)
        """
        self.hardware = HardwareManager()
        self._memory_threshold_pct = memory_threshold_pct

    def _check_memory_safety(self) -> None:
        """
        Verifies if there is sufficient RAM before starting workers.

        Raises:
            MemoryError: If RAM usage exceeds threshold, preventing OOM.
        """
        mem = psutil.virtual_memory()
        if mem.percent > self._memory_threshold_pct:
            available_gb = mem.available / (1024**3)
            total_gb = mem.total / (1024**3)
            raise MemoryError(
                f"⚠️  RAM usage {mem.percent:.1f}% exceeds safety threshold "
                f"({self._memory_threshold_pct}%). "
                f"Available: {available_gb:.1f} GB / {total_gb:.1f} GB total. "
                f"Recomendação: Fechar aplicações ou reduzir max_workers."
            )

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
        t = task_type.lower()
        backend_kwargs = backend_kwargs or {}

        if t == "auto":
            return self._auto_execute_sync(
                fn, args_list, max_workers, desc, shared_data
            )
        elif t in ("io_thread", "thread"):
            strategy = IoThreadingStrategy(self.hardware, max_workers)
            try:
                # The execute method is async, but its implementation is sync-compatible
                # We can call it and get the coroutine, but since we are not in an
                # async context, we can't await it. We'll call the executor's map directly.
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
        # Check memory safety before starting workers
        self._check_memory_safety()

        t = task_type.lower()
        backend_kwargs = backend_kwargs or {}

        if t == "auto":
            return await self._auto_execute(
                fn, args_list, max_workers, desc, shared_data
            )
        elif t in ("io_thread", "thread"):
            strategy = IoThreadingStrategy(self.hardware, max_workers)
            try:
                return await strategy.execute(fn, args_list, desc=desc)
            finally:
                if hasattr(strategy, "shutdown"):
                    strategy.shutdown()
        elif t in ("io_async", "asyncio"):
            strategy = IoAsyncioStrategy(self.hardware, max_workers)
            try:
                return await strategy.execute(fn, args_list, desc=desc)
            finally:
                if hasattr(strategy, "shutdown"):
                    strategy.shutdown()
        elif t in ("dask", "process", "joblib", "ray", "cpu"):
            if t == "cpu":
                t = "process"
            executor = None

            try:
                executor = ExecutorFactory.create(t, max_workers, **backend_kwargs)
                return executor.map(fn, args_list, desc=desc, shared_data=shared_data)
            finally:
                if executor:
                    executor.shutdown()
        elif t == "polars":
            try:
                strategy = GpuCudfStrategy(self.hardware)
                return await strategy.execute(fn, args_list, **backend_kwargs)
            except RuntimeError as e:
                logger.warning(
                    f"Falha ao usar GpuCudfStrategy ({e}), usando fallback para 'process'."
                )
                executor = None
                try:
                    executor = ExecutorFactory.create("process", max_workers)
                    return executor.map(
                        fn, args_list, desc=desc, shared_data=shared_data
                    )
                finally:
                    if executor:
                        executor.shutdown()
        else:
            raise ValueError(f"Tipo de tarefa desconhecido: {task_type!r}")

    async def _auto_execute(
        self,
        fn: Callable[..., Any],
        args_list: list[tuple],
        max_workers: int | None,
        desc: str | None,
        shared_data: Any,
    ) -> list[Any]:
        """
        Automatically selects and executes the most appropriate execution strategy based on function type and data characteristics.
        This method analyzes the input function and arguments to determine the optimal execution strategy:
        - For coroutine functions: Uses IoAsyncioStrategy for async execution
        - For large numpy arrays (>64 MiB): Uses JoblibExecutor for memory-efficient processing
        - For large serialized objects (>64 MiB): Uses DaskExecutor for distributed processing
        - For general lightweight cases: Uses ProcessExecutor for parallel processing
        Args:
            fn (Callable[..., Any]): The function to execute across multiple argument sets
            args_list (list[tuple]): List of argument tuples to pass to the function
            max_workers (int | None): Maximum number of workers for execution (if applicable)
            desc (str | None): Description for progress tracking
            shared_data (Any): Data to be shared across executions (if supported by executor)
        Returns:
            list[Any]: List of results from executing the function with each argument set
        Note:
            The method uses a 64 MiB threshold (TH = 1 << 26) to determine data size categories.
            For pickle size estimation, only the first 10 argument sets are sampled for performance.
        """
        if inspect.iscoroutinefunction(fn):
            return await IoAsyncioStrategy(self.hardware, max_workers).execute(
                fn, args_list, desc=desc
            )
        TH = 1 << 26  # 64 MiB
        has_big_nd = any(
            isinstance(arg, np.ndarray) and arg.nbytes > TH
            for args in args_list[:1000]
            for arg in args
        )
        if has_big_nd:
            return JoblibExecutor(n_jobs=max_workers).map(
                fn, args_list, desc=desc, shared_data=shared_data
            )
        sample = args_list[:10]
        sizes = []
        for args in sample:
            try:
                sizes.append(len(pickle.dumps(args)))
            except Exception:
                sizes.append(0)
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        if avg_size > TH:
            return DaskExecutor().map(fn, args_list, desc=desc, shared_data=shared_data)

        return ProcessExecutor(max_workers=max_workers).map(fn, args_list, desc=desc)

    def _auto_execute_sync(
        self,
        fn: Callable[..., Any],
        args_list: list[tuple],
        max_workers: int | None,
        desc: str | None,
        shared_data: Any,
    ) -> list[Any]:
        """Synchronous version of _auto_execute."""
        if inspect.iscoroutinefunction(fn):
            raise ValueError(
                "Coroutine functions are not supported in execute_sync. Use execute instead."
            )

        TH = 1 << 26  # 64 MiB
        has_big_nd = any(
            isinstance(arg, np.ndarray) and arg.nbytes > TH
            for args in args_list[:1000]
            for arg in args
        )
        if has_big_nd:
            return JoblibExecutor(n_jobs=max_workers).map(
                fn, args_list, desc=desc, shared_data=shared_data
            )

        sample = args_list[:10]
        sizes = []
        for args in sample:
            try:
                sizes.append(len(pickle.dumps(args)))
            except Exception:
                sizes.append(0)
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        if avg_size > TH:
            return DaskExecutor().map(fn, args_list, desc=desc, shared_data=shared_data)

        return ProcessExecutor(max_workers=max_workers).map(fn, args_list, desc=desc)

    def submit(
        self, fn: Callable[..., Any], args: tuple = (), *, task_type: str = "io_bound"
    ):
        exe = ExecutorFactory.create(kind="thread")
        logger.debug(f"Submetendo tarefa única com backend: {exe.__class__.__name__}")
        fut = exe.submit(fn, *args)
        return exe, fut


def query_lazyframe(
    lazyframe: pl.LazyFrame, query_sql: str, table_name: str = "df"
) -> pl.DataFrame:
    """
    Executes an SQL query on a Polars LazyFrame using DuckDB and returns the result as a Polars DataFrame.
    Args:
        lazyframe (pl.LazyFrame): The Polars LazyFrame to query.
        query_sql (str): The SQL query to execute.
        table_name (str, optional): The name to register the LazyFrame as in DuckDB. Defaults to "df".
    Returns:
        pl.DataFrame: The result of the SQL query as a Polars DataFrame.
    """
    polars_df = lazyframe.collect()
    conn = duckdb.connect()
    conn.register(table_name, polars_df)
    rel = conn.execute(query_sql)

    return rel.pl()


async def run_async(
    coro_fn: Callable[..., Any],
    items: Sequence[tuple[Any, ...]],
    *,
    concurrency: int | None = None,
    timeout: float | None = None,
    desc: str | None = None,
    **kwargs: Any,
) -> list[Any]:
    if timeout is not None:
        logger.warning("run_async: 'timeout' está deprecado e será ignorado")
    logger.warning(
        "run_async está deprecado; use ConcurrencyManager.execute(task_type='io_async')"
    )
    cm = ConcurrencyManager()
    return await cm.execute(
        coro_fn, list(items), task_type="io_async", max_workers=concurrency, desc=desc
    )


def first_success(
    fn: Callable[..., _R],
    args_list: list[tuple],
    *,
    ranker: Callable[[Any], float] | None = None,
    max_workers: int = 4,
    perfect_score: float | None = None,
) -> _R:
    """
    Execute fn with different arguments until one succeeds with a good score.

    Args:
        fn: Function to execute
        args_list: List of argument tuples to try
        ranker: Optional function to rank results (higher is better)
        max_workers: Number of parallel workers
        perfect_score: If a result achieves this score, stop early

    Returns:
        The best result according to the ranker

    Raises:
        Exception: If all attempts fail
    """
    if not args_list:
        raise ValueError("No arguments provided")

    if ranker is None:

        def default_ranker(x):
            return 1.0  # Default: all results have equal score

        ranker = default_ranker

    executor = ThreadExecutor(max_workers=max_workers)
    try:
        # Use *args if each element is a tuple of arguments
        futures = [executor.submit(fn, *args) for args in args_list]

        best_result = None
        best_score = float("-inf")
        exceptions = []

        for future in futures:
            try:
                result = future.result()
                score = ranker(result)

                if score > best_score:
                    best_score = score
                    best_result = result

                    # Early exit if perfect score achieved
                    if perfect_score is not None and score >= perfect_score:
                        return best_result

            except Exception as e:
                exceptions.append(e)
                continue

        if best_result is not None:
            return best_result

        # All attempts failed
        if exceptions:
            raise exceptions[0]
        else:
            raise RuntimeError("No successful results")

    finally:
        executor.shutdown()


__all__ = [
    "progress_bar",
    "ThreadExecutor",
    "ProcessExecutor",
    "DaskExecutor",
    "RayExecutor",
    "JoblibExecutor",
    "ExecutorFactory",
    "ConcurrencyManager",
    "CpuMultiprocessingStrategy",
    "IoThreadingStrategy",
    "IoAsyncioStrategy",
    "GpuCudfStrategy",
    "query_lazyframe",
    "first_success",
]

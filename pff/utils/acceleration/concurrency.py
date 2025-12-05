from __future__ import annotations

import asyncio
import functools
import inspect
import math
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
from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar
import threading

class GlobalLock:
    """
    A wrapper around threading.Lock to provide a consistent interface
    and avoid direct threading imports in business logic.
    """
    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._lock.__exit__(exc_type, exc_val, exc_tb)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

def get_lock() -> GlobalLock:
    """Returns a new GlobalLock instance."""
    return GlobalLock()


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

from ..core.logger import logger

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
            ) as progress:  #  refresh mais frequente
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
            logger.debug(f"Rich progress failed: {e}, using fallback")
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

                bar_width = min(30, terminal_width - 60)  #  Mais espaço para texto
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

        #  ADAPTIVE: Use runtime resource detection for max_pending
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
                f" Adaptive ProcessExecutor: {max_workers} workers, "
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
        if not isinstance(args_list, (list, tuple)):
            args_list = list(args_list)

        total = len(args_list)
        if total == 0:
            return []

        future_shared_data = None
        shared_direct = None
        if shared_data is not None:
            try:
                data_size = sys.getsizeof(shared_data)
            except (TypeError, AttributeError):
                data_size = None
            if data_size is not None and data_size <= 256 * 1024:
                shared_direct = shared_data
            else:
                future_shared_data = self._client.scatter(shared_data, broadcast=True)

        threads_total = sum(self._client.nthreads().values()) or 1
        batch_size = kwargs.pop("batch_size", None)
        if batch_size is None and total > threads_total * 256:
            batch_size = min(1024, max(32, total // (threads_total * 16)))

        if batch_size and batch_size > 1:
            def batch_runner(batch: list[tuple], shared=None):
                results = []
                for args in batch:
                    if shared is not None:
                        results.append(fn(shared, *args))
                    else:
                        results.append(fn(*args))
                return results

            batched: list[tuple[int, list[tuple[Any, ...]]]] = []
            for start in range(0, total, batch_size):
                batched.append((start, args_list[start : start + batch_size]))

            futures: dict[Any, tuple[int, int]] = {}
            for offset, batch in batched:
                if future_shared_data is not None:
                    fut = self._client.submit(batch_runner, batch, future_shared_data)
                elif shared_direct is not None:
                    fut = self._client.submit(batch_runner, batch, shared_direct)
                else:
                    fut = self._client.submit(batch_runner, batch)
                futures[fut] = (offset, len(batch))

            results: list[Any] = [None] * total
            completed = 0
            pbar = progress_bar(range(total), total=total, desc=desc, enabled=bool(desc))
            pbar_iter = iter(pbar)

            for fut in dask_as_completed(futures.keys()):
                offset, _ = futures.pop(fut)
                chunk = fut.result()
                results[offset : offset + len(chunk)] = chunk
                completed += len(chunk)
                for _ in range(len(chunk)):
                    try:
                        next(pbar_iter)
                    except StopIteration:
                        break

            return results

        max_pending = min(threads_total * 4, total)
        results: list[Any] = [None] * total
        pending: dict[Any, int] = {}
        idx = 0
        completed = 0

        pbar = progress_bar(range(total), total=total, desc=desc, enabled=bool(desc))
        pbar_iter = iter(pbar)

        while completed < total or pending:
            while len(pending) < max_pending and idx < total:
                if future_shared_data is not None:
                    fut = self._client.submit(fn, future_shared_data, *args_list[idx])
                elif shared_direct is not None:
                    fut = self._client.submit(fn, shared_direct, *args_list[idx])
                else:
                    fut = self._client.submit(fn, *args_list[idx])
                pending[fut] = idx
                idx += 1

            if not pending:
                break

            done_futs = [f for f in pending.keys() if f.done()]

            if not done_futs:
                time.sleep(0.005)
                continue

            for fut in done_futs:
                original_idx = pending.pop(fut)
                results[original_idx] = fut.result()
                completed += 1
                try:
                    next(pbar_iter)
                except StopIteration:
                    pass

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
                runtime_env = init_kwargs.pop("runtime_env", {}) or {}
                env_vars = runtime_env.get("env_vars", {})
                env_vars.setdefault("PYTHONHASHSEED", "0")
                runtime_env["env_vars"] = env_vars
                init_kwargs["runtime_env"] = runtime_env
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
            return self._exec.map(fn, args_list, desc=desc, **kwargs)

        shared_data = kwargs.pop("shared_data", None)
        use_gpu = kwargs.pop("use_gpu", False)
        resources = kwargs.pop("resources", None)
        scheduling_strategy = kwargs.pop("scheduling_strategy", None)
        max_pending_override = kwargs.pop("max_pending", None)
        shared_ref = ray.put(shared_data) if shared_data is not None else None
        call_kwargs = dict(kwargs)

        num_gpus = 0.0
        if isinstance(use_gpu, (int, float)):
            num_gpus = float(use_gpu)
        elif use_gpu:
            num_gpus = 1.0

        remote_options: dict[str, Any] = {}
        if num_gpus > 0:
            remote_options["num_gpus"] = num_gpus
        if resources:
            remote_options["resources"] = resources
        if scheduling_strategy is not None:
            remote_options["scheduling_strategy"] = scheduling_strategy

        args_list = list(args_list)
        total_tasks = len(args_list)

        if total_tasks > 50000:
            batch_size = max(100, total_tasks // 1000)
            return self._map_batched(
                fn,
                args_list,
                batch_size,
                desc,
                shared_ref,
                remote_options,
                call_kwargs,
            )

        import functools

        def _invoke(shared, *call_args):
            target = fn
            extra_args = call_args
            extra_kwargs = {}
            if isinstance(fn, functools.partial):
                target = fn.func
                extra_args = fn.args + call_args
                extra_kwargs = fn.keywords or {}
            combined_kwargs = {**call_kwargs, **extra_kwargs}
            if shared is not None:
                return target(shared, *extra_args, **combined_kwargs)
            return target(*extra_args, **combined_kwargs)

        remote_fn = ray.remote(_invoke)
        if remote_options:
            remote_fn = remote_fn.options(**remote_options)

        try:
            available = ray.available_resources()
        except Exception:
            available = {}
        try:
            cluster = ray.cluster_resources()
        except Exception:
            cluster = {}

        available_cpus = available.get("CPU") or cluster.get("CPU") or 1
        available_cpus = max(1, int(math.floor(available_cpus)))

        resource_limit = available_cpus * 16
        if num_gpus > 0:
            available_gpu = (available.get("GPU") or cluster.get("GPU") or 0.0)
            if available_gpu:
                gpu_slots = max(1, int(available_gpu / num_gpus))
                resource_limit = min(resource_limit, max(gpu_slots, 1))

        max_inflight_default = min(10000, total_tasks if total_tasks else 1)
        if resource_limit <= 0:
            resource_limit = max_inflight_default

        if max_pending_override is not None:
            try:
                max_inflight = max(1, int(max_pending_override))
            except (TypeError, ValueError):
                max_inflight = min(max_inflight_default, max(resource_limit, 1))
        else:
            max_inflight = min(max_inflight_default, max(resource_limit, 1))

        results = [None] * total_tasks
        pending = {}
        idx = 0

        pbar = progress_bar(range(total_tasks), total=total_tasks, desc=desc, enabled=bool(desc))
        pbar_iter = iter(pbar)

        while idx < total_tasks or pending:
            while len(pending) < max_inflight and idx < total_tasks:
                if shared_ref is not None:
                    ref = remote_fn.remote(shared_ref, *args_list[idx])
                else:
                    ref = remote_fn.remote(None, *args_list[idx])
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
        self,
        fn: Callable,
        args_list: list,
        batch_size: int,
        desc: str | None,
        shared_ref,
        remote_options: dict[str, Any],
        call_kwargs: dict[str, Any],
    ) -> list[Any]:
        """Execute in batches to reduce Ray task overhead for 100K+ tasks."""

        import functools

        def _invoke(shared, batch_args):
            target = fn
            partial_args = ()
            partial_kwargs = {}
            if isinstance(fn, functools.partial):
                target = fn.func
                partial_args = fn.args
                partial_kwargs = fn.keywords or {}
            results = []
            for args in batch_args:
                call_args = partial_args + args
                combined_kwargs = {**call_kwargs, **partial_kwargs}
                if shared is not None:
                    results.append(target(shared, *call_args, **combined_kwargs))
                else:
                    results.append(target(*call_args, **combined_kwargs))
            return results

        batch_worker = ray.remote(_invoke)
        if remote_options:
            batch_worker = batch_worker.options(**remote_options)

        batches = [
            args_list[i:i + batch_size] for i in range(0, len(args_list), batch_size)
        ]

        if shared_ref is not None:
            batch_refs = [batch_worker.remote(shared_ref, batch) for batch in batches]
        else:
            batch_refs = [batch_worker.remote(None, batch) for batch in batches]
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
            try:
                return ProcessExecutor(max_workers=max_workers)
            except (PermissionError, OSError, RuntimeError) as exc:
                logger.warning(
                    f"Process backend unavailable ({exc}); using thread executor fallback"
                )
                return ThreadExecutor(max_workers=max_workers)
        if k == "dask":
            return DaskExecutor(address=backend_kwargs.get("address"), **backend_kwargs)
        if k == "ray":
            try:
                return RayExecutor(**backend_kwargs)
            except PermissionError as exc:
                logger.warning(
                    f"Ray backend unavailable due to permission error; falling back to thread executor ({exc})"
                )
                return ThreadExecutor(max_workers=max_workers)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"Ray backend failed to initialize ({exc}); using thread executor fallback"
                )
                return ThreadExecutor(max_workers=max_workers)
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
        # Store only serializable GPU metadata
        self.gpus: list[GPUInfo] = []
        try:
            pynvml.nvmlInit()
            cnt = pynvml.nvmlDeviceGetCount()
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
        finally:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

    def shutdown(self):
        # No handles kept; nothing to release beyond NVML shutdown
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    def __getstate__(self) -> dict[str, Any]:
        return {
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "gpus": [asdict(g) for g in self.gpus],
        }

    def __setstate__(self, state: dict[str, Any]):
        self.physical_cores = state.get("physical_cores", 1)
        self.logical_cores = state.get("logical_cores", self.physical_cores)
        gpus_raw = state.get("gpus", [])
        self.gpus = [GPUInfo(**g) for g in gpus_raw if isinstance(g, dict)]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()

    def get_handle(self, gpu: GPUInfo):
        # Handles not stored; acquire ad-hoc
        try:
            pynvml.nvmlInit()
            if gpu.uuid:
                h = pynvml.nvmlDeviceGetHandleByUUID(gpu.uuid)
            else:
                h = pynvml.nvmlDeviceGetHandleByIndex(gpu.id)
            return h
        except pynvml.NVMLError:
            return None
        finally:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass


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
        from pff.utils.ops.global_interrupt_manager import (
            PRIORITY_HIGH,
            get_interrupt_manager,
            should_stop,
        )

        self.hardware = HardwareManager()
        self._memory_threshold_pct = memory_threshold_pct
        self._should_stop = should_stop
        get_interrupt_manager().register_callback(
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
        proc = psutil.Process()
        with proc.oneshot():
            mem = psutil.virtual_memory()
            rss = proc.memory_info().rss / (1024**3)
        logger.debug(
            "Memória comprometida: %.2f%% (RSS processo: %.2f GB)",
            mem.percent,
            rss,
        )
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
        """Shutdown Ray/Dask workers on interrupt."""
        logger.info("ConcurrencyManager: iniciando shutdown de workers")

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
        if self._should_stop():
            logger.warning("ConcurrencyManager: execution cancelled due to interrupt")
            raise KeyboardInterrupt("Concurrent execution interrupted")

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
        if self._should_stop():
            logger.warning("ConcurrencyManager: execution cancelled due to interrupt")
            raise KeyboardInterrupt("Concurrent execution interrupted")

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
        logger.debug(f"Submitting single task with backend: {exe.__class__.__name__}")
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
    conn.execute(f"PRAGMA threads={max(os.cpu_count() or 1, 1)}")
    conn.execute("PRAGMA enable_object_cache = true")
    temp_dir = os.getenv("PFF_DUCKDB_TEMP")
    if temp_dir:
        conn.execute(f"PRAGMA temp_directory='{temp_dir}'")
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
        logger.warning("run_async: 'timeout' is deprecated and will be ignored")
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


class DurableRayTrainer:
    """Ray durable trainer with fault tolerance and node affinity."""

    def __init__(self, checkpoint_dir: str | None = None, max_retries: int = 3) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.max_retries = max_retries

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def create_durable_trainable(self, train_fn: Callable[..., Any]) -> Any:
        """
        Create a durable trainable with checkpoint persistence.

        Args:
            train_fn: Training function to make durable

        Returns:
            Durable trainable function
        """
        max_retries = self.max_retries
        checkpoint_dir = self.checkpoint_dir

        @ray.remote(max_retries=max_retries)
        def durable_trainable(*args: Any, **kwargs: Any) -> Any:
            try:
                checkpoint_path = None
                if checkpoint_dir:
                    import os
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f"checkpoint_{ray.get_runtime_context().get_job_id()}.pkl"
                    )

                if checkpoint_path and os.path.exists(checkpoint_path):
                    import joblib
                    state = joblib.load(checkpoint_path)
                    kwargs.update(state)

                result = train_fn(*args, **kwargs)

                if checkpoint_path and result:
                    joblib.dump(result, checkpoint_path)

                return result

            except Exception:
                raise

        return durable_trainable

    def create_node_affinity_executor(
        self,
        fn: Callable[..., Any],
        node_ip: str | None = None,
        soft: bool = True
    ) -> Any:
        """
        Create executor with node affinity scheduling.

        Args:
            fn: Function to execute
            node_ip: Target node IP address
            soft: Whether to use soft affinity

        Returns:
            Remote function with node affinity
        """
        max_retries = self.max_retries

        try:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            if node_ip is None:
                import socket
                node_ip = socket.gethostname()

            scheduling_strategy = NodeAffinitySchedulingStrategy(
                node_id=node_ip, soft=soft
            )

            @ray.remote(scheduling_strategy=scheduling_strategy, max_retries=max_retries)
            def node_affinity_fn(*args: Any, **kwargs: Any) -> Any:
                return fn(*args, **kwargs)

            return node_affinity_fn

        except Exception as e:
            logger.warning(f"Node affinity scheduling not available: {e}")
            return ray.remote(fn)

    def execute_with_fault_tolerance(
        self,
        fn: Callable[..., Any],
        args_list: list[tuple[Any, ...]],
        *,
        node_ip: str | None = None,
        checkpoint_every: int = 10,
        desc: str | None = None
    ) -> list[Any]:
        """
        Execute tasks with fault tolerance and node affinity.

        Args:
            fn: Function to execute
            args_list: List of argument tuples
            node_ip: Target node IP for affinity
            checkpoint_every: Checkpoint frequency
            desc: Description for progress tracking

        Returns:
            List of results
        """
        logger.info(f"Executing {len(args_list)} tasks with fault tolerance")

        max_retries = self.max_retries

        try:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            if node_ip is None:
                import socket
                node_ip = socket.gethostname()

            scheduling_strategy = NodeAffinitySchedulingStrategy(
                node_id=node_ip, soft=True
            )

            @ray.remote(scheduling_strategy=scheduling_strategy, max_retries=max_retries)
            def resilient_fn(*args: Any) -> Any:
                return fn(*args)

        except Exception:
            logger.warning("Node affinity not available, using standard execution")
            resilient_fn = ray.remote(fn)

        futures = [resilient_fn.remote(*args) for args in args_list]

        results = []
        for i, future in enumerate(futures):
            try:
                result = ray.get(future)
                results.append(result)

                if (i + 1) % checkpoint_every == 0:
                    # Progress updates are debug-level to avoid spam
                    logger.debug(f"Ray tasks progress: {i + 1}/{len(args_list)} completed")

            except Exception as e:
                logger.error(f"Task {i} failed permanently: {e}")
                results.append(None)

        logger.debug(f"Ray batch completed: {len(results)}/{len(args_list)} tasks")
        return results


def get_durable_trainer(
    checkpoint_dir: str | None = None,
    max_retries: int = 3
) -> DurableRayTrainer:
    """Get durable trainer instance."""
    return DurableRayTrainer(checkpoint_dir=checkpoint_dir, max_retries=max_retries)


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
    "DurableRayTrainer",
    "get_durable_trainer",
    "query_lazyframe",
    "first_success",
]

from __future__ import annotations

import asyncio
import functools
import inspect
import math
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

try:
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
except ImportError:
    Progress = None  # type: ignore[assignment,misc]
    BarColumn = MofNCompleteColumn = SpinnerColumn = TaskProgressColumn = TextColumn = (
        TimeElapsedColumn
    ) = TimeRemainingColumn = None  # type: ignore[assignment]

from ..core.logging import logger

Args = tuple[Any, ...]
_R = TypeVar("_R")

if TYPE_CHECKING:
    pass

_duckdb = None
_joblib = None
_polars = None
_psutil = None
_pynvml = None
_ray = None
_dask_client = None
_dask_as_completed = None


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


def _require_duckdb():
    global _duckdb
    if _duckdb is None:
        try:
            import duckdb as _duckdb_mod
        except ImportError as exc:
            raise RuntimeError(
                "duckdb não está disponível; instale a dependência para usar query_lazyframe."
            ) from exc
        _duckdb = _duckdb_mod
    return _duckdb


def _require_joblib():
    global _joblib
    if _joblib is None:
        try:
            import joblib as _joblib_mod
        except ImportError as exc:
            raise RuntimeError(
                "joblib não está disponível; instale a dependência para usar JoblibExecutor."
            ) from exc
        _joblib = _joblib_mod
    return _joblib


def _require_polars():
    global _polars
    if _polars is None:
        try:
            import polars as _polars_mod
        except ImportError as exc:
            raise RuntimeError(
                "polars não está disponível; instale a dependência para usar query_lazyframe."
            ) from exc
        _polars = _polars_mod
    return _polars


def _require_psutil():
    global _psutil
    if _psutil is None:
        try:
            import psutil as _psutil_mod
        except ImportError as exc:
            raise RuntimeError(
                "psutil não está disponível; instale a dependência para usar ConcurrencyManager."
            ) from exc
        _psutil = _psutil_mod
    return _psutil


def _try_import_pynvml() -> Any:
    global _pynvml
    if _pynvml is None:
        try:
            import pynvml as _pynvml_mod
        except ImportError:
            _pynvml = False
        else:
            _pynvml = _pynvml_mod
    return _pynvml if _pynvml is not False else None


def _require_ray():
    global _ray
    if _ray is None:
        try:
            import ray as _ray_mod
        except ImportError as exc:
            raise RuntimeError(
                "ray não está disponível; instale a dependência para usar RayExecutor."
            ) from exc
        _ray = _ray_mod
    return _ray


def _require_dask():
    global _dask_client, _dask_as_completed
    if _dask_client is None or _dask_as_completed is None:
        try:
            from dask.distributed import Client as DaskClient
            from dask.distributed import (
                as_completed as dask_as_completed,
            )
        except ImportError as exc:
            raise RuntimeError(
                "dask.distributed não está disponível; instale a dependência para usar DaskExecutor."
            ) from exc
        _dask_client = DaskClient
        _dask_as_completed = dask_as_completed
    return _dask_client, _dask_as_completed


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
    if Progress is not None and sys.stderr.isatty():
        Spinner = SpinnerColumn
        Text = TextColumn
        Bar = BarColumn
        TaskProgress = TaskProgressColumn
        MofN = MofNCompleteColumn
        Elapsed = TimeElapsedColumn
        Remaining = TimeRemainingColumn

        columns = [
            Spinner() if Spinner is not None else None,
            Text("[progress.description]{task.description}") if Text is not None else None,
            Bar(bar_width=40) if Bar is not None else None,
            TaskProgress() if TaskProgress is not None else None,
            Text("•") if Text is not None else None,
            MofN() if MofN is not None else None,
            Elapsed() if Elapsed is not None else None,
            Remaining() if Remaining is not None else None,
        ]
        columns = [c for c in columns if c is not None]
        try:
            with Progress(*columns, transient=False, refresh_per_second=4) as progress:
                task = progress.add_task(desc or "Processando...", total=total)
                for item in iterable:
                    yield item
                    progress.update(task, advance=1)
                progress.update(task, completed=total if total else progress.tasks[task].completed)
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
                if idx > 1 and elapsed > 1:
                    rate = idx / elapsed
                    if rate > 0:
                        eta_seconds = (total - idx) / rate
                        eta_str = f" ETA: {_format_time(eta_seconds)}"
                    else:
                        eta_str = " ETA: calculando..."
                else:
                    eta_str = " ETA: calculando..."

                bar_width = min(30, terminal_width - 60)
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
                    f"\r{desc or 'Processando'} {spinner} {idx} items [{_format_time(elapsed)}]"
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
        final_msg = f"\r{desc or 'Concluído'}: {items_processed} items em {_format_time(elapsed)}"
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
        self,
        fn: Callable[..., Any],
        args_list: Iterable[Args],
        *,
        desc: str | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        args_list_materialized = (
            list(args_list) if not isinstance(args_list, (list, tuple)) else args_list
        )
        total = len(args_list_materialized)
        if total == 0:
            return []

        max_workers = getattr(self._pool, "_max_workers", None) or os.cpu_count() or 4
        max_pending = min(total, max(100, max_workers * 10))

        results: list[Any] = [None] * total
        pending: dict[Any, int] = {}
        idx = 0
        completed = 0

        pbar = progress_bar(range(total), total=total, desc=desc)
        pbar_iter = iter(pbar)

        while completed < total or pending:
            from pff.shared.ops.global_interrupt_manager import should_stop

            if should_stop():
                break

            while len(pending) < max_pending and idx < total:
                if should_stop():
                    break
                fut = self._pool.submit(fn, *args_list_materialized[idx])
                pending[fut] = idx
                idx += 1

            if not pending:
                break

            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED, timeout=0.1)
            if not done:
                continue

            for fut in done:
                original_idx = pending.pop(fut)
                results[original_idx] = fut.result()
                completed += 1
                try:
                    next(pbar_iter)
                except StopIteration:
                    pass

        return results

    def submit(self, fn, *args):
        return self._pool.submit(fn, *args)

    def shutdown(self):
        self._pool.shutdown(wait=True)


class ProcessExecutor(BaseExecutor):
    def __init__(self, max_workers: int | None = None):
        ctx = mp.get_context("spawn")
        self._pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)

    @staticmethod
    def _batch_worker(fn: Callable[..., Any], batch: list[Args]) -> list[Any]:
        return [fn(*args) for args in batch]

    def map(
        self,
        fn: Callable[..., Any],
        args_list: Iterable[Args],
        *,
        desc: str | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Execute function over args_list with bounded memory using lazy task submission.

        Prevents OOM by limiting concurrent futures to avoid memory explosion
        with large task lists (e.g., 100K+ items).
        """
        if not isinstance(args_list, (list, tuple)):
            args_list = list(args_list)

        total = len(args_list)
        if total == 0:
            return []

        chunksize = kwargs.pop("chunksize", None)
        try:
            if chunksize is not None:
                chunksize = int(chunksize)
                if chunksize < 2:
                    chunksize = None
        except (TypeError, ValueError):
            chunksize = None

        try:
            from pff.shared.system.resource_manager import get_resource_manager

            resource_manager = get_resource_manager()

            max_workers = getattr(self._pool, "_max_workers", None) or os.cpu_count() or 4
            limits = resource_manager.calculate_limits(
                task_count=total,
                estimated_task_size=5000,
                max_workers=max_workers,
            )
            max_pending = limits.max_pending_futures

            from pff.shared.core.logging import logger

            logger.debug(
                f" Adaptive ProcessExecutor: {max_workers} workers, "
                f"{max_pending} max pending (90% memory safe)"
            )
        except Exception:
            max_workers = getattr(self._pool, "_max_workers", None) or os.cpu_count() or 4
            max_pending = max(100, max_workers * 10)

        if chunksize:
            batches = [
                (start, args_list[start : start + chunksize])
                for start in range(0, total, chunksize)
            ]
            total_batches = len(batches)
            max_pending = min(max_pending, total_batches)
            results: list[Any] = [None] * total
            pending_batches: dict[Any, tuple[int, int]] = {}
            idx = 0
            completed = 0

            pbar = progress_bar(range(total), total=total, desc=desc, enabled=bool(desc))
            pbar_iter = iter(pbar)

            while completed < total or pending_batches:
                while len(pending_batches) < max_pending and idx < total_batches:
                    offset, batch_args = batches[idx]
                    batch_args_any: Any = batch_args
                    fut = self._pool.submit(self._batch_worker, fn, batch_args_any)
                    pending_batches[fut] = (offset, len(batch_args))
                    idx += 1

                if not pending_batches:
                    break

                done, _ = wait(pending_batches.keys(), return_when=FIRST_COMPLETED, timeout=0.1)
                if not done:
                    continue

                for fut in done:
                    offset, batch_len = pending_batches.pop(fut)
                    batch_results = fut.result()
                    results[offset : offset + batch_len] = batch_results
                    completed += batch_len
                    for _ in range(int(batch_len)):
                        try:
                            next(pbar_iter)
                        except StopIteration:
                            break

            return results

        results = [None] * total
        pending_tasks: dict[Any, int] = {}
        idx = 0
        completed = 0

        pbar = progress_bar(range(total), total=total, desc=desc, enabled=bool(desc))
        pbar_iter = iter(pbar)

        while completed < total or pending_tasks:
            while len(pending_tasks) < max_pending and idx < total:
                fut = self._pool.submit(fn, *args_list[idx])
                pending_tasks[fut] = idx
                idx += 1

            if not pending_tasks:
                break

            done, _ = wait(pending_tasks.keys(), return_when=FIRST_COMPLETED, timeout=0.1)
            if not done:
                continue

            for fut in done:
                original_idx = pending_tasks.pop(fut)
                results[original_idx] = fut.result()
                completed += 1
                try:
                    next(pbar_iter)
                except StopIteration:
                    pass

        return results

    def submit(self, fn, *args):
        return self._pool.submit(fn, *args)

    def shutdown(self):
        self._pool.shutdown(wait=True)


class DaskExecutor(BaseExecutor):
    def __init__(self, address: str | None = None, **client_kwargs: Any):
        DaskClient, dask_as_completed = _require_dask()
        self._client = DaskClient(address=address, **client_kwargs)
        self._as_completed = dask_as_completed

    def map(
        self,
        fn: Callable[..., Any],
        args_list: Iterable[tuple],
        *,
        desc: str | None = None,
        shared_data: Any = None,
        **kwargs: Any,
    ) -> list[Any]:
        if not isinstance(args_list, list):
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

        try:
            nthreads_data: Any = self._client.nthreads()
            if hasattr(nthreads_data, "values"):
                threads_total = sum(nthreads_data.values()) or 1
            else:
                threads_total = 1
        except Exception:
            threads_total = 1
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
                chunk = cast(list[tuple[Any, ...]], list(args_list[start : start + batch_size]))
                batched.append((start, chunk))

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

            for fut in self._as_completed(futures.keys()):
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
        results = [None] * total
        pending_dask: dict[Any, int] = {}
        idx = 0
        completed = 0
        ac = self._as_completed([])

        pbar = progress_bar(range(total), total=total, desc=desc, enabled=bool(desc))
        pbar_iter = iter(pbar)

        while completed < total or pending_dask:
            while len(pending_dask) < max_pending and idx < total:
                if future_shared_data is not None:
                    fut = self._client.submit(fn, future_shared_data, *args_list[idx])
                elif shared_direct is not None:
                    fut = self._client.submit(fn, shared_direct, *args_list[idx])
                else:
                    fut = self._client.submit(fn, *args_list[idx])
                pending_dask[fut] = idx
                ac.add(fut)
                idx += 1

            if not pending_dask:
                break

            try:
                fut = next(ac)
            except StopIteration:
                continue

            if fut in pending_dask:
                original_idx = pending_dask.pop(fut)
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
            logger.warning("Ray on Windows is unstable; using DaskExecutor as fallback")
            self._exec = DaskExecutor(**init_kwargs)
            self._ray = None
        else:
            ray = _require_ray()
            self._ray = ray
            if not ray.is_initialized():
                runtime_env = init_kwargs.pop("runtime_env", {}) or {}
                env_vars = runtime_env.get("env_vars", {})
                env_vars.setdefault("PYTHONHASHSEED", "0")
                env_vars.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
                cwd = os.getcwd()
                python_path = env_vars.get("PYTHONPATH", "")
                if python_path:
                    env_vars["PYTHONPATH"] = f"{cwd}:{python_path}"
                else:
                    env_vars["PYTHONPATH"] = cwd
                runtime_env["env_vars"] = env_vars
                init_kwargs["runtime_env"] = runtime_env
                logger.debug(f"Initializing Ray with PYTHONPATH={cwd}")
                ray.init(**init_kwargs)
            self._exec = None

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

        ray = self._ray or _require_ray()

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
            available_gpu = available.get("GPU") or cluster.get("GPU") or 0.0
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

        results: list[Any] = [None] * total_tasks
        pending: dict[Any, int] = {}
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

            ready, _ = ray.wait(
                list(pending.keys()), num_returns=min(100, len(pending)), timeout=0.01
            )

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

        ray = self._ray or _require_ray()

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

        batches = [args_list[i : i + batch_size] for i in range(0, len(args_list), batch_size)]

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
        ray = self._ray or _require_ray()
        remote_fn = ray.remote(fn)
        return remote_fn.remote(*args)

    def shutdown(self):
        pass


class JoblibExecutor(BaseExecutor):
    """
    Executor baseado em Joblib, usando memmapping para grandes numpy.ndarray
    """

    def __init__(self, n_jobs: int | None = None, mmap_threshold: int = 1 << 26):
        self._joblib = _require_joblib()
        self.n_jobs = n_jobs or self._joblib.cpu_count()
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
        if isinstance(shared_data, np.ndarray) and shared_data.nbytes >= self.mmap_thresh:
            shm_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mmap", dir=shm_dir)
            tmp.close()
            self._joblib.dump(shared_data, tmp.name, compress=False)
            shared_mm = self._joblib.load(tmp.name, mmap_mode="r")
            mmap_path = tmp.name
        else:
            shared_mm = shared_data

        target_fn = functools.partial(fn, shared_mm) if shared_mm is not None else fn

        def _wrapper(args):
            return target_fn(*args)

        results = list(
            self._joblib.Parallel(n_jobs=self.n_jobs)(
                self._joblib.delayed(_wrapper)(args) for args in progress_bar(args_list, desc=desc)
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
        raise NotImplementedError("JoblibExecutor does not support asynchronous 'submit'.")

    def shutdown(self):
        pass


class ExecutorFactory:
    @staticmethod
    def create(kind: str, max_workers: int | None = None, **backend_kwargs: Any) -> BaseExecutor:
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
            except Exception as exc:
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
        psutil = _require_psutil()
        self.physical_cores = psutil.cpu_count(logical=False) or 1
        self.logical_cores = psutil.cpu_count(logical=True) or 1
        self.gpus: list[GPUInfo] = []
        pynvml: Any = _try_import_pynvml()
        if pynvml is None:
            logger.debug("pynvml not available; GPU detection disabled")
            return
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
        except Exception as exc:
            logger.debug(f"Failed to initialize GPU metadata via NVML: {exc}")
            self.gpus = []
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def shutdown(self):
        pynvml = _try_import_pynvml()
        if pynvml is None:
            return
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
        pynvml = _try_import_pynvml()
        if pynvml is None:
            return None
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

    def get_telemetry(self) -> dict[str, Any]:
        """Returns real-time hardware telemetry."""
        psutil = _require_psutil()
        with psutil.Process().oneshot():
            cpu_usage = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()

        telemetry = {
            "cpu_usage": cpu_usage,
            "ram_usage_pct": mem.percent,
            "ram_total_gb": mem.total / (1024**3),
            "ram_used_gb": mem.used / (1024**3),
            "gpus": [],
        }

        pynvml = _try_import_pynvml()
        if pynvml:
            try:
                pynvml.nvmlInit()
                for gpu in self.gpus:
                    handle = self.get_handle(gpu)
                    if handle:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        telemetry["gpus"].append(
                            {
                                "id": gpu.id,
                                "name": gpu.name,
                                "utilization": util.gpu,
                                "vram_total": mem_info.total,
                                "vram_used": mem_info.used,
                                "vram_usage_pct": (mem_info.used / mem_info.total * 100),
                            }
                        )
            except Exception:
                pass
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

        return telemetry


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

                self._ray = ray
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


class ConcurrencyManager:
    def __init__(self, memory_threshold_pct: float | None = None):
        """
        Initialize ConcurrencyManager with hardware detection and memory monitoring.

        Args:
            memory_threshold_pct: Maximum RAM usage percentage before raising error.
                                 Default: read from env PFF_MEMORY_THRESHOLD_PCT or 85.0
        """
        from pff.shared.ops.global_interrupt_manager import (
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
                logger.warning(f"GPUs near memory limit: {alerts}. Consider reducing batch sizes.")

    def _shutdown_workers(self) -> None:
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
        if self._should_stop():
            logger.warning("ConcurrencyManager: execution cancelled due to interrupt")
            raise KeyboardInterrupt("Concurrent execution interrupted")

        t = task_type.lower()
        backend_kwargs = backend_kwargs or {}

        if t == "auto":
            return self._auto_execute_sync(fn, args_list, max_workers, desc, shared_data)
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
        if self._should_stop():
            logger.warning("ConcurrencyManager: execution cancelled due to interrupt")
            raise KeyboardInterrupt("Concurrent execution interrupted")

        self._check_memory_safety()

        t = task_type.lower()
        backend_kwargs = backend_kwargs or {}

        if t == "auto":
            return await self._auto_execute(fn, args_list, max_workers, desc, shared_data)
        elif t in ("io_thread", "thread"):
            logger.debug(
                "Executing tasks in thread pool (async wrapper)",
                task_count=len(args_list),
                workers=max_workers,
            )
            strategy = IoThreadingStrategy(self.hardware, max_workers)
            try:
                return await strategy.execute(fn, args_list, desc=desc)
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
                return await strategy.execute(fn, args_list, desc=desc)
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
            strategy = GpuCudfStrategy(self.hardware)  # type: ignore[assignment]
            return await strategy.execute(fn, args_list, **backend_kwargs)
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

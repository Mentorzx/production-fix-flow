"""Process-based executor implementation using multiprocessing."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any

from ..protocols import Args, BaseExecutor
from ..utils import progress_bar


class ProcessExecutor(BaseExecutor):
    """Process pool executor with memory-safe bounded execution."""

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
            from ....system.resource_manager import get_resource_manager

            resource_manager = get_resource_manager()

            max_workers = getattr(self._pool, "_max_workers", None) or os.cpu_count() or 4
            limits = resource_manager.calculate_limits(
                task_count=total,
                estimated_task_size=5000,
                max_workers=max_workers,
            )
            max_pending = limits.max_pending_futures

            from ....core.logging import logger

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

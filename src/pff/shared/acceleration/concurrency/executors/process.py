"""Process-based executor implementation using multiprocessing."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any

from ..protocols import Args, BaseExecutor
from ..utils import progress_bar
from ....system.probe import get_safe_cpu_count


class ProcessExecutor(BaseExecutor):
    """Process pool executor with memory-safe bounded execution."""

    def __init__(self, max_workers: int | None = None):
        """Execute init.



        Args:

            max_workers: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

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

        chunksize = self._resolve_chunksize(kwargs.pop("chunksize", None))
        max_pending = self._resolve_max_pending(total)
        if chunksize:
            return self._map_in_batches(
                fn=fn,
                args_list=list(args_list),
                total=total,
                chunksize=chunksize,
                max_pending=max_pending,
                desc=desc,
            )
        return self._map_unbatched(
            fn=fn,
            args_list=list(args_list),
            total=total,
            max_pending=max_pending,
            desc=desc,
        )

    @staticmethod
    def _resolve_chunksize(raw_chunksize: Any) -> int | None:
        """Execute resolve chunksize.



        Args:

            raw_chunksize: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        try:
            if raw_chunksize is None:
                return None
            chunksize = int(raw_chunksize)
            return chunksize if chunksize >= 2 else None
        except (TypeError, ValueError):
            return None

    def _resolve_max_pending(self, total: int) -> int:
        """Execute resolve max pending.



        Args:

            total: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        try:
            from ....system.resource_manager import get_resource_manager

            resource_manager = get_resource_manager()
            max_workers = getattr(
                self._pool, "_max_workers", None
            ) or get_safe_cpu_count(logical=True)
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
            return max_pending
        except Exception:
            max_workers = getattr(
                self._pool, "_max_workers", None
            ) or get_safe_cpu_count(logical=True)
            return max(100, max_workers * 10)

    def _map_in_batches(
        self,
        *,
        fn: Callable[..., Any],
        args_list: list[Args],
        total: int,
        chunksize: int,
        max_pending: int,
        desc: str | None,
    ) -> list[Any]:
        """Execute map in batches.



        Args:

            fn: Input value used by this callable.

            args_list: Input value used by this callable.

            total: Input value used by this callable.

            chunksize: Input value used by this callable.

            max_pending: Input value used by this callable.

            desc: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

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
        pbar_iter = iter(
            progress_bar(range(total), total=total, desc=desc, enabled=bool(desc))
        )
        while completed < total or pending_batches:
            while len(pending_batches) < max_pending and idx < total_batches:
                offset, batch_args = batches[idx]
                fut = self._pool.submit(self._batch_worker, fn, batch_args)
                pending_batches[fut] = (offset, len(batch_args))
                idx += 1
            if not pending_batches:
                break
            done, _ = wait(
                pending_batches.keys(), return_when=FIRST_COMPLETED, timeout=0.1
            )
            if not done:
                continue
            for fut in done:
                offset, batch_len = pending_batches.pop(fut)
                results[offset : offset + batch_len] = fut.result()
                completed += batch_len
                self._advance_progress(pbar_iter, int(batch_len))
        return results

    def _map_unbatched(
        self,
        *,
        fn: Callable[..., Any],
        args_list: list[Args],
        total: int,
        max_pending: int,
        desc: str | None,
    ) -> list[Any]:
        """Execute map unbatched.



        Args:

            fn: Input value used by this callable.

            args_list: Input value used by this callable.

            total: Input value used by this callable.

            max_pending: Input value used by this callable.

            desc: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        results = [None] * total
        pending_tasks: dict[Any, int] = {}
        idx = 0
        completed = 0
        pbar_iter = iter(
            progress_bar(range(total), total=total, desc=desc, enabled=bool(desc))
        )
        while completed < total or pending_tasks:
            while len(pending_tasks) < max_pending and idx < total:
                fut = self._pool.submit(fn, *args_list[idx])
                pending_tasks[fut] = idx
                idx += 1
            if not pending_tasks:
                break
            done, _ = wait(
                pending_tasks.keys(), return_when=FIRST_COMPLETED, timeout=0.1
            )
            if not done:
                continue
            for fut in done:
                original_idx = pending_tasks.pop(fut)
                results[original_idx] = fut.result()
                completed += 1
                self._advance_progress(pbar_iter, 1)
        return results

    @staticmethod
    def _advance_progress(pbar_iter: Any, steps: int) -> None:
        """Execute advance progress.



        Args:

            pbar_iter: Input value used by this callable.

            steps: Input value used by this callable.

        """

        for _ in range(steps):
            try:
                next(pbar_iter)
            except StopIteration:
                break

    def submit(self, fn, *args):
        """Execute submit.



        Args:

            fn: Input value used by this callable.

            *args: Additional positional arguments.



        Returns:

            Return value produced by the callable.

        """

        return self._pool.submit(fn, *args)

    def shutdown(self):
        """Execute shutdown."""

        self._pool.shutdown(wait=True)

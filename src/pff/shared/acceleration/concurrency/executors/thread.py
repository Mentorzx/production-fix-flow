"""Thread-based executor implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from ..protocols import Args, BaseExecutor
from ..utils import progress_bar
from ....system.probe import get_safe_cpu_count


class ThreadExecutor(BaseExecutor):
    """Thread pool executor with progress tracking."""

    def __init__(self, max_workers: int | None = None):
        """Execute init.



        Args:

            max_workers: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def map(
        self,
        fn: Callable[..., Any],
        args_list: Iterable[Args],
        *,
        desc: str | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Execute function over args_list with bounded memory."""
        args_list_materialized = (
            list(args_list) if not isinstance(args_list, (list, tuple)) else args_list
        )
        total = len(args_list_materialized)
        if total == 0:
            return []

        max_workers = getattr(self._pool, "_max_workers", None) or get_safe_cpu_count(
            logical=True
        )
        max_pending = min(total, max(100, max_workers * 10))

        results: list[Any] = [None] * total
        pending: dict[Any, int] = {}
        idx = 0
        completed = 0

        pbar = progress_bar(range(total), total=total, desc=desc)
        pbar_iter = iter(pbar)

        while completed < total or pending:
            from ....ops.global_interrupt_manager import should_stop

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

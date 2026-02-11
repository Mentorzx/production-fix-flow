"""Dask-based distributed executor implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

from ..protocols import BaseExecutor
from ..utils import _require_dask, progress_bar


class DaskExecutor(BaseExecutor):
    """Dask distributed executor with batching support."""

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
                data_size = __import__("sys").getsizeof(shared_data)
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

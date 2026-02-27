"""Dask-based distributed executor implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

from ..protocols import BaseExecutor
from ..utils import _require_dask, progress_bar


class DaskExecutor(BaseExecutor):
    """Dask distributed executor with batching support."""

    def __init__(self, address: str | None = None, **client_kwargs: Any):
        """Execute init.



        Args:

            address: Optional input value.

            **client_kwargs: Additional keyword arguments.

        """

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
        """Execute map.



        Args:

            fn: Input value used by this callable.

            args_list: Input value used by this callable.

            desc: Optional input value.

            shared_data: Optional input value.

            **kwargs: Additional keyword arguments.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if not isinstance(args_list, list):
            args_list = list(args_list)

        total = len(args_list)
        if total == 0:
            return []

        future_shared_data, shared_direct = self._resolve_shared_data(shared_data)
        threads_total = self._resolve_threads_total()
        batch_size = self._resolve_batch_size(total, threads_total, kwargs)
        if batch_size and batch_size > 1:
            return self._map_batched(
                fn=fn,
                args_list=cast(list[tuple[Any, ...]], args_list),
                total=total,
                batch_size=batch_size,
                desc=desc,
                future_shared_data=future_shared_data,
                shared_direct=shared_direct,
            )
        return self._map_streaming(
            fn=fn,
            args_list=cast(list[tuple[Any, ...]], args_list),
            total=total,
            threads_total=threads_total,
            desc=desc,
            future_shared_data=future_shared_data,
            shared_direct=shared_direct,
        )

    def _resolve_shared_data(self, shared_data: Any) -> tuple[Any | None, Any | None]:
        """Execute resolve shared data.



        Args:

            shared_data: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if shared_data is None:
            return None, None
        try:
            data_size = __import__("sys").getsizeof(shared_data)
        except (TypeError, AttributeError):
            data_size = None
        if data_size is not None and data_size <= 256 * 1024:
            return None, shared_data
        return self._client.scatter(shared_data, broadcast=True), None

    def _resolve_threads_total(self) -> int:
        """Execute resolve threads total.



        Returns:

            Return value produced by the callable.

        """

        try:
            nthreads_data: Any = self._client.nthreads()
            if hasattr(nthreads_data, "values"):
                return sum(nthreads_data.values()) or 1
        except Exception:
            return 1
        return 1

    def _resolve_batch_size(
        self, total: int, threads_total: int, kwargs: dict[str, Any]
    ) -> int | None:
        """Execute resolve batch size.



        Args:

            total: Input value used by this callable.

            threads_total: Input value used by this callable.

            kwargs: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        batch_size = kwargs.pop("batch_size", None)
        if batch_size is None and total > threads_total * 256:
            return min(1024, max(32, total // (threads_total * 16)))
        return batch_size

    @staticmethod
    def _batch_runner(
        fn: Callable[..., Any], batch: list[tuple], shared: Any | None = None
    ) -> list[Any]:
        """Execute batch runner.



        Args:

            fn: Input value used by this callable.

            batch: Input value used by this callable.

            shared: Optional input value.



        Returns:

            Return value produced by the callable.

        """

        results = []
        for args in batch:
            if shared is not None:
                results.append(fn(shared, *args))
            else:
                results.append(fn(*args))
        return results

    def _map_batched(
        self,
        *,
        fn: Callable[..., Any],
        args_list: list[tuple[Any, ...]],
        total: int,
        batch_size: int,
        desc: str | None,
        future_shared_data: Any | None,
        shared_direct: Any | None,
    ) -> list[Any]:
        """Execute map batched.



        Args:

            fn: Input value used by this callable.

            args_list: Input value used by this callable.

            total: Input value used by this callable.

            batch_size: Input value used by this callable.

            desc: Input value used by this callable.

            future_shared_data: Input value used by this callable.

            shared_direct: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        batched: list[tuple[int, list[tuple[Any, ...]]]] = []
        for start in range(0, total, batch_size):
            batched.append((start, list(args_list[start : start + batch_size])))
        futures: dict[Any, tuple[int, int]] = {}
        for offset, batch in batched:
            fut = self._submit_batch(
                fn=fn,
                batch=batch,
                future_shared_data=future_shared_data,
                shared_direct=shared_direct,
            )
            futures[fut] = (offset, len(batch))
        results: list[Any] = [None] * total
        pbar_iter = iter(
            progress_bar(range(total), total=total, desc=desc, enabled=bool(desc))
        )
        for fut in self._as_completed(futures.keys()):
            offset, _ = futures.pop(fut)
            chunk = fut.result()
            results[offset : offset + len(chunk)] = chunk
            self._advance_progress(pbar_iter, len(chunk))
        return results

    def _submit_batch(
        self,
        *,
        fn: Callable[..., Any],
        batch: list[tuple[Any, ...]],
        future_shared_data: Any | None,
        shared_direct: Any | None,
    ) -> Any:
        """Execute submit batch.



        Args:

            fn: Input value used by this callable.

            batch: Input value used by this callable.

            future_shared_data: Input value used by this callable.

            shared_direct: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if future_shared_data is not None:
            return self._client.submit(
                self._batch_runner, fn, batch, future_shared_data
            )
        if shared_direct is not None:
            return self._client.submit(self._batch_runner, fn, batch, shared_direct)
        return self._client.submit(self._batch_runner, fn, batch)

    def _map_streaming(
        self,
        *,
        fn: Callable[..., Any],
        args_list: list[tuple[Any, ...]],
        total: int,
        threads_total: int,
        desc: str | None,
        future_shared_data: Any | None,
        shared_direct: Any | None,
    ) -> list[Any]:
        """Execute map streaming.



        Args:

            fn: Input value used by this callable.

            args_list: Input value used by this callable.

            total: Input value used by this callable.

            threads_total: Input value used by this callable.

            desc: Input value used by this callable.

            future_shared_data: Input value used by this callable.

            shared_direct: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        max_pending = min(threads_total * 4, total)
        results = [None] * total
        pending_dask: dict[Any, int] = {}
        idx = 0
        completed = 0
        ac = self._as_completed([])
        pbar_iter = iter(
            progress_bar(range(total), total=total, desc=desc, enabled=bool(desc))
        )
        while completed < total or pending_dask:
            while len(pending_dask) < max_pending and idx < total:
                fut = self._submit_item(
                    fn=fn,
                    args=args_list[idx],
                    future_shared_data=future_shared_data,
                    shared_direct=shared_direct,
                )
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
                self._advance_progress(pbar_iter, 1)
        return results

    def _submit_item(
        self,
        *,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        future_shared_data: Any | None,
        shared_direct: Any | None,
    ) -> Any:
        """Execute submit item.



        Args:

            fn: Input value used by this callable.

            args: Input value used by this callable.

            future_shared_data: Input value used by this callable.

            shared_direct: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if future_shared_data is not None:
            return self._client.submit(fn, future_shared_data, *args)
        if shared_direct is not None:
            return self._client.submit(fn, shared_direct, *args)
        return self._client.submit(fn, *args)

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

        return self._client.submit(fn, *args)

    def shutdown(self):
        """Execute shutdown."""

        self._client.close()

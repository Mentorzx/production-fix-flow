"""Joblib-based executor implementation with memory mapping support."""

from __future__ import annotations

import functools
import os
import tempfile
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from ..protocols import BaseExecutor
from ..utils import _require_joblib, progress_bar


class JoblibExecutor(BaseExecutor):
    """
    Executor baseado em Joblib, usando memmapping para grandes numpy.ndarray
    """

    def __init__(self, n_jobs: int | None = None, mmap_threshold: int = 1 << 26):
        """Execute init.



        Args:

            n_jobs: Optional input value.

            mmap_threshold: Optional input value.

        """

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
        """Execute shutdown."""

        pass

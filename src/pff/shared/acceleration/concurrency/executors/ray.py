"""Ray-based distributed executor implementation."""

from __future__ import annotations

import functools
import math
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ..protocols import Args, BaseExecutor
from ..utils import _require_ray, progress_bar


class RayExecutor(BaseExecutor):
    """Ray distributed executor with adaptive batching for massive parallelism."""

    def __init__(self, **init_kwargs: Any):
        """Execute init.



        Args:

            **init_kwargs: Additional keyword arguments.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        from ....core.logging import logger

        if sys.platform == "win32":
            logger.warning("Ray on Windows is unstable; using DaskExecutor as fallback")
            from .dask import DaskExecutor

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
                # In src-layout repos, Ray workers need "src" on sys.path to import the package.
                src_root = Path(cwd) / "src"
                import_root = str(src_root) if (src_root / "pff").is_dir() else cwd
                python_path = env_vars.get("PYTHONPATH", "")
                if python_path:
                    env_vars["PYTHONPATH"] = f"{import_root}:{python_path}"
                else:
                    env_vars["PYTHONPATH"] = import_root
                runtime_env["env_vars"] = env_vars
                init_kwargs["runtime_env"] = runtime_env
                logger.debug(f"Initializing Ray with PYTHONPATH={import_root}")
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

        num_gpus, remote_options = self._resolve_remote_options(
            use_gpu=use_gpu,
            resources=resources,
            scheduling_strategy=scheduling_strategy,
        )

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
        remote_fn = self._build_remote_invoker(
            ray=ray,
            fn=fn,
            call_kwargs=call_kwargs,
            remote_options=remote_options,
        )
        max_inflight = self._resolve_max_inflight(
            ray=ray,
            total_tasks=total_tasks,
            num_gpus=num_gpus,
            max_pending_override=max_pending_override,
        )
        return self._map_unbatched(
            ray=ray,
            remote_fn=remote_fn,
            args_list=args_list,
            total_tasks=total_tasks,
            max_inflight=max_inflight,
            shared_ref=shared_ref,
            desc=desc,
        )

    @staticmethod
    def _resolve_remote_options(
        *,
        use_gpu: Any,
        resources: Any,
        scheduling_strategy: Any,
    ) -> tuple[float, dict[str, Any]]:
        """Execute resolve remote options.



        Args:

            use_gpu: Input value used by this callable.

            resources: Input value used by this callable.

            scheduling_strategy: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

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
        return num_gpus, remote_options

    @staticmethod
    def _build_remote_invoker(
        *,
        ray: Any,
        fn: Callable[..., Any],
        call_kwargs: dict[str, Any],
        remote_options: dict[str, Any],
    ) -> Any:
        """Execute build remote invoker.



        Args:

            ray: Input value used by this callable.

            fn: Input value used by this callable.

            call_kwargs: Input value used by this callable.

            remote_options: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        def _invoke(shared, *call_args):
            """Execute invoke.



            Args:

                shared: Input value used by this callable.

                *call_args: Additional positional arguments.



            Returns:

                Return value produced by the callable.

            """

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
        return remote_fn

    def _resolve_max_inflight(
        self,
        *,
        ray: Any,
        total_tasks: int,
        num_gpus: float,
        max_pending_override: Any,
    ) -> int:
        """Execute resolve max inflight.



        Args:

            ray: Input value used by this callable.

            total_tasks: Input value used by this callable.

            num_gpus: Input value used by this callable.

            max_pending_override: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

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
                return max(1, int(max_pending_override))
            except (TypeError, ValueError):
                pass
        return min(max_inflight_default, max(resource_limit, 1))

    def _map_unbatched(
        self,
        *,
        ray: Any,
        remote_fn: Any,
        args_list: list[Args],
        total_tasks: int,
        max_inflight: int,
        shared_ref: Any,
        desc: str | None,
    ) -> list[Any]:
        """Execute map unbatched.



        Args:

            ray: Input value used by this callable.

            remote_fn: Input value used by this callable.

            args_list: Input value used by this callable.

            total_tasks: Input value used by this callable.

            max_inflight: Input value used by this callable.

            shared_ref: Input value used by this callable.

            desc: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        results: list[Any] = [None] * total_tasks
        pending: dict[Any, int] = {}
        idx = 0
        pbar_iter = iter(
            progress_bar(range(total_tasks), total=total_tasks, desc=desc, enabled=bool(desc))
        )
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
        ray = self._ray or _require_ray()

        def _invoke(shared, batch_args):
            """Execute invoke.



            Args:

                shared: Input value used by this callable.

                batch_args: Input value used by this callable.



            Returns:

                Return value produced by the callable.

            """

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
        """Execute submit.



        Args:

            fn: Input value used by this callable.

            *args: Additional positional arguments.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if self._exec:
            return self._exec.submit(fn, *args)
        ray = self._ray or _require_ray()
        remote_fn = ray.remote(fn)
        return remote_fn.remote(*args)

    def shutdown(self):
        """Execute shutdown."""

        pass

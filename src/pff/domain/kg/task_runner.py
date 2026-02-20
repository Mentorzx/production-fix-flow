"""Strategy pattern for task execution (Ray/Dask/Thread)."""

import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

try:
    import importlib.util

    HAS_RAY = importlib.util.find_spec("ray") is not None
except ImportError:
    HAS_RAY = False

from pff.shared import ConcurrencyManager
from pff.shared.core.logging import logger
from pff.shared.system.probe import get_safe_cpu_count, get_system_ram_gb


class TaskRunner(ABC):
    """Represent TaskRunner."""

    @abstractmethod
    async def execute(self, func: Callable, args: list, desc: str) -> list[Any]:
        """Execute execute.



        Args:

            func: Input value used by this callable.

            args: Input value used by this callable.

            desc: Input value used by this callable.

        """

        pass


class RayRunner(TaskRunner):
    """Represent RayRunner.



    Notes:

        Encapsulates behavior while preserving architecture boundaries.

    """

    async def execute(self, func: Callable, args: list, desc: str) -> list[Any]:
        """Execute execute.



        Args:

            func: Input value used by this callable.

            args: Input value used by this callable.

            desc: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        cm = ConcurrencyManager()
        return await cm.execute(func, args, task_type="ray", desc=desc)  # type: ignore[return-value]


class DaskRunner(TaskRunner):
    """Represent DaskRunner.



    Notes:

        Encapsulates behavior while preserving architecture boundaries.

    """

    def __init__(self, config: dict):
        """Execute init.



        Args:

            config: Input value used by this callable.

        """

        self.config = config

    async def execute(self, func: Callable, args: list, desc: str) -> list[Any]:
        """Execute execute.



        Args:

            func: Input value used by this callable.

            args: Input value used by this callable.

            desc: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        cm = ConcurrencyManager()

        _, available_gb = get_system_ram_gb()
        safe_workers = max(1, int(available_gb / 2.0))

        n_workers = min(safe_workers, self.config.get("n_workers", 4))

        backend_kwargs = {
            "n_workers": n_workers,
            "threads_per_worker": 1,
            "memory_limit": "2GB",
            "processes": True,
            "silence_logs": 30,
        }
        logger.info(f"DaskRunner: {n_workers} workers configurados")
        return await cm.execute(  # type: ignore[return-value]
            func, args, task_type="dask", backend_kwargs=backend_kwargs, desc=desc
        )


class ThreadRunner(TaskRunner):
    """Represent ThreadRunner.



    Notes:

        Encapsulates behavior while preserving architecture boundaries.

    """

    async def execute(self, func: Callable, args: list, desc: str) -> list[Any]:
        """Execute execute.



        Args:

            func: Input value used by this callable.

            args: Input value used by this callable.

            desc: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        cm = ConcurrencyManager()
        backend_kwargs = {"max_workers": min(2, get_safe_cpu_count(logical=True))}
        return await cm.execute(  # type: ignore[return-value]
            func, args, task_type="thread", backend_kwargs=backend_kwargs, desc=desc
        )


class SequentialRunner(TaskRunner):
    """Represent SequentialRunner.



    Notes:

        Encapsulates behavior while preserving architecture boundaries.

    """

    async def execute(self, func: Callable, args: list, desc: str) -> list[Any]:
        """Execute execute.



        Args:

            func: Input value used by this callable.

            args: Input value used by this callable.

            desc: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        cm = ConcurrencyManager()
        return await cm.execute(func, args, task_type="sequential", desc=desc)  # type: ignore[return-value]


class TaskRunnerFactory:
    """Represent TaskRunnerFactory."""

    @staticmethod
    def get_runner(config: dict | None = None) -> TaskRunner:
        """Execute get runner.



        Args:

            config: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        config = config or {}

        if sys.platform != "win32" and HAS_RAY:
            return RayRunner()

        return DaskRunner(config)

    @staticmethod
    def get_specific_runner(runner_type: str, config: dict | None = None) -> TaskRunner:
        """Execute get specific runner.



        Args:

            runner_type: Input value used by this callable.

            config: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        config = config or {}
        if runner_type == "ray" and HAS_RAY:
            return RayRunner()
        elif runner_type == "dask":
            return DaskRunner(config)
        elif runner_type == "thread":
            return ThreadRunner()
        else:
            return SequentialRunner()

    @staticmethod
    async def execute_with_fallback(
        backends: list[str],
        func: Callable,
        args: list,
        *,
        desc: str,
        config_by_backend: dict[str, dict] | None = None,
    ) -> list[Any]:
        """Execute execute with fallback.



        Args:

            backends: Input value used by this callable.

            func: Input value used by this callable.

            args: Input value used by this callable.

            desc: Input value used by this callable.

            config_by_backend: Optional input value.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if not backends:
            raise ValueError("At least one backend must be provided")
        if len(backends) != 1:
            raise ValueError(
                "Fallback execution is disabled: provide exactly one backend to execute."
            )

        config_by_backend = config_by_backend or {}
        task_type = backends[0]
        logger.info(f"Executando backend selecionado: {task_type}")
        runner = TaskRunnerFactory.get_specific_runner(
            task_type, config=config_by_backend.get(task_type)
        )
        return await runner.execute(func, args, desc=desc)

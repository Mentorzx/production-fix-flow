"""Strategy pattern for task execution (Ray/Dask/Thread)."""

import os
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


class TaskRunner(ABC):
    @abstractmethod
    async def execute(self, func: Callable, args: list, desc: str) -> list[Any]:
        pass


class RayRunner(TaskRunner):
    async def execute(self, func: Callable, args: list, desc: str) -> list[Any]:
        cm = ConcurrencyManager()
        return await cm.execute(func, args, task_type="ray", desc=desc)


class DaskRunner(TaskRunner):
    def __init__(self, config: dict):
        self.config = config

    async def execute(self, func: Callable, args: list, desc: str) -> list[Any]:
        cm = ConcurrencyManager()

        import psutil

        mem_gb = psutil.virtual_memory().available / (1024**3)
        safe_workers = max(1, int(mem_gb / 2.0))

        n_workers = min(safe_workers, self.config.get("n_workers", 4))

        backend_kwargs = {
            "n_workers": n_workers,
            "threads_per_worker": 1,
            "memory_limit": "2GB",
            "processes": True,
            "silence_logs": 30,
        }
        logger.info(f"DaskRunner: {n_workers} workers configurados")
        return await cm.execute(
            func, args, task_type="dask", backend_kwargs=backend_kwargs, desc=desc
        )


class ThreadRunner(TaskRunner):
    async def execute(self, func: Callable, args: list, desc: str) -> list[Any]:
        cm = ConcurrencyManager()
        backend_kwargs = {"max_workers": min(2, os.cpu_count() or 2)}
        return await cm.execute(
            func, args, task_type="thread", backend_kwargs=backend_kwargs, desc=desc
        )


class SequentialRunner(TaskRunner):
    async def execute(self, func: Callable, args: list, desc: str) -> list[Any]:
        cm = ConcurrencyManager()
        return await cm.execute(func, args, task_type="sequential", desc=desc)


class TaskRunnerFactory:
    @staticmethod
    def get_runner(config: dict | None = None) -> TaskRunner:
        config = config or {}

        if sys.platform != "win32" and HAS_RAY:
            return RayRunner()

        return DaskRunner(config)

    @staticmethod
    def get_specific_runner(runner_type: str, config: dict | None = None) -> TaskRunner:
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
        config_by_backend = config_by_backend or {}
        for idx, task_type in enumerate(backends):
            try:
                logger.info(f" Tentando executar com backend: {task_type}")
                runner = TaskRunnerFactory.get_specific_runner(
                    task_type, config=config_by_backend.get(task_type)
                )
                return await runner.execute(func, args, desc=desc)
            except Exception as exc:
                logger.error(f"Backend {task_type} failed: {exc}")
                if idx == len(backends) - 1:
                    logger.error("All backends failed!")
                    raise
                logger.info("Tentando proximo backend...")
        return []

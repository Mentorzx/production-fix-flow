"""Concurrency module for parallel execution strategies.

This module provides various executors and strategies for concurrent execution,
including thread-based, process-based, and distributed computing backends.
"""

from __future__ import annotations

# Executors
from .executors import (
    DaskExecutor,
    JoblibExecutor,
    ProcessExecutor,
    RayExecutor,
    ThreadExecutor,
)
from .hardware import GPUInfo, HardwareManager
from .manager import (
    ConcurrencyManager,
    DurableRayTrainer,
    ExecutorFactory,
    get_durable_trainer,
)
from .protocols import Args, BaseExecutor
from .strategies import (
    CpuMultiprocessingStrategy,
    DaskRayCompat,
    ExecutionStrategy,
    GpuCudfStrategy,
    IoAsyncioStrategy,
    IoThreadingStrategy,
)
from .utils import (
    _format_time,
    _require_psutil,
    get_lock,
    GlobalLock,
    progress_bar,
)

__all__ = [
    # Executors
    "BaseExecutor",
    "ThreadExecutor",
    "ProcessExecutor",
    "DaskExecutor",
    "RayExecutor",
    "JoblibExecutor",
    # Manager & Factory
    "ConcurrencyManager",
    "ExecutorFactory",
    "DurableRayTrainer",
    "get_durable_trainer",
    # Hardware
    "HardwareManager",
    "GPUInfo",
    # Strategies
    "ExecutionStrategy",
    "CpuMultiprocessingStrategy",
    "IoThreadingStrategy",
    "IoAsyncioStrategy",
    "GpuCudfStrategy",
    "DaskRayCompat",
    # Utilities
    "Args",
    "GlobalLock",
    "get_lock",
    "progress_bar",
    "_format_time",
    # Private utilities (for backward compatibility with tests)
    "_require_psutil",
]

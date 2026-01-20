"""
Unified Resource Manager - Hardware detection + adaptive resource allocation.

This module combines hardware detection and adaptive resource management to provide
optimal resource allocation with safety margins.

Key features:
- Hardware detection (RAM, CPU, GPU)
- Adaptive resource limits (90% memory, 90% CPU by default)
- Manifest-based configuration (resource_usage: 90%)
- OS-specific optimizations (Linux COW, Windows spawn)
- OOM prevention with runtime monitoring

Author: PFF Team
Version: 2.0.0 (2025-10-22) - Unified hardware_detector + adaptive_resources
"""

import multiprocessing as mp
import os
import platform
import threading
from dataclasses import dataclass
from typing import Any

import psutil

from pff.shared.core.logging import logger
from pff.shared.system.cuda import is_cuda_available


@dataclass
class HardwareProfile:
    """Hardware profile with detected system resources."""

    total_ram_gb: float
    available_ram_gb: float
    cpu_cores: int
    cpu_threads: int
    has_gpu: bool
    gpu_memory_gb: float | None
    is_wsl: bool
    platform: str
    profile_name: str


class HardwareDetector:
    """Detect hardware and provide system profile."""

    @staticmethod
    def detect() -> HardwareProfile:
        """
        Detect current hardware configuration.

        Returns:
            HardwareProfile: Detected hardware specifications.
        """
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024**3)
        available_ram_gb = mem.available / (1024**3)

        cpu_cores = psutil.cpu_count(logical=False) or 4
        cpu_threads = psutil.cpu_count(logical=True) or 8

        has_gpu, gpu_memory_gb = HardwareDetector._detect_gpu()

        is_wsl = (
            "microsoft" in platform.uname().release.lower()
            or "wsl" in platform.uname().release.lower()
        )

        profile_name = HardwareDetector._classify_machine(total_ram_gb, has_gpu)

        return HardwareProfile(
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            has_gpu=has_gpu,
            gpu_memory_gb=gpu_memory_gb,
            is_wsl=is_wsl,
            platform=platform.system(),
            profile_name=profile_name,
        )

    @staticmethod
    def _detect_gpu() -> tuple[bool, float | None]:
        """
        Detect NVIDIA GPU and its memory.

        Returns:
            Tuple of (has_gpu, gpu_memory_gb).
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_gb = mem_info.total / (1024**3)
            pynvml.nvmlShutdown()
            return True, gpu_memory_gb
        except Exception:
            return False, None

    @staticmethod
    def _classify_machine(total_ram_gb: float, has_gpu: bool) -> str:
        """
        Classify machine into low_spec, mid_spec, or high_spec.

        Args:
            total_ram_gb: Total RAM in GB
            has_gpu: Whether a GPU is present

        Returns:
            Profile name: "low_spec", "mid_spec", or "high_spec"
        """
        if total_ram_gb < 10:
            return "low_spec"
        elif total_ram_gb < 20 and not has_gpu:
            return "mid_spec"
        else:
            return "high_spec"


def recommended_numba_threads() -> int:
    """Return recommended Numba thread count (physical cores)."""

    return 10


_numba_config_lock = threading.Lock()
_numba_configured = False


def configure_numba_threads() -> int:
    global _numba_configured
    if _numba_configured:
        return int(os.environ.get("NUMBA_NUM_THREADS", os.cpu_count() or 1))

    with _numba_config_lock:
        if _numba_configured:
            return int(os.environ.get("NUMBA_NUM_THREADS", os.cpu_count() or 1))

        env_threads = os.environ.get("NUMBA_NUM_THREADS")
        if env_threads:
            try:
                threads = int(env_threads)

                _numba_configured = True
                return threads
            except ValueError:
                threads = recommended_numba_threads()
        else:
            threads = recommended_numba_threads()
            os.environ["NUMBA_NUM_THREADS"] = str(threads)
        if threads == 12:
            import sys

            print(
                f"WARNING: Downgrading Numba threads from 12 to 10 to prevent RuntimeError. Env was: {env_threads}",
                file=sys.stderr,
            )
            threads = 10
            os.environ["NUMBA_NUM_THREADS"] = "10"

        try:
            import numba

            try:
                # Check current threads to avoid unnecessary setting which might raise RuntimeError
                current_threads = getattr(numba, "get_num_threads", lambda: -1)()
                if current_threads != threads:
                    setter = getattr(numba, "set_num_threads", None)
                    if setter:
                        setter(threads)
            except Exception:
                pass
        except ImportError:
            pass

        _numba_configured = True
        return threads


@dataclass
class ResourceLimits:
    """System resource limits with safety margins."""

    total_memory: int
    available_memory: int
    safe_memory_limit: int
    per_worker_memory: int

    total_cpus: int
    available_cpus: int
    optimal_workers: int

    max_batch_size: int
    max_pending_futures: int

    platform: str
    has_cow: bool
    profile_name: str

    cpu_usage_percent: float
    memory_usage_percent: float

    def __str__(self) -> str:
        return (
            f"ResourceLimits(\n"
            f"  Profile: {self.profile_name}\n"
            f"  Memory: {self.available_memory / 1024**3:.1f} GB available, "
            f"{self.safe_memory_limit / 1024**3:.1f} GB safe limit ({self.memory_usage_percent:.0f}%)\n"
            f"  CPU: {self.optimal_workers}/{self.total_cpus} workers ({self.cpu_usage_percent:.0f}% usage)\n"
            f"  Per-worker: {self.per_worker_memory / 1024**2:.0f} MB\n"
            f"  Batch: {self.max_batch_size} tasks, {self.max_pending_futures} max pending\n"
            f"  Platform: {self.platform}, COW: {self.has_cow}\n"
            f")"
        )


class ResourceManager:
    """
    Unified resource manager combining hardware detection and adaptive allocation.

    Features:
    - Hardware detection (RAM, CPU, GPU)
    - Configurable resource usage (default: 90% memory, 90% CPU)
    - Manifest-based configuration (resource_usage: 90)
    - OS-specific optimizations (COW detection)
    - OOM prevention with safety margins

    Example:
        >>> manager = ResourceManager(cpu_usage_percent=90, memory_usage_percent=90)
        >>> limits = manager.calculate_limits(
        ...     task_count=128319,
        ...     estimated_task_size=5000
        ... )
        >>> print(f"Use {limits.optimal_workers} workers (10% reserved for OS)")
    """

    def __init__(self, cpu_usage_percent: float = 90.0, memory_usage_percent: float = 90.0):
        """
        Initialize resource manager.

        Args:
            cpu_usage_percent: % of CPUs to use (default: 90%, leaves 10% for OS)
            memory_usage_percent: % of memory to use (default: 90%, 10% margin)
        """
        self.cpu_usage_percent = cpu_usage_percent
        self.memory_usage_percent = memory_usage_percent

        self.hardware = HardwareDetector.detect()

        self._platform = platform.system()
        self._has_cow = self._detect_cow_support()

    def _detect_cow_support(self) -> bool:
        """
        Detect if OS supports copy-on-write (COW) for fork().

        Linux: Yes (fork with COW)
        Windows: No (spawn always copies)
        macOS: Yes (fork with COW, but spawn default in 3.8+)
        """
        if self._platform == "Linux":
            return True
        elif self._platform == "Darwin":
            return mp.get_start_method() == "fork"
        else:
            return False

    def get_current_resources(self) -> dict[str, Any]:
        """Get current system resource usage."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=False)

        return {
            "memory_total_gb": memory.total / 1024**3,
            "memory_available_gb": memory.available / 1024**3,
            "memory_used_gb": memory.used / 1024**3,
            "memory_percent": memory.percent,
            "cpu_count": self.hardware.cpu_threads,
            "cpu_percent": cpu_percent,
            "profile": self.hardware.profile_name,
        }

    def calculate_limits(
        self,
        task_count: int,
        estimated_task_size: int,
        shared_data_size: int = 0,
        min_workers: int = 1,
        max_workers: int | None = None,
    ) -> ResourceLimits:
        """
        Calculate optimal resource limits based on current system state.

        Args:
            task_count: Number of tasks to process
            estimated_task_size: Estimated memory per task (bytes)
            shared_data_size: Size of data shared across workers (bytes)
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers (None = use cpu_usage_percent)

        Returns:
            ResourceLimits object with calculated safe limits
        """
        memory = psutil.virtual_memory()
        total_cpus = self.hardware.cpu_threads

        available_memory = memory.available
        safe_memory_limit = int(available_memory * (self.memory_usage_percent / 100))

        if max_workers is None:
            max_workers_from_cpu = int(total_cpus * (self.cpu_usage_percent / 100))
            max_workers_from_cpu = max(min_workers, max_workers_from_cpu)
        else:
            max_workers_from_cpu = max(min_workers, max_workers)

        optimal_workers = min(max_workers_from_cpu, total_cpus - 1)
        optimal_workers = max(min_workers, optimal_workers)

        if self._has_cow:
            per_worker_overhead = 50 * 1024 * 1024
            shared_data_per_worker = 0
        else:
            per_worker_overhead = 50 * 1024 * 1024
            shared_data_per_worker = shared_data_size

        per_worker_memory = per_worker_overhead + shared_data_per_worker

        workers_base_memory = optimal_workers * per_worker_memory

        memory_for_tasks = safe_memory_limit - workers_base_memory

        if memory_for_tasks < 0:
            logger.warning(
                f"Insufficient memory! Available: {available_memory / 1024**3:.1f} GB, "
                f"Need: {workers_base_memory / 1024**3:.1f} GB for {optimal_workers} workers"
            )
            optimal_workers = max(1, int(safe_memory_limit / per_worker_memory))
            workers_base_memory = optimal_workers * per_worker_memory
            memory_for_tasks = safe_memory_limit - workers_base_memory

        if estimated_task_size > 0:
            max_concurrent_tasks = int(memory_for_tasks / estimated_task_size)
        else:
            max_concurrent_tasks = int(memory_for_tasks / (10 * 1024 * 1024))

        max_concurrent_tasks = max(100, max_concurrent_tasks)
        max_concurrent_tasks = min(10000, max_concurrent_tasks)

        ideal_batch_multiplier = 50
        max_batch_size = optimal_workers * ideal_batch_multiplier
        max_batch_size = min(max_batch_size, max_concurrent_tasks // 2)
        max_batch_size = max(100, max_batch_size)

        max_pending_futures = optimal_workers * 10
        max_pending_futures = min(max_pending_futures, max_concurrent_tasks)
        max_pending_futures = max(100, max_pending_futures)

        limits = ResourceLimits(
            total_memory=memory.total,
            available_memory=available_memory,
            safe_memory_limit=safe_memory_limit,
            per_worker_memory=per_worker_memory,
            total_cpus=total_cpus,
            available_cpus=total_cpus,
            optimal_workers=optimal_workers,
            max_batch_size=max_batch_size,
            max_pending_futures=max_pending_futures,
            platform=self._platform,
            has_cow=self._has_cow,
            profile_name=self.hardware.profile_name,
            cpu_usage_percent=self.cpu_usage_percent,
            memory_usage_percent=self.memory_usage_percent,
        )

        logger.debug(f"Calculated adaptive resource limits:\n{limits}")

        return limits

    def should_throttle(self, threshold_percent: float = 85.0) -> bool:
        """
        Check if we should throttle task submission due to high memory usage.

        Args:
            threshold_percent: Throttle threshold (default: 85%)

        Returns:
            True if should throttle (pause task submission)
        """
        memory = psutil.virtual_memory()

        if memory.percent > threshold_percent:
            logger.warning(
                f" Throttling task submission: memory at {memory.percent:.1f}% "
                f"(threshold: {threshold_percent:.0f}%)"
            )
            return True

        return False


_global_manager: ResourceManager | None = None
_global_manager_lock = threading.Lock()
_MULTIPROC_AVAILABLE: bool | None = None


def get_resource_manager(
    cpu_usage_percent: float = 90.0, memory_usage_percent: float = 90.0
) -> ResourceManager:
    """
    Get global resource manager instance.

    Args:
        cpu_usage_percent: % of CPUs to use (default: 90%)
        memory_usage_percent: % of memory to use (default: 90%)

    Returns:
        ResourceManager singleton instance
    """
    global _global_manager
    if _global_manager is None:
        with _global_manager_lock:
            if _global_manager is None:
                _global_manager = ResourceManager(
                    cpu_usage_percent=cpu_usage_percent,
                    memory_usage_percent=memory_usage_percent,
                )
    return _global_manager


def detect_hardware() -> HardwareProfile:
    """Quick helper to detect hardware."""
    return HardwareDetector.detect()


def calculate_optimal_resources(
    task_count: int,
    estimated_task_size: int = 5000,
    shared_data_size: int = 0,
    cpu_usage_percent: float = 90.0,
    memory_usage_percent: float = 90.0,
) -> tuple[int, int, int]:
    """
    Quick helper to calculate optimal workers, batch size, and max pending.

    Args:
        task_count: Number of tasks
        estimated_task_size: Bytes per task
        shared_data_size: Bytes of shared data
        cpu_usage_percent: % of CPUs to use (default: 90%)
        memory_usage_percent: % of memory to use (default: 90%)

    Returns:
        (optimal_workers, max_batch_size, max_pending_futures)
    """
    manager = get_resource_manager(cpu_usage_percent, memory_usage_percent)
    limits = manager.calculate_limits(
        task_count=task_count,
        estimated_task_size=estimated_task_size,
        shared_data_size=shared_data_size,
    )
    return limits.optimal_workers, limits.max_batch_size, limits.max_pending_futures


def get_memory_safe_workers(chunk_size: int = 1000) -> int:
    """
    Estimate a safe worker count based on available memory and CPU cores.

    Args:
        chunk_size: Approximate work size per worker (used to scale memory).

    Returns:
        Recommended number of worker processes.
    """
    global _MULTIPROC_AVAILABLE
    if _MULTIPROC_AVAILABLE is None:
        try:
            ctx = mp.get_context()
            ctx.Lock()
            _MULTIPROC_AVAILABLE = True
        except Exception:
            _MULTIPROC_AVAILABLE = False
    if not _MULTIPROC_AVAILABLE:
        return 0

    try:
        available_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        available_gb = 1.0

    memory_per_worker_gb = 0.5 * (chunk_size / 1000)
    safe_workers = int((available_gb * 0.7) / max(memory_per_worker_gb, 0.1))
    cpu_count = os.cpu_count() or 4
    return max(1, min(safe_workers, cpu_count))


def get_auto_dataloader_workers(
    dataset_size: int,
    batch_size: int,
    *,
    min_workers: int = 0,
    max_workers: int = 8,
    vram_threshold_gb: float | None = None,
) -> int:
    """
    Estimate DataLoader workers using dataset size, batch size, and VRAM headroom.

    Args:
        dataset_size: Number of samples in the dataset.
        batch_size: Training batch size.
        min_workers: Minimum workers when auto-tuning (0 allows disabling).
        max_workers: Upper bound on workers.
        vram_threshold_gb: If total VRAM is below this, cap to min_workers.

    Returns:
        Recommended worker count (0 disables multiprocessing).
    """
    if dataset_size <= max(batch_size * 4, 1):
        return 0

    base_workers = get_memory_safe_workers(chunk_size=batch_size)
    if base_workers <= 0:
        return 0

    cpu_count = os.cpu_count() or 4
    workers = min(base_workers, cpu_count, max_workers)
    workers = max(min_workers, workers)

    if vram_threshold_gb is not None:
        try:
            info = get_cuda_memory_info()
            if info is not None:
                total_gb = info["total_bytes"] / (1024**3)
                if total_gb < float(vram_threshold_gb):
                    return max(min_workers, 0)
        except Exception:
            pass

    return workers


def get_cuda_memory_info() -> dict[str, float] | None:
    """Return CUDA memory usage information when available.

    Returns:
        dict with keys: free_bytes, total_bytes, used_bytes, free_ratio
        or None if CUDA is unavailable or an error occurs.
    """
    if not is_cuda_available():
        return None

    try:
        import torch
    except Exception:
        return None

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception as exc:
        logger.debug(f"Falha ao ler memoria CUDA via torch: {exc}")
        return None

    if total_bytes <= 0:
        return None

    used_bytes = total_bytes - free_bytes
    free_ratio = free_bytes / total_bytes
    return {
        "free_bytes": float(free_bytes),
        "total_bytes": float(total_bytes),
        "used_bytes": float(used_bytes),
        "free_ratio": float(free_ratio),
    }


def get_cuda_free_ratio(default: float | None = None) -> float | None:
    """Convenience wrapper to fetch CUDA free memory ratio."""
    info = get_cuda_memory_info()
    if info is None:
        return default
    return float(info.get("free_ratio", default if default is not None else 0.0))

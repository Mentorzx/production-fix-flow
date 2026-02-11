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
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import psutil  # type: ignore[import-untyped]

from pff.shared.core.config_loader import load_config
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger
from pff.shared.core.config import PERFORMANCE_CONFIG_PATH
from pff.shared.system.cuda import is_cuda_available


from pff.shared.system.probe import (
    get_gpu_total_memory_gb,
    get_safe_cpu_count,
    get_system_ram_gb,
)


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


StorageType = Literal["nvme", "ssd", "hdd", "wsl", "unknown"]


@dataclass(frozen=True)
class HardwareClassificationThresholds:
    """Thresholds used to classify a machine profile.

    These thresholds must be centralized to avoid drift.
    """

    mid_min_ram_gb: float
    high_min_ram_gb: float
    high_requires_gpu: bool


@dataclass(frozen=True)
class ResourceManagerTuning:
    """Tuning knobs for ResourceManager (kept config-driven)."""

    per_worker_overhead_mb: int
    default_task_size_bytes: int
    min_concurrent_tasks: int
    max_concurrent_tasks: int
    ideal_batch_multiplier: int
    max_batch_fraction_of_concurrency: float
    pending_futures_multiplier: int
    telemetry_ttl_ms: int


@lru_cache(maxsize=1)
def _load_resource_manager_tuning() -> ResourceManagerTuning:
    defaults = ResourceManagerTuning(
        per_worker_overhead_mb=50,
        default_task_size_bytes=10 * 1024 * 1024,
        min_concurrent_tasks=100,
        max_concurrent_tasks=10_000,
        ideal_batch_multiplier=50,
        max_batch_fraction_of_concurrency=0.5,
        pending_futures_multiplier=10,
        telemetry_ttl_ms=250,
    )

    try:
        raw = load_config(PERFORMANCE_CONFIG_PATH)
        if not raw:
            return defaults
        rm_cfg = raw.get("performance", {}).get("resource_manager", {})
        if not isinstance(rm_cfg, dict):
            return defaults
    except Exception:
        return defaults

    def _get_int(key: str, default: int) -> int:
        value = rm_cfg.get(key, default)
        try:
            return int(value)
        except Exception:
            return default

    def _get_float(key: str, default: float) -> float:
        value = rm_cfg.get(key, default)
        try:
            return float(value)
        except Exception:
            return default

    return ResourceManagerTuning(
        per_worker_overhead_mb=_get_int(
            "per_worker_overhead_mb", defaults.per_worker_overhead_mb
        ),
        default_task_size_bytes=_get_int(
            "default_task_size_bytes", defaults.default_task_size_bytes
        ),
        min_concurrent_tasks=_get_int(
            "min_concurrent_tasks", defaults.min_concurrent_tasks
        ),
        max_concurrent_tasks=_get_int(
            "max_concurrent_tasks", defaults.max_concurrent_tasks
        ),
        ideal_batch_multiplier=_get_int(
            "ideal_batch_multiplier", defaults.ideal_batch_multiplier
        ),
        max_batch_fraction_of_concurrency=_get_float(
            "max_batch_fraction_of_concurrency",
            defaults.max_batch_fraction_of_concurrency,
        ),
        pending_futures_multiplier=_get_int(
            "pending_futures_multiplier", defaults.pending_futures_multiplier
        ),
        telemetry_ttl_ms=_get_int("telemetry_ttl_ms", defaults.telemetry_ttl_ms),
    )


@lru_cache(maxsize=1)
def _load_classification_thresholds() -> HardwareClassificationThresholds:
    defaults = HardwareClassificationThresholds(
        mid_min_ram_gb=7.0,
        high_min_ram_gb=24.0,
        high_requires_gpu=True,
    )

    try:
        raw = load_config(PERFORMANCE_CONFIG_PATH)
        if not raw:
            return defaults
        cfg = raw.get("performance", {}).get("hardware_classification", {})
        if not isinstance(cfg, dict):
            return defaults
    except Exception:
        return defaults

    def _get_float(key: str, default: float) -> float:
        value = cfg.get(key, default)
        try:
            return float(value)
        except Exception:
            return default

    return HardwareClassificationThresholds(
        mid_min_ram_gb=_get_float("mid_min_ram_gb", defaults.mid_min_ram_gb),
        high_min_ram_gb=_get_float("high_min_ram_gb", defaults.high_min_ram_gb),
        high_requires_gpu=bool(
            cfg.get("high_requires_gpu", defaults.high_requires_gpu)
        ),
    )


def _detect_storage_type(*, is_wsl: bool) -> StorageType:
    if is_wsl:
        return "wsl"

    env_override = _resolve_storage_override()
    if env_override is not None:
        return env_override

    if platform.system() != "Linux":
        return "unknown"

    rotational_flags = _read_rotational_flags()
    return _classify_storage_from_flags(rotational_flags)


def _resolve_storage_override() -> StorageType | None:
    env_override = os.environ.get("PFF_STORAGE_TYPE")
    if env_override in {"nvme", "ssd", "hdd"}:
        return env_override  # type: ignore[return-value]
    return None


def _read_rotational_flags() -> list[int]:
    try:
        sys_block = "/sys/block"
        if not os.path.isdir(sys_block):
            return []

        rotational_flags: list[int] = []
        for dev in os.listdir(sys_block):
            if dev.startswith("loop") or dev.startswith("ram"):
                continue
            rotational_path = os.path.join(sys_block, dev, "queue", "rotational")
            if os.path.exists(rotational_path):
                raw = FileManager.read(rotational_path)
                try:
                    rotational_flags.append(int(str(raw).strip()))
                except Exception:
                    continue
        return rotational_flags
    except Exception:
        return []


def _classify_storage_from_flags(rotational_flags: list[int]) -> StorageType:
    if not rotational_flags:
        return "unknown"
    if any(v == 1 for v in rotational_flags):
        return "hdd"
    return "ssd"


class HardwareDetector:
    """Detect hardware and provide system profile."""

    @staticmethod
    def detect() -> HardwareProfile:
        """
        Detect current hardware configuration.

        Returns:
            HardwareProfile: Detected hardware specifications.
        """
        total_ram_gb, available_ram_gb = get_system_ram_gb()

        cpu_cores = get_safe_cpu_count(logical=False)
        cpu_threads = get_safe_cpu_count(logical=True)

        has_gpu, gpu_memory_gb = HardwareDetector._detect_gpu()

        release = platform.uname().release.lower()
        is_wsl = "microsoft" in release or "wsl" in release

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
        return get_gpu_total_memory_gb(device_index=0)

    @staticmethod
    def _classify_machine(total_ram_gb: float, has_gpu: bool) -> str:
        """Classify machine into low_spec, mid_spec, or high_spec."""
        thresholds = _load_classification_thresholds()

        if total_ram_gb >= thresholds.high_min_ram_gb and (
            (not thresholds.high_requires_gpu) or has_gpu
        ):
            return "high_spec"

        if total_ram_gb >= thresholds.mid_min_ram_gb:
            return "mid_spec"

        return "low_spec"


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
        >>> f"Use {limits.optimal_workers} workers (10% reserved for OS)"
    """

    def __init__(
        self,
        cpu_usage_percent: float = 90.0,
        memory_usage_percent: float = 90.0,
        *,
        storage_type: StorageType | None = None,
    ):
        """
        Initialize resource manager.

        Args:
            cpu_usage_percent: % of CPUs to use (default: 90%, leaves 10% for OS)
            memory_usage_percent: % of memory to use (default: 90%, 10% margin)
        """
        self.cpu_usage_percent = cpu_usage_percent
        self.memory_usage_percent = memory_usage_percent

        self.hardware = HardwareDetector.detect()

        self._storage_type: StorageType = (
            storage_type
            if storage_type is not None
            else _detect_storage_type(is_wsl=self.hardware.is_wsl)
        )

        self._platform = platform.system()
        self._has_cow = self._detect_cow_support()

        self._resource_tuning = _load_resource_manager_tuning()

        ttl_ms = max(int(self._resource_tuning.telemetry_ttl_ms), 0)
        self._telemetry_ttl_s = ttl_ms / 1000.0
        self._telemetry_lock = threading.Lock()
        self._telemetry_last_ts: float | None = None
        self._telemetry_cache: dict[str, Any] | None = None

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if not (1.0 <= float(self.cpu_usage_percent) <= 100.0):
            raise ValueError(
                f"cpu_usage_percent must be in [1, 100], got {self.cpu_usage_percent}"
            )
        if not (1.0 <= float(self.memory_usage_percent) <= 100.0):
            raise ValueError(
                f"memory_usage_percent must be in [1, 100], got {self.memory_usage_percent}"
            )

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
        if self._telemetry_ttl_s > 0.0:
            now = time.monotonic()
            with self._telemetry_lock:
                if (
                    self._telemetry_cache is not None
                    and self._telemetry_last_ts is not None
                    and (now - self._telemetry_last_ts) < self._telemetry_ttl_s
                ):
                    return dict(self._telemetry_cache)

        memory = psutil.virtual_memory()

        cpu_percent = psutil.cpu_percent(interval=None, percpu=False)

        snapshot = {
            "memory_total_gb": memory.total / 1024**3,
            "memory_available_gb": memory.available / 1024**3,
            "memory_used_gb": memory.used / 1024**3,
            "memory_percent": memory.percent,
            "cpu_count": self.hardware.cpu_threads,
            "cpu_percent": cpu_percent,
            "profile": self.hardware.profile_name,
        }

        if self._telemetry_ttl_s > 0.0:
            now = time.monotonic()
            with self._telemetry_lock:
                self._telemetry_cache = snapshot
                self._telemetry_last_ts = now

        return dict(snapshot)

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
        self._validate_calculate_limits_inputs(
            task_count=task_count,
            estimated_task_size=estimated_task_size,
            shared_data_size=shared_data_size,
            min_workers=min_workers,
            max_workers=max_workers,
        )

        memory = psutil.virtual_memory()
        total_cpus = self.hardware.cpu_threads

        available_memory = memory.available
        safe_memory_limit = int(available_memory * (self.memory_usage_percent / 100))

        max_workers_from_cpu = self._resolve_max_workers_from_cpu(
            min_workers=min_workers,
            max_workers=max_workers,
            total_cpus=total_cpus,
        )

        optimal_workers = min(max_workers_from_cpu, total_cpus - 1)
        optimal_workers = max(min_workers, optimal_workers)

        per_worker_overhead = self._resource_tuning.per_worker_overhead_mb * 1024 * 1024
        shared_data_per_worker = 0 if self._has_cow else shared_data_size

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

        max_concurrent_tasks = self._resolve_max_concurrent_tasks(
            memory_for_tasks=memory_for_tasks,
            estimated_task_size=estimated_task_size,
        )

        max_concurrent_tasks = max(
            self._resource_tuning.min_concurrent_tasks, max_concurrent_tasks
        )
        max_concurrent_tasks = min(
            self._resource_tuning.max_concurrent_tasks, max_concurrent_tasks
        )

        ideal_batch_multiplier = max(self._resource_tuning.ideal_batch_multiplier, 1)
        max_batch_size = optimal_workers * ideal_batch_multiplier

        max_batch_fraction = float(
            self._resource_tuning.max_batch_fraction_of_concurrency
        )
        if max_batch_fraction < 0.0:
            max_batch_fraction = 0.0
        if max_batch_fraction > 1.0:
            max_batch_fraction = 1.0
        max_batch_size = min(
            max_batch_size, int(max_concurrent_tasks * max_batch_fraction)
        )
        max_batch_size = max(self._resource_tuning.min_concurrent_tasks, max_batch_size)

        max_pending_futures = optimal_workers * max(
            self._resource_tuning.pending_futures_multiplier, 1
        )
        max_pending_futures = min(max_pending_futures, max_concurrent_tasks)
        max_pending_futures = max(
            self._resource_tuning.min_concurrent_tasks, max_pending_futures
        )

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

    @staticmethod
    def _validate_calculate_limits_inputs(
        *,
        task_count: int,
        estimated_task_size: int,
        shared_data_size: int,
        min_workers: int,
        max_workers: int | None,
    ) -> None:
        if task_count < 0:
            raise ValueError(f"task_count must be >= 0, got {task_count}")
        if estimated_task_size < 0:
            raise ValueError(
                f"estimated_task_size must be >= 0, got {estimated_task_size}"
            )
        if shared_data_size < 0:
            raise ValueError(f"shared_data_size must be >= 0, got {shared_data_size}")
        if min_workers < 1:
            raise ValueError(f"min_workers must be >= 1, got {min_workers}")
        if max_workers is not None and max_workers < min_workers:
            raise ValueError(
                f"max_workers must be >= min_workers when provided, got max_workers={max_workers} min_workers={min_workers}"
            )

    def _resolve_max_workers_from_cpu(
        self, *, min_workers: int, max_workers: int | None, total_cpus: int
    ) -> int:
        if max_workers is None:
            max_workers_from_cpu = int(total_cpus * (self.cpu_usage_percent / 100))
            return max(min_workers, max_workers_from_cpu)
        return max(min_workers, max_workers)

    def _resolve_max_concurrent_tasks(
        self, *, memory_for_tasks: int, estimated_task_size: int
    ) -> int:
        if estimated_task_size > 0:
            return int(memory_for_tasks / estimated_task_size)
        return int(
            memory_for_tasks / max(self._resource_tuning.default_task_size_bytes, 1)
        )

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


def reset_resource_manager_for_tests() -> None:
    """Reset global ResourceManager singleton (test-only helper)."""
    global _global_manager
    with _global_manager_lock:
        _global_manager = None


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


@dataclass(frozen=True)
class PostgresTuningPolicy:
    """Policy inputs for PostgreSQL tuning."""

    storage_type: StorageType
    work_mem_cap_mb: int


def _fmt_mb(value_mb: int) -> str:
    return f"{max(int(value_mb), 1)}MB"


def _fmt_gb(value_gb: float) -> str:
    mb = int(round(max(value_gb, 0.0) * 1024))
    return _fmt_mb(mb)


def _parse_mem_to_mb(value: str) -> int:
    raw = value.strip().upper()
    if raw.endswith("GB"):
        return int(float(raw[:-2]) * 1024)
    if raw.endswith("MB"):
        return int(float(raw[:-2]))
    return int(float(raw))


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
    cpus = get_safe_cpu_count(logical=True)
    return max(1, min(safe_workers, cpus))


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

    workers = min(base_workers, max_workers)
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
        logger.debug(f"Failed to read CUDA memory via torch: {exc}")
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

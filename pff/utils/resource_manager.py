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

Author: Claude Code
Version: 2.0.0 (2025-10-22) - Unified hardware_detector + adaptive_resources
"""

import os
import platform
import psutil
import multiprocessing as mp
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from loguru import logger


# =============================================================================
# Hardware Detection (from hardware_detector.py)
# =============================================================================

@dataclass
class HardwareProfile:
    """Hardware profile with detected system resources."""

    total_ram_gb: float
    available_ram_gb: float
    cpu_cores: int
    cpu_threads: int
    has_gpu: bool
    gpu_memory_gb: Optional[float]
    is_wsl: bool
    platform: str
    profile_name: str  # "low_spec", "mid_spec", or "high_spec"


class HardwareDetector:
    """Detect hardware and provide system profile."""

    @staticmethod
    def detect() -> HardwareProfile:
        """
        Detect current hardware configuration.

        Returns:
            HardwareProfile: Detected hardware specifications.
        """
        # RAM detection
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024 ** 3)
        available_ram_gb = mem.available / (1024 ** 3)

        # CPU detection
        cpu_cores = psutil.cpu_count(logical=False) or 4  # Physical cores
        cpu_threads = psutil.cpu_count(logical=True) or 8  # Logical threads

        # GPU detection
        has_gpu, gpu_memory_gb = HardwareDetector._detect_gpu()

        # WSL detection
        is_wsl = "microsoft" in platform.uname().release.lower() or \
                 "wsl" in platform.uname().release.lower()

        # Machine profile classification
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
            profile_name=profile_name
        )

    @staticmethod
    def _detect_gpu() -> tuple[bool, Optional[float]]:
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
            gpu_memory_gb = mem_info.total / (1024 ** 3)
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


# =============================================================================
# Adaptive Resource Management (from adaptive_resources.py)
# =============================================================================

@dataclass
class ResourceLimits:
    """System resource limits with safety margins."""

    # Memory limits (bytes)
    total_memory: int
    available_memory: int
    safe_memory_limit: int  # memory_usage% of available
    per_worker_memory: int

    # CPU limits
    total_cpus: int
    available_cpus: int
    optimal_workers: int

    # Batch limits
    max_batch_size: int
    max_pending_futures: int

    # System info
    platform: str
    has_cow: bool  # Copy-on-write support (Linux fork)
    profile_name: str  # "low_spec", "mid_spec", "high_spec"

    # Configuration
    cpu_usage_percent: float  # % of CPUs to use (default: 90%)
    memory_usage_percent: float  # % of memory to use (default: 90%)

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

    def __init__(
        self,
        cpu_usage_percent: float = 90.0,
        memory_usage_percent: float = 90.0
    ):
        """
        Initialize resource manager.

        Args:
            cpu_usage_percent: % of CPUs to use (default: 90%, leaves 10% for OS)
            memory_usage_percent: % of memory to use (default: 90%, 10% margin)
        """
        self.cpu_usage_percent = cpu_usage_percent
        self.memory_usage_percent = memory_usage_percent

        # Detect hardware
        self.hardware = HardwareDetector.detect()

        # Detect platform-specific features
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
        elif self._platform == "Darwin":  # macOS
            return mp.get_start_method() == "fork"
        else:  # Windows
            return False

    def get_current_resources(self) -> Dict[str, Any]:
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
        max_workers: Optional[int] = None,
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
        # Get current system state
        memory = psutil.virtual_memory()
        total_cpus = self.hardware.cpu_threads

        # Calculate safe memory limit (memory_usage_percent of available)
        available_memory = memory.available
        safe_memory_limit = int(available_memory * (self.memory_usage_percent / 100))

        # Determine max workers based on CPU usage % and user limits
        if max_workers is None:
            # 🔧 FIX: Use cpu_usage_percent to leave margin for OS
            # Example: 12 CPUs × 90% = 10.8 → 10 workers (2 threads free for OS)
            max_workers_from_cpu = int(total_cpus * (self.cpu_usage_percent / 100))
            max_workers_from_cpu = max(min_workers, max_workers_from_cpu)
        else:
            max_workers_from_cpu = max_workers

        optimal_workers = min(max_workers_from_cpu, total_cpus - 1)  # Always leave at least 1 thread
        optimal_workers = max(min_workers, optimal_workers)

        # Estimate memory per worker
        if self._has_cow:
            # With COW, shared data is not copied until modified
            per_worker_overhead = 50 * 1024 * 1024  # 50 MB base overhead
            shared_data_per_worker = 0  # Shared via COW
        else:
            # Without COW (Windows spawn), each worker gets full copy
            per_worker_overhead = 50 * 1024 * 1024  # 50 MB
            shared_data_per_worker = shared_data_size

        per_worker_memory = per_worker_overhead + shared_data_per_worker

        # Calculate how much memory workers will use
        workers_base_memory = optimal_workers * per_worker_memory

        # Calculate remaining memory for task processing
        memory_for_tasks = safe_memory_limit - workers_base_memory

        if memory_for_tasks < 0:
            # Not enough memory for even base workers
            logger.warning(
                f"Insufficient memory! Available: {available_memory / 1024**3:.1f} GB, "
                f"Need: {workers_base_memory / 1024**3:.1f} GB for {optimal_workers} workers"
            )
            # Reduce workers to fit
            optimal_workers = max(1, int(safe_memory_limit / per_worker_memory))
            workers_base_memory = optimal_workers * per_worker_memory
            memory_for_tasks = safe_memory_limit - workers_base_memory

        # Calculate optimal batch size and pending futures
        if estimated_task_size > 0:
            max_concurrent_tasks = int(memory_for_tasks / estimated_task_size)
        else:
            # Conservative default: assume 10 MB per task
            max_concurrent_tasks = int(memory_for_tasks / (10 * 1024 * 1024))

        # Limit to reasonable bounds
        max_concurrent_tasks = max(100, max_concurrent_tasks)  # At least 100
        max_concurrent_tasks = min(10000, max_concurrent_tasks)  # At most 10K

        # Batch size: chunk tasks into manageable batches
        ideal_batch_multiplier = 50
        max_batch_size = optimal_workers * ideal_batch_multiplier
        max_batch_size = min(max_batch_size, max_concurrent_tasks // 2)
        max_batch_size = max(100, max_batch_size)

        # Max pending futures: keep 10x workers worth of tasks pending
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

        logger.info(f"📊 Calculated adaptive resource limits:\n{limits}")

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
                f"⚠️ Throttling task submission: memory at {memory.percent:.1f}% "
                f"(threshold: {threshold_percent:.0f}%)"
            )
            return True

        return False


# =============================================================================
# Global Singleton
# =============================================================================

_global_manager: Optional[ResourceManager] = None


def get_resource_manager(
    cpu_usage_percent: float = 90.0,
    memory_usage_percent: float = 90.0
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
        _global_manager = ResourceManager(
            cpu_usage_percent=cpu_usage_percent,
            memory_usage_percent=memory_usage_percent
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
) -> Tuple[int, int, int]:
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

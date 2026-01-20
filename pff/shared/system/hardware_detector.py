"""
Hardware Detection Module - Auto-detect system resources for optimal PostgreSQL configuration.

This module automatically detects available hardware resources (RAM, CPU cores, GPU)
and provides optimal PostgreSQL configuration parameters based on the detected hardware.

Supports multiple machines with different specs:
- Machine 1 (low_spec): 8GB RAM, 4-8 CPU cores (WSL dev)
- Machine 2 (mid_spec): 16GB RAM, 12 CPU cores (current: Fedora WSL)
- Machine 3 (high_spec): 32GB RAM, 8-16 CPU cores, RTX 3070 Ti (production)

Author: PFF Team
Version: 1.0.0
"""

import platform
from dataclasses import dataclass

import psutil

from pff.shared.core.logging import logger


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
    machine_name: str

    @property
    def profile_name(self) -> str:
        """Alias for machine_name for backward compatibility."""
        return self.machine_name


class HardwareDetector:
    """Detect hardware and provide optimal PostgreSQL configuration."""

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

        cpu_cores = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)

        has_gpu, gpu_memory_gb = HardwareDetector._detect_gpu()

        is_wsl = (
            "microsoft" in platform.uname().release.lower()
            or "wsl" in platform.uname().release.lower()
        )

        machine_name = HardwareDetector._classify_machine(total_ram_gb, has_gpu)

        return HardwareProfile(
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            has_gpu=has_gpu,
            gpu_memory_gb=gpu_memory_gb,
            is_wsl=is_wsl,
            platform=platform.system(),
            machine_name=machine_name,
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
        Classify machine as low_spec, mid_spec, or high_spec.

        Args:
            total_ram_gb: Total RAM in GB.
            has_gpu: Whether GPU is available.

        Returns:
            Machine classification: "low_spec", "mid_spec", or "high_spec".
        """
        if total_ram_gb >= 24 and has_gpu:
            return "high_spec"
        elif total_ram_gb >= 12 or (total_ram_gb >= 7 and total_ram_gb < 12):
            return "mid_spec"
        else:
            return "low_spec"


@dataclass
class PostgreSQLConfig:
    """PostgreSQL configuration parameters optimized for detected hardware."""

    shared_buffers: str
    effective_cache_size: str
    work_mem: str
    maintenance_work_mem: str
    max_connections: int
    max_parallel_workers_per_gather: int
    max_parallel_workers: int
    max_worker_processes: int
    wal_buffers: str
    default_statistics_target: int
    random_page_cost: float
    effective_io_concurrency: int
    checkpoint_completion_target: float
    min_wal_size: str
    max_wal_size: str

    def to_dict(self) -> dict[str, str | int | float]:
        """Convert config to dictionary for easy access."""
        return {
            "shared_buffers": self.shared_buffers,
            "effective_cache_size": self.effective_cache_size,
            "work_mem": self.work_mem,
            "maintenance_work_mem": self.maintenance_work_mem,
            "max_connections": self.max_connections,
            "max_parallel_workers_per_gather": self.max_parallel_workers_per_gather,
            "max_parallel_workers": self.max_parallel_workers,
            "max_worker_processes": self.max_worker_processes,
            "wal_buffers": self.wal_buffers,
            "default_statistics_target": self.default_statistics_target,
            "random_page_cost": self.random_page_cost,
            "effective_io_concurrency": self.effective_io_concurrency,
            "checkpoint_completion_target": self.checkpoint_completion_target,
            "min_wal_size": self.min_wal_size,
            "max_wal_size": self.max_wal_size,
        }


class PostgreSQLConfigGenerator:
    """Generate optimal PostgreSQL configuration based on hardware profile."""

    @staticmethod
    def generate(profile: HardwareProfile) -> PostgreSQLConfig:
        """
        Generate optimal PostgreSQL configuration for the detected hardware.

        Based on PostgreSQL best practices:
        - shared_buffers: 25% of RAM (capped at 8GB for low_spec)
        - effective_cache_size: 75% of RAM
        - work_mem: RAM / (max_connections * 3)
        - maintenance_work_mem: RAM / 16 (capped at 2GB)

        Args:
            profile: Hardware profile.

        Returns:
            PostgreSQLConfig: Optimized configuration parameters.
        """
        int(profile.total_ram_gb * 1024)

        if profile.machine_name == "high_spec":
            shared_buffers = "8GB"
            effective_cache_size = "24GB"
            work_mem = "256MB"
            maintenance_work_mem = "2GB"
            max_connections = 200
            max_parallel_workers = 8
            max_worker_processes = 16
        elif profile.machine_name == "mid_spec":
            shared_buffers = "4GB"
            effective_cache_size = "12GB"
            work_mem = "128MB"
            maintenance_work_mem = "1GB"
            max_connections = 150
            max_parallel_workers = 6
            max_worker_processes = profile.cpu_threads
        else:
            shared_buffers = "2GB"
            effective_cache_size = "6GB"
            work_mem = "64MB"
            maintenance_work_mem = "512MB"
            max_connections = 100
            max_parallel_workers = 4
            max_worker_processes = profile.cpu_threads

        max_parallel_workers_per_gather = min(4, profile.cpu_cores)

        wal_buffers = "16MB"

        random_page_cost = 1.1
        effective_io_concurrency = 200

        return PostgreSQLConfig(
            shared_buffers=shared_buffers,
            effective_cache_size=effective_cache_size,
            work_mem=work_mem,
            maintenance_work_mem=maintenance_work_mem,
            max_connections=max_connections,
            max_parallel_workers_per_gather=max_parallel_workers_per_gather,
            max_parallel_workers=max_parallel_workers,
            max_worker_processes=max_worker_processes,
            wal_buffers=wal_buffers,
            default_statistics_target=100,
            random_page_cost=random_page_cost,
            effective_io_concurrency=effective_io_concurrency,
            checkpoint_completion_target=0.9,
            min_wal_size="1GB",
            max_wal_size="4GB",
        )

    @staticmethod
    def generate_postgresql_conf(config: PostgreSQLConfig) -> str:
        """
        Generate postgresql.conf snippet with optimized settings.

        Args:
            config: PostgreSQL configuration.

        Returns:
            String with postgresql.conf format.
        """
        return f"""# PFF - Auto-generated PostgreSQL configuration
# Generated based on detected hardware

# Memory Configuration
shared_buffers = {config.shared_buffers}
effective_cache_size = {config.effective_cache_size}
work_mem = {config.work_mem}
maintenance_work_mem = {config.maintenance_work_mem}
wal_buffers = {config.wal_buffers}

# Connection Settings
max_connections = {config.max_connections}

# Parallel Query Settings
max_parallel_workers_per_gather = {config.max_parallel_workers_per_gather}
max_parallel_workers = {config.max_parallel_workers}
max_worker_processes = {config.max_worker_processes}

# Query Planner Settings
default_statistics_target = {config.default_statistics_target}
random_page_cost = {config.random_page_cost}
effective_io_concurrency = {config.effective_io_concurrency}

# WAL Settings
checkpoint_completion_target = {config.checkpoint_completion_target}
min_wal_size = {config.min_wal_size}
max_wal_size = {config.max_wal_size}

# Logging
log_min_duration_statement = 1000  # Log slow queries (>1s)

# Extensions
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.max = 10000
pg_stat_statements.track = all
"""


def get_optimal_config() -> tuple[HardwareProfile, PostgreSQLConfig]:
    """
    Convenience function to get hardware profile and optimal PostgreSQL config.

    Returns:
        Tuple of (HardwareProfile, PostgreSQLConfig).
    """
    profile = HardwareDetector.detect()
    config = PostgreSQLConfigGenerator.generate(profile)
    return profile, config


def print_hardware_info():
    """Print detected hardware information (for debugging/info)."""
    profile = HardwareDetector.detect()

    logger.debug("Hardware Detection Results")
    logger.debug(f"Machine Type: {profile.machine_name.upper()}")
    logger.debug(f"Platform: {profile.platform} ({'WSL' if profile.is_wsl else 'Native'})")
    logger.debug(
        f"RAM: {profile.total_ram_gb:.1f} GB total, {profile.available_ram_gb:.1f} GB available"
    )
    logger.debug(f"CPU: {profile.cpu_cores} cores, {profile.cpu_threads} threads")

    if profile.has_gpu:
        logger.debug(f"GPU: NVIDIA ({profile.gpu_memory_gb:.1f} GB VRAM)")
    else:
        logger.debug("GPU: Not detected")

    gpu_str = f"GPU {profile.gpu_memory_gb:.0f}GB" if profile.has_gpu else "CPU only"
    logger.info(
        f"Hardware detectado: {profile.cpu_cores} cores, {profile.total_ram_gb:.0f}GB RAM, {gpu_str}"
    )


if __name__ == "__main__":
    print_hardware_info()

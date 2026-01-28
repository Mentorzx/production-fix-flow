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

from dataclasses import dataclass

from pff.shared.core.logging import logger
from pff.shared.system.resource_manager import (
    HardwareDetector as UnifiedHardwareDetector,
    HardwareProfile as UnifiedHardwareProfile,
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

        unified: UnifiedHardwareProfile = UnifiedHardwareDetector.detect()
        return HardwareProfile(
            total_ram_gb=unified.total_ram_gb,
            available_ram_gb=unified.available_ram_gb,
            cpu_cores=unified.cpu_cores,
            cpu_threads=unified.cpu_threads,
            has_gpu=unified.has_gpu,
            gpu_memory_gb=unified.gpu_memory_gb,
            is_wsl=unified.is_wsl,
            platform=unified.platform,
            machine_name=unified.profile_name,
        )

    @staticmethod
    def _detect_gpu() -> tuple[bool, float | None]:
        """
        Detect NVIDIA GPU and its memory.

        Returns:
            Tuple of (has_gpu, gpu_memory_gb).
        """
        return UnifiedHardwareDetector._detect_gpu()

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
        return UnifiedHardwareDetector._classify_machine(total_ram_gb, has_gpu)


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
        total_ram_mb = max(int(profile.total_ram_gb * 1024), 256)

        shared_buffers_mb = int(total_ram_mb * 0.25)
        shared_buffers_mb = max(128, min(shared_buffers_mb, 8 * 1024))
        shared_buffers = f"{shared_buffers_mb}MB"

        effective_cache_size_mb = int(total_ram_mb * 0.75)
        effective_cache_size = f"{max(256, effective_cache_size_mb)}MB"

        if profile.machine_name == "high_spec":
            max_connections = 200
            max_parallel_workers = min(8, profile.cpu_threads)
        elif profile.machine_name == "mid_spec":
            max_connections = 150
            max_parallel_workers = min(6, profile.cpu_threads)
        else:
            max_connections = 100
            max_parallel_workers = min(4, profile.cpu_threads)

        work_mem_mb = int(total_ram_mb / max(max_connections * 3, 1))
        work_mem_mb = max(4, min(work_mem_mb, 64))
        work_mem = f"{work_mem_mb}MB"

        maintenance_work_mem_mb = int(total_ram_mb / 16)
        maintenance_work_mem_mb = max(64, min(maintenance_work_mem_mb, 2 * 1024))
        maintenance_work_mem = f"{maintenance_work_mem_mb}MB"

        max_worker_processes = max(2, int(profile.cpu_threads))
        max_parallel_workers_per_gather = min(4, max(1, int(profile.cpu_cores)))

        wal_buffers = "-1"

        if profile.is_wsl:
            storage_type = "wsl"
        else:
            storage_type = "ssd"

        if storage_type in {"hdd", "wsl"}:
            random_page_cost = 1.5
            effective_io_concurrency = 50
        elif storage_type == "nvme":
            random_page_cost = 1.1
            effective_io_concurrency = 200
        else:
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
        wal_buffers_line = (
            "# wal_buffers: auto-tuned by PostgreSQL"
            if str(config.wal_buffers) == "-1"
            else f"wal_buffers = {config.wal_buffers}"
        )

        return f"""# PFF - Auto-generated PostgreSQL configuration
 # Generated based on detected hardware

 # Memory Configuration
 shared_buffers = {config.shared_buffers}
 effective_cache_size = {config.effective_cache_size}
 work_mem = {config.work_mem}
 maintenance_work_mem = {config.maintenance_work_mem}
 {wal_buffers_line}

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

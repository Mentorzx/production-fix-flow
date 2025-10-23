"""
Hardware Detection Module - Auto-detect system resources for optimal PostgreSQL configuration.

This module automatically detects available hardware resources (RAM, CPU cores, GPU)
and provides optimal PostgreSQL configuration parameters based on the detected hardware.

Supports multiple machines with different specs:
- Machine 1 (low_spec): 8GB RAM, 4-8 CPU cores (WSL dev)
- Machine 2 (mid_spec): 16GB RAM, 12 CPU cores (current: Fedora WSL)
- Machine 3 (high_spec): 32GB RAM, 8-16 CPU cores, RTX 3070 Ti (production)

Author: Claude Code
Version: 1.0.0
"""

import os
import platform
import psutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
    machine_name: str  # Identifier: "low_spec" or "high_spec"

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
        # RAM detection
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024 ** 3)
        available_ram_gb = mem.available / (1024 ** 3)

        # CPU detection
        cpu_cores = psutil.cpu_count(logical=False)  # Physical cores
        cpu_threads = psutil.cpu_count(logical=True)  # Logical threads

        # GPU detection
        has_gpu, gpu_memory_gb = HardwareDetector._detect_gpu()

        # WSL detection
        is_wsl = "microsoft" in platform.uname().release.lower() or \
                 "wsl" in platform.uname().release.lower()

        # Machine profile classification
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
            machine_name=machine_name
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
        Classify machine as low_spec, mid_spec, or high_spec.

        Args:
            total_ram_gb: Total RAM in GB.
            has_gpu: Whether GPU is available.

        Returns:
            Machine classification: "low_spec", "mid_spec", or "high_spec".
        """
        # NOTE: WSL may report less RAM than physical (e.g., 7.6GB for 16GB system)
        # Use heuristics to detect actual system
        if total_ram_gb >= 24 and has_gpu:
            return "high_spec"  # Production machine (32GB RAM + RTX 3070 Ti)
        elif total_ram_gb >= 12 or (total_ram_gb >= 7 and total_ram_gb < 12):
            return "mid_spec"  # Current dev machine (16GB RAM, WSL reports ~7-8GB)
        else:
            return "low_spec"  # Low-end dev machine (8GB RAM or less)


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
        total_ram_mb = int(profile.total_ram_gb * 1024)

        if profile.machine_name == "high_spec":
            # Production machine: 32GB RAM, 8-16 cores, RTX 3070 Ti
            shared_buffers = "8GB"  # 25% of 32GB
            effective_cache_size = "24GB"  # 75% of 32GB
            work_mem = "256MB"  # For complex queries
            maintenance_work_mem = "2GB"  # For CREATE INDEX, VACUUM
            max_connections = 200
            max_parallel_workers = 8
            max_worker_processes = 16
        elif profile.machine_name == "mid_spec":
            # Current dev machine: 16GB RAM, 12 cores (WSL reports ~7-8GB)
            shared_buffers = "4GB"  # 25% of 16GB
            effective_cache_size = "12GB"  # 75% of 16GB
            work_mem = "128MB"  # Good for moderate queries
            maintenance_work_mem = "1GB"  # For CREATE INDEX, VACUUM
            max_connections = 150
            max_parallel_workers = 6
            max_worker_processes = profile.cpu_threads
        else:
            # Low-spec dev machine: 8GB RAM or less
            shared_buffers = "2GB"  # 25% of 8GB
            effective_cache_size = "6GB"  # 75% of 8GB
            work_mem = "64MB"  # Smaller for limited RAM
            maintenance_work_mem = "512MB"  # Smaller for limited RAM
            max_connections = 100
            max_parallel_workers = 4
            max_worker_processes = profile.cpu_threads

        # Parallel workers per gather (typically 2-4)
        max_parallel_workers_per_gather = min(4, profile.cpu_cores)

        # WAL buffers (3% of shared_buffers, max 16MB)
        wal_buffers = "16MB"

        # SSD optimizations (random_page_cost, effective_io_concurrency)
        random_page_cost = 1.1  # SSD (default 4.0 is for HDD)
        effective_io_concurrency = 200  # SSD (default 1 is for HDD)

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
            default_statistics_target=100,  # Default, increase to 500 for complex queries
            random_page_cost=random_page_cost,
            effective_io_concurrency=effective_io_concurrency,
            checkpoint_completion_target=0.9,  # Spread checkpoints
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
    print(f"🖥️  Hardware Detection Results")
    print(f"─" * 60)
    print(f"Machine Type: {profile.machine_name.upper()}")
    print(f"Platform: {profile.platform} ({'WSL' if profile.is_wsl else 'Native'})")
    print(f"RAM: {profile.total_ram_gb:.1f} GB total, {profile.available_ram_gb:.1f} GB available")
    print(f"CPU: {profile.cpu_cores} cores, {profile.cpu_threads} threads")

    if profile.has_gpu:
        print(f"GPU: NVIDIA ({profile.gpu_memory_gb:.1f} GB VRAM)")
    else:
        print(f"GPU: Not detected")

    print(f"\n📊 Recommended PostgreSQL Configuration")
    print(f"─" * 60)

    config = PostgreSQLConfigGenerator.generate(profile)
    for key, value in config.to_dict().items():
        print(f"{key:35} = {value}")


if __name__ == "__main__":
    print_hardware_info()

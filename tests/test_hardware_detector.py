"""
Tests for Hardware Detection (pff/utils/hardware_detector.py)

Tests cover:
- Hardware profile detection
- PostgreSQL config generation
- Machine classification (low/mid/high spec)
"""

import pytest
from unittest.mock import Mock, patch
from pff.utils.hardware_detector import (
    HardwareDetector,
    HardwareProfile,
    PostgreSQLConfigGenerator,
    get_optimal_config,
)


@pytest.mark.unit
class TestHardwareDetector:
    """Test HardwareDetector class."""

    def test_detect_returns_valid_profile(self):
        """Test that detect() returns a valid HardwareProfile."""
        profile = HardwareDetector.detect()

        assert isinstance(profile, HardwareProfile)
        assert profile.total_ram_gb > 0
        assert profile.cpu_cores > 0
        assert profile.cpu_threads >= profile.cpu_cores
        assert profile.machine_name in ["low_spec", "mid_spec", "high_spec"]

    def test_machine_classification_high_spec(self):
        """Test high_spec classification (32GB RAM + GPU)."""
        machine_name = HardwareDetector._classify_machine(total_ram_gb=32.0, has_gpu=True)
        assert machine_name == "high_spec"

    def test_machine_classification_mid_spec_full_ram(self):
        """Test mid_spec classification (16GB RAM, no GPU)."""
        machine_name = HardwareDetector._classify_machine(total_ram_gb=16.0, has_gpu=False)
        assert machine_name == "mid_spec"

    def test_machine_classification_mid_spec_wsl_limited(self):
        """Test mid_spec classification (WSL reporting ~7-8GB for 16GB system)."""
        machine_name = HardwareDetector._classify_machine(total_ram_gb=7.6, has_gpu=False)
        assert machine_name == "mid_spec"

    def test_machine_classification_low_spec(self):
        """Test low_spec classification (8GB RAM or less)."""
        machine_name = HardwareDetector._classify_machine(total_ram_gb=6.0, has_gpu=False)
        assert machine_name == "low_spec"

    def test_gpu_detection_returns_tuple(self):
        """Test GPU detection returns (bool, Optional[float]) tuple."""
        has_gpu, gpu_memory_gb = HardwareDetector._detect_gpu()

        assert isinstance(has_gpu, bool)
        assert gpu_memory_gb is None or isinstance(gpu_memory_gb, float)

        # If GPU detected, memory should be positive
        if has_gpu:
            assert gpu_memory_gb > 0


@pytest.mark.unit
class TestPostgreSQLConfigGenerator:
    """Test PostgreSQL configuration generator."""

    def test_config_generation_high_spec(self):
        """Test PostgreSQL config for high_spec machine (32GB RAM, GPU)."""
        profile = HardwareProfile(
            total_ram_gb=32.0,
            available_ram_gb=28.0,
            cpu_cores=8,
            cpu_threads=16,
            has_gpu=True,
            gpu_memory_gb=8.0,
            is_wsl=False,
            platform="Linux",
            machine_name="high_spec",
        )

        config = PostgreSQLConfigGenerator.generate(profile)

        assert config.shared_buffers == "8GB"
        assert config.effective_cache_size == "24GB"
        assert config.work_mem == "256MB"
        assert config.maintenance_work_mem == "2GB"
        assert config.max_connections == 200
        assert config.max_parallel_workers == 8

    def test_config_generation_mid_spec(self):
        """Test PostgreSQL config for mid_spec machine (16GB RAM, 12 cores)."""
        profile = HardwareProfile(
            total_ram_gb=16.0,
            available_ram_gb=14.0,
            cpu_cores=6,
            cpu_threads=12,
            has_gpu=False,
            gpu_memory_gb=None,
            is_wsl=True,
            platform="Linux",
            machine_name="mid_spec",
        )

        config = PostgreSQLConfigGenerator.generate(profile)

        assert config.shared_buffers == "4GB"
        assert config.effective_cache_size == "12GB"
        assert config.work_mem == "128MB"
        assert config.maintenance_work_mem == "1GB"
        assert config.max_connections == 150
        assert config.max_parallel_workers == 6
        assert config.max_worker_processes == 12

    def test_config_generation_low_spec(self):
        """Test PostgreSQL config for low_spec machine (8GB RAM)."""
        profile = HardwareProfile(
            total_ram_gb=8.0,
            available_ram_gb=6.0,
            cpu_cores=4,
            cpu_threads=4,
            has_gpu=False,
            gpu_memory_gb=None,
            is_wsl=False,
            platform="Linux",
            machine_name="low_spec",
        )

        config = PostgreSQLConfigGenerator.generate(profile)

        assert config.shared_buffers == "2GB"
        assert config.effective_cache_size == "6GB"
        assert config.work_mem == "64MB"
        assert config.maintenance_work_mem == "512MB"
        assert config.max_connections == 100
        assert config.max_parallel_workers == 4

    def test_config_to_dict(self):
        """Test PostgreSQLConfig.to_dict() method."""
        profile = HardwareProfile(
            total_ram_gb=16.0,
            available_ram_gb=14.0,
            cpu_cores=6,
            cpu_threads=12,
            has_gpu=False,
            gpu_memory_gb=None,
            is_wsl=True,
            platform="Linux",
            machine_name="mid_spec",
        )

        config = PostgreSQLConfigGenerator.generate(profile)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "shared_buffers" in config_dict
        assert "effective_cache_size" in config_dict
        assert "max_connections" in config_dict
        assert config_dict["shared_buffers"] == "4GB"

    def test_generate_postgresql_conf_format(self):
        """Test PostgreSQL conf file generation format."""
        profile = HardwareProfile(
            total_ram_gb=16.0,
            available_ram_gb=14.0,
            cpu_cores=6,
            cpu_threads=12,
            has_gpu=False,
            gpu_memory_gb=None,
            is_wsl=True,
            platform="Linux",
            machine_name="mid_spec",
        )

        config = PostgreSQLConfigGenerator.generate(profile)
        conf_str = PostgreSQLConfigGenerator.generate_postgresql_conf(config)

        # Verify essential parameters are in the output
        assert "shared_buffers = 4GB" in conf_str
        assert "effective_cache_size = 12GB" in conf_str
        assert "max_connections = 150" in conf_str
        assert "max_parallel_workers = 6" in conf_str
        assert "random_page_cost = 1.1" in conf_str  # SSD optimization
        assert "shared_preload_libraries = 'pg_stat_statements'" in conf_str


@pytest.mark.unit
class TestGetOptimalConfig:
    """Test get_optimal_config() convenience function."""

    def test_get_optimal_config_returns_tuple(self):
        """Test that get_optimal_config() returns (profile, config) tuple."""
        profile, config = get_optimal_config()

        assert isinstance(profile, HardwareProfile)
        assert profile.total_ram_gb > 0

        # Verify config matches profile classification
        if profile.machine_name == "high_spec":
            assert config.shared_buffers == "8GB"
        elif profile.machine_name == "mid_spec":
            assert config.shared_buffers == "4GB"
        else:  # low_spec
            assert config.shared_buffers == "2GB"

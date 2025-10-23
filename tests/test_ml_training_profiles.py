"""Tests for ML Training Profiles - Hardware-aware configurations."""

import pytest

from pff.utils.hardware_detector import HardwareDetector, HardwareProfile
from pff.utils.ml_training_profiles import (
    MLTrainingProfile,
    MLTrainingProfileGenerator,
    get_ml_training_profile,
)


class TestMLTrainingProfileGeneration:
    """Test ML training profile generation for different hardware specs."""

    def test_low_spec_profile_safe_limits(self):
        """Low-spec profile should have conservative limits to prevent OOM."""
        low_spec_hw = HardwareProfile(
            total_ram_gb=8.0,
            available_ram_gb=5.0,
            cpu_cores=4,
            cpu_threads=4,
            has_gpu=False,
            gpu_memory_gb=None,
            is_wsl=False,
            platform="Linux",
            machine_name="low_spec",
        )

        profile = MLTrainingProfileGenerator.generate(low_spec_hw)

        # Verify safe limits
        assert profile.machine_name == "low_spec"
        assert profile.transe.embedding_dim == 64  # Reduced dimension
        assert profile.transe.batch_size <= 512  # Small batch
        assert profile.transe.max_entities == 50_000  # Hard limit
        assert profile.transe.use_gpu is False
        assert profile.anyburl.max_memory_gb <= 6  # Leave RAM for OS
        assert profile.ray_object_store_memory_gb <= 2

    def test_mid_spec_profile_moderate_config(self):
        """Mid-spec profile should balance performance and safety."""
        mid_spec_hw = HardwareProfile(
            total_ram_gb=11.7,
            available_ram_gb=6.4,
            cpu_cores=6,
            cpu_threads=12,
            has_gpu=False,
            gpu_memory_gb=None,
            is_wsl=True,
            platform="Linux",
            machine_name="mid_spec",
        )

        profile = MLTrainingProfileGenerator.generate(mid_spec_hw)

        # Verify moderate config
        assert profile.machine_name == "mid_spec"
        assert profile.transe.embedding_dim == 128  # Full dimension
        assert profile.transe.batch_size == 512
        assert profile.transe.max_entities == 200_000  # Limited but reasonable
        assert profile.transe.num_epochs == 50  # Reduced for faster iteration
        assert profile.anyburl.max_memory_gb == 10  # Leave 2-6GB for OS
        assert profile.ray_object_store_memory_gb == 4

    def test_high_spec_profile_full_training(self):
        """High-spec profile should enable full production training."""
        high_spec_hw = HardwareProfile(
            total_ram_gb=32.0,
            available_ram_gb=24.0,
            cpu_cores=8,
            cpu_threads=16,
            has_gpu=True,
            gpu_memory_gb=8.0,
            is_wsl=False,
            platform="Linux",
            machine_name="high_spec",
        )

        profile = MLTrainingProfileGenerator.generate(high_spec_hw)

        # Verify production config
        assert profile.machine_name == "high_spec"
        assert profile.transe.embedding_dim == 256  # Large embeddings
        assert profile.transe.batch_size == 2048  # Large batch for GPU
        assert profile.transe.max_entities is None  # No limit
        assert profile.transe.use_gpu is True
        assert profile.transe.num_epochs == 100  # Full training
        assert profile.anyburl.max_memory_gb == 20
        assert profile.ray_object_store_memory_gb == 8

    def test_gpu_detection_affects_transe_config(self):
        """GPU availability should affect TransE configuration."""
        # No GPU
        no_gpu_hw = HardwareProfile(
            total_ram_gb=16.0,
            available_ram_gb=10.0,
            cpu_cores=6,
            cpu_threads=12,
            has_gpu=False,
            gpu_memory_gb=None,
            is_wsl=True,
            platform="Linux",
            machine_name="mid_spec",
        )

        profile_no_gpu = MLTrainingProfileGenerator.generate(no_gpu_hw)
        assert profile_no_gpu.transe.use_gpu is False

        # With GPU
        with_gpu_hw = HardwareProfile(
            total_ram_gb=32.0,
            available_ram_gb=24.0,
            cpu_cores=8,
            cpu_threads=16,
            has_gpu=True,
            gpu_memory_gb=8.0,
            is_wsl=False,
            platform="Linux",
            machine_name="high_spec",
        )

        profile_with_gpu = MLTrainingProfileGenerator.generate(with_gpu_hw)
        assert profile_with_gpu.transe.use_gpu is True

    def test_warnings_for_low_spec(self):
        """Low-spec profile should have multiple warnings."""
        low_spec_hw = HardwareProfile(
            total_ram_gb=8.0,
            available_ram_gb=5.0,
            cpu_cores=4,
            cpu_threads=4,
            has_gpu=False,
            gpu_memory_gb=None,
            is_wsl=False,
            platform="Linux",
            machine_name="low_spec",
        )

        profile = MLTrainingProfileGenerator.generate(low_spec_hw)
        warnings = profile.get_warnings()

        assert len(warnings) >= 3
        assert any("LOW_SPEC" in w for w in warnings)
        assert any("50k entidades" in w for w in warnings)
        assert any("CPU apenas" in w for w in warnings)

    def test_warnings_for_mid_spec(self):
        """Mid-spec profile should have informational warnings."""
        mid_spec_hw = HardwareProfile(
            total_ram_gb=11.7,
            available_ram_gb=6.4,
            cpu_cores=6,
            cpu_threads=12,
            has_gpu=False,
            gpu_memory_gb=None,
            is_wsl=True,
            platform="Linux",
            machine_name="mid_spec",
        )

        profile = MLTrainingProfileGenerator.generate(mid_spec_hw)
        warnings = profile.get_warnings()

        assert len(warnings) >= 3
        assert any("MID_SPEC" in w for w in warnings)
        assert any("desenvolvimento" in w for w in warnings)

    def test_warnings_for_high_spec(self):
        """High-spec profile should have success messages."""
        high_spec_hw = HardwareProfile(
            total_ram_gb=32.0,
            available_ram_gb=24.0,
            cpu_cores=8,
            cpu_threads=16,
            has_gpu=True,
            gpu_memory_gb=8.0,
            is_wsl=False,
            platform="Linux",
            machine_name="high_spec",
        )

        profile = MLTrainingProfileGenerator.generate(high_spec_hw)
        warnings = profile.get_warnings()

        assert len(warnings) >= 2
        assert any("HIGH_SPEC" in w for w in warnings)
        assert any("GPU detectada" in w for w in warnings)


class TestMLTrainingProfileOOMPrevention:
    """Test that profiles prevent OOM errors."""

    def test_low_spec_anyburl_memory_limit(self):
        """AnyBURL memory should be capped for low-spec machines."""
        low_spec_hw = HardwareProfile(
            total_ram_gb=8.0,
            available_ram_gb=5.0,
            cpu_cores=4,
            cpu_threads=4,
            has_gpu=False,
            gpu_memory_gb=None,
            is_wsl=False,
            platform="Linux",
            machine_name="low_spec",
        )

        profile = MLTrainingProfileGenerator.generate(low_spec_hw)

        # AnyBURL should use at most 75% of RAM (6GB of 8GB)
        assert profile.anyburl.max_memory_gb <= 6

    def test_mid_spec_ray_object_store_limit(self):
        """Ray object store should be limited for mid-spec machines."""
        mid_spec_hw = HardwareProfile(
            total_ram_gb=11.7,
            available_ram_gb=6.4,
            cpu_cores=6,
            cpu_threads=12,
            has_gpu=False,
            gpu_memory_gb=None,
            is_wsl=True,
            platform="Linux",
            machine_name="mid_spec",
        )

        profile = MLTrainingProfileGenerator.generate(mid_spec_hw)

        # Ray object store should be 4GB (reasonable for 12GB RAM)
        assert profile.ray_object_store_memory_gb == 4

    def test_entity_limit_prevents_oom_low_spec(self):
        """Low-spec machines should have hard entity limit."""
        low_spec_hw = HardwareProfile(
            total_ram_gb=8.0,
            available_ram_gb=5.0,
            cpu_cores=4,
            cpu_threads=4,
            has_gpu=False,
            gpu_memory_gb=None,
            is_wsl=False,
            platform="Linux",
            machine_name="low_spec",
        )

        profile = MLTrainingProfileGenerator.generate(low_spec_hw)

        # Low-spec MUST have entity limit
        assert profile.transe.max_entities is not None
        assert profile.transe.max_entities == 50_000

    def test_entity_limit_increased_for_mid_spec(self):
        """Mid-spec machines should have higher but still safe entity limit."""
        mid_spec_hw = HardwareProfile(
            total_ram_gb=11.7,
            available_ram_gb=6.4,
            cpu_cores=6,
            cpu_threads=12,
            has_gpu=False,
            gpu_memory_gb=None,
            is_wsl=True,
            platform="Linux",
            machine_name="mid_spec",
        )

        profile = MLTrainingProfileGenerator.generate(mid_spec_hw)

        # Mid-spec should have limit but higher than low-spec
        assert profile.transe.max_entities is not None
        assert profile.transe.max_entities == 200_000

    def test_no_entity_limit_for_high_spec(self):
        """High-spec machines should have no entity limit."""
        high_spec_hw = HardwareProfile(
            total_ram_gb=32.0,
            available_ram_gb=24.0,
            cpu_cores=8,
            cpu_threads=16,
            has_gpu=True,
            gpu_memory_gb=8.0,
            is_wsl=False,
            platform="Linux",
            machine_name="high_spec",
        )

        profile = MLTrainingProfileGenerator.generate(high_spec_hw)

        # High-spec should have no limit
        assert profile.transe.max_entities is None


class TestMLTrainingProfileIntegration:
    """Integration tests for ML training profiles."""

    def test_get_ml_training_profile_returns_valid_config(self):
        """get_ml_training_profile() should return valid config for current machine."""
        profile = get_ml_training_profile()

        # Should be a valid profile
        assert isinstance(profile, MLTrainingProfile)
        assert profile.machine_name in ["low_spec", "mid_spec", "high_spec"]

        # Should have all required configs
        assert profile.transe is not None
        assert profile.anyburl is not None
        assert profile.lightgbm is not None
        assert profile.ray_num_cpus > 0
        assert profile.ray_object_store_memory_gb is not None

    def test_current_machine_profile_matches_hardware(self):
        """Current machine profile should match hardware detection."""
        hardware = HardwareDetector.detect()
        ml_profile = get_ml_training_profile()

        # Machine names should match
        assert ml_profile.machine_name == hardware.machine_name

    def test_thread_count_respects_hardware_limits(self):
        """All thread counts should respect hardware CPU limits."""
        profile = get_ml_training_profile()
        hardware = HardwareDetector.detect()

        # All thread configs should be <= available threads
        assert profile.transe.num_workers <= hardware.cpu_threads
        assert profile.anyburl.num_threads <= hardware.cpu_threads
        assert profile.lightgbm.num_threads <= hardware.cpu_threads
        assert profile.ray_num_cpus <= hardware.cpu_threads

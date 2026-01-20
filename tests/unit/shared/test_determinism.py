"""Tests for pff/shared/determinism.py - reproducibility utilities."""

import os
import random

import numpy as np
import pytest

from pff.shared.determinism import (
    configure_numba_threads,
    configure_torch_determinism,
    set_global_seed,
    validate_determinism,
)


class TestSetGlobalSeed:
    """Tests for the set_global_seed function."""

    def test_set_global_seed_python_random(self):
        """Verify Python random is seeded correctly."""
        set_global_seed(42)
        val1 = random.random()
        set_global_seed(42)
        val2 = random.random()
        assert val1 == val2

    def test_set_global_seed_numpy_random(self):
        """Verify NumPy random is seeded correctly."""
        set_global_seed(42)
        arr1 = np.random.rand(5)
        set_global_seed(42)
        arr2 = np.random.rand(5)
        np.testing.assert_array_equal(arr1, arr2)

    def test_set_global_seed_different_seeds_differ(self):
        """Verify different seeds produce different results."""
        set_global_seed(42)
        val1 = random.random()
        set_global_seed(123)
        val2 = random.random()
        assert val1 != val2

    def test_set_global_seed_sets_env_variable(self):
        """Verify PYTHONHASHSEED is set."""
        set_global_seed(99)
        assert os.environ.get("PYTHONHASHSEED") == "99"

    def test_set_global_seed_default_value(self):
        """Verify default seed value works."""
        set_global_seed()  # Default is 42
        val1 = random.random()
        set_global_seed(42)
        val2 = random.random()
        assert val1 == val2

    def test_set_global_seed_with_torch(self):
        """Verify PyTorch seeding works if available."""
        try:
            import torch

            set_global_seed(42)
            t1 = torch.rand(5)
            set_global_seed(42)
            t2 = torch.rand(5)
            assert torch.allclose(t1, t2)
        except ImportError:
            pytest.skip("PyTorch not installed")


class TestValidateDeterminism:
    """Tests for the validate_determinism function."""

    def test_validate_determinism_with_deterministic_func(self):
        """Verify deterministic function passes validation."""

        def deterministic_sum(a, b):
            return a + b

        result = validate_determinism(deterministic_sum, 1, 2)
        assert result is True

    def test_validate_determinism_with_seeded_random(self):
        """Verify seeded random function passes validation."""

        def seeded_random():
            return random.random()

        result = validate_determinism(seeded_random)
        assert result is True

    def test_validate_determinism_with_numpy_function(self):
        """Verify NumPy-based function passes validation."""

        def numpy_sum():
            arr = np.random.rand(10)
            return np.sum(arr)

        result = validate_determinism(numpy_sum)
        assert result is True

    def test_validate_determinism_with_custom_tolerance(self):
        """Verify custom tolerance works."""

        def small_variation():
            return 1.0 + np.random.rand() * 1e-8  # Tiny variation

        # This should fail with tight tolerance since we're adding random noise
        # But since set_global_seed resets, it should pass
        result = validate_determinism(small_variation, tolerance=1e-6)
        assert result is True


class TestConfigureTorchDeterminism:
    """Tests for configure_torch_determinism."""

    def test_configure_torch_determinism_without_torch(self):
        """Verify function handles missing torch gracefully."""
        # Should not raise even if torch config fails internally
        configure_torch_determinism(enforce=True)

    def test_configure_torch_determinism_sets_cublas_env(self):
        """Verify CUBLAS_WORKSPACE_CONFIG is set."""
        # Clear first
        if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]
        configure_torch_determinism(enforce=True)
        # Should be set to default
        assert "CUBLAS_WORKSPACE_CONFIG" in os.environ


class TestConfigureNumbaThreads:
    """Tests for configure_numba_threads."""

    def test_configure_numba_threads_returns_int(self):
        """Verify function returns an integer."""
        result = configure_numba_threads()
        assert isinstance(result, int)
        assert result >= 0

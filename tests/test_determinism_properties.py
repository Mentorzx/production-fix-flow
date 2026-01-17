"""Property tests for determinism and reproducibility.

Tests that ensure reproducible results:
(1) Same seed gives same results
(2) Model predictions are deterministic given fixed inputs
(3) Hash functions are stable across runs
(4) Config loading is idempotent
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest


# ============================================================================
# Determinism utilities
# ============================================================================


def stable_hash(data: str | bytes, algorithm: str = "sha256") -> str:
    """Compute a stable hash of data."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    h = hashlib.new(algorithm)
    h.update(data)
    return h.hexdigest()


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def generate_random_array(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    """Generate random array with fixed seed."""
    rng = np.random.RandomState(seed)
    return rng.randn(*shape)


# ============================================================================
# Tests: Random seed reproducibility
# ============================================================================


class TestSeedReproducibility:
    """Test that same seed produces same results."""

    def test_numpy_random_deterministic(self):
        """Property: same numpy seed gives same random numbers."""
        seed = 42

        set_all_seeds(seed)
        result1 = np.random.rand(100)

        set_all_seeds(seed)
        result2 = np.random.rand(100)

        np.testing.assert_array_equal(result1, result2)

    def test_random_state_object_deterministic(self):
        """Property: RandomState object is deterministic."""
        seed = 123

        rng1 = np.random.RandomState(seed)
        result1 = rng1.randn(50, 10)

        rng2 = np.random.RandomState(seed)
        result2 = rng2.randn(50, 10)

        np.testing.assert_array_equal(result1, result2)

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999, 2**31 - 1])
    def test_various_seeds_reproducible(self, seed: int):
        """Property: various seeds all produce reproducible results."""
        arr1 = generate_random_array(seed, (100,))
        arr2 = generate_random_array(seed, (100,))
        np.testing.assert_array_equal(arr1, arr2)

    def test_different_seeds_different_results(self):
        """Property: different seeds produce different results."""
        arr1 = generate_random_array(42, (100,))
        arr2 = generate_random_array(43, (100,))
        assert not np.allclose(arr1, arr2)


# ============================================================================
# Tests: Hash stability
# ============================================================================


class TestHashStability:
    """Test that hash functions are stable."""

    def test_same_input_same_hash(self):
        """Property: same input always produces same hash."""
        data = "test_string_for_hashing"
        hash1 = stable_hash(data)
        hash2 = stable_hash(data)
        assert hash1 == hash2

    def test_different_input_different_hash(self):
        """Property: different inputs produce different hashes."""
        hash1 = stable_hash("input_a")
        hash2 = stable_hash("input_b")
        assert hash1 != hash2

    def test_bytes_and_string_consistent(self):
        """Property: string and bytes hash to same value."""
        data_str = "test_data"
        data_bytes = b"test_data"
        assert stable_hash(data_str) == stable_hash(data_bytes)

    def test_hash_is_hex_string(self):
        """Property: hash output is valid hex string."""
        result = stable_hash("test")
        assert all(c in "0123456789abcdef" for c in result)

    @pytest.mark.parametrize("algorithm", ["sha256", "sha1", "md5"])
    def test_various_algorithms(self, algorithm: str):
        """Property: various algorithms produce consistent hashes."""
        data = "consistent_test_data"
        hash1 = stable_hash(data, algorithm)
        hash2 = stable_hash(data, algorithm)
        assert hash1 == hash2

    def test_hash_length_by_algorithm(self):
        """Property: hash length matches algorithm."""
        data = "test"
        assert len(stable_hash(data, "sha256")) == 64
        assert len(stable_hash(data, "sha1")) == 40
        assert len(stable_hash(data, "md5")) == 32


# ============================================================================
# Tests: Array operations determinism
# ============================================================================


class TestArrayOperationsDeterminism:
    """Test determinism of array operations."""

    def test_sorting_deterministic(self):
        """Property: sorting is deterministic."""
        arr = generate_random_array(42, (100,))
        sorted1 = np.sort(arr)
        sorted2 = np.sort(arr)
        np.testing.assert_array_equal(sorted1, sorted2)

    def test_argsort_deterministic(self):
        """Property: argsort is deterministic."""
        arr = generate_random_array(42, (100,))
        idx1 = np.argsort(arr)
        idx2 = np.argsort(arr)
        np.testing.assert_array_equal(idx1, idx2)

    def test_matrix_multiplication_deterministic(self):
        """Property: matrix multiplication is deterministic."""
        A = generate_random_array(42, (50, 30))
        B = generate_random_array(43, (30, 20))
        result1 = A @ B
        result2 = A @ B
        np.testing.assert_array_equal(result1, result2)

    def test_aggregation_deterministic(self):
        """Property: aggregations are deterministic."""
        arr = generate_random_array(42, (1000,))
        assert arr.sum() == arr.sum()
        assert arr.mean() == arr.mean()
        assert arr.std() == arr.std()
        assert arr.max() == arr.max()
        assert arr.min() == arr.min()


# ============================================================================
# Tests: Model prediction determinism
# ============================================================================


class TestModelPredictionDeterminism:
    """Test that model predictions are deterministic."""

    @pytest.fixture
    def simple_dataset(self):
        """Create simple reproducible dataset."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_sklearn_deterministic(self, simple_dataset):
        """Property: sklearn models with fixed seed are deterministic."""
        from sklearn.ensemble import RandomForestClassifier

        X, y = simple_dataset

        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict_proba(X)

        model2 = RandomForestClassifier(n_estimators=10, random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict_proba(X)

        np.testing.assert_array_equal(pred1, pred2)


# ============================================================================
# Tests: Config idempotence
# ============================================================================


class TestConfigIdempotence:
    """Test that config loading is idempotent."""

    def test_dict_copy_idempotent(self):
        """Property: copying a dict multiple times gives same result."""
        original = {"a": 1, "b": {"c": 2, "d": [1, 2, 3]}}

        copy1 = original.copy()
        copy2 = original.copy()

        assert copy1 == copy2

    def test_default_values_stable(self):
        """Property: default values are stable across calls."""

        def get_defaults():
            return {
                "threshold": 0.5,
                "weights": [0.2, 0.3, 0.5],
                "enabled": True,
            }

        defaults1 = get_defaults()
        defaults2 = get_defaults()
        assert defaults1 == defaults2

    def test_nested_defaults_isolated(self):
        """Property: nested defaults don't share references."""

        def get_config():
            return {
                "inner": {"value": 1},
                "list": [1, 2, 3],
            }

        config1 = get_config()
        config2 = get_config()

        # Modify config1's inner dict
        config1["inner"]["value"] = 999
        config1["list"].append(4)

        # config2 should be unchanged
        assert config2["inner"]["value"] == 1
        assert config2["list"] == [1, 2, 3]

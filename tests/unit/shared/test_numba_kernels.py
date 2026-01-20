"""Tests for pff/shared/acceleration/numba_kernels.py.

Tests VocabularyEncoder, BloomFilter, TripleStoreSoA, batch size calculation,
and negative sampling without heavy computation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pff.shared.acceleration.numba_kernels import (
    NUMBA_AVAILABLE,
    BloomFilter,
    TripleStoreSoA,
    VocabularyEncoder,
    _to_py_scalar,
    _to_py_str,
    binary_search_range,
    calculate_optimal_batch_size,
    find_matching_triples_accelerated,
    generate_emu_noise,
    generate_negative_samples,
    get_numba_diagnostics,
)

# ─────────────────────────── _to_py_scalar / _to_py_str Tests ────────────────


class TestToPyScalar:
    """Tests for _to_py_scalar utility."""

    def test_to_py_scalar_none(self) -> None:
        """None should return None."""
        assert _to_py_scalar(None) is None

    def test_to_py_scalar_python_types(self) -> None:
        """Python primitives should pass through."""
        assert _to_py_scalar(42) == 42
        assert _to_py_scalar("hello") == "hello"
        assert _to_py_scalar(3.14) == 3.14

    def test_to_py_scalar_numpy_scalar(self) -> None:
        """NumPy scalars should be converted to Python."""
        assert _to_py_scalar(np.int32(42)) == 42
        assert _to_py_scalar(np.float64(3.14)) == 3.14

    def test_to_py_scalar_numpy_0d_array(self) -> None:
        """0-d numpy array should return scalar."""
        arr = np.array(42)
        assert _to_py_scalar(arr) == 42

    def test_to_py_scalar_numpy_1element_array(self) -> None:
        """1-element numpy array should return scalar."""
        arr = np.array([42])
        assert _to_py_scalar(arr) == 42

    def test_to_py_scalar_numpy_multi_element_array(self) -> None:
        """Multi-element numpy array should return tuple."""
        arr = np.array([1, 2, 3])
        result = _to_py_scalar(arr)
        assert result == (1, 2, 3)

    def test_to_py_scalar_list(self) -> None:
        """Lists should be handled."""
        assert _to_py_scalar([42]) == 42
        assert _to_py_scalar([1, 2, 3]) == (1, 2, 3)


class TestToPyStr:
    """Tests for _to_py_str utility."""

    def test_to_py_str_none(self) -> None:
        """None should return empty string."""
        assert _to_py_str(None) == ""

    def test_to_py_str_string(self) -> None:
        """String should pass through."""
        assert _to_py_str("hello") == "hello"

    def test_to_py_str_int(self) -> None:
        """Int should be converted to string."""
        assert _to_py_str(42) == "42"

    def test_to_py_str_numpy_scalar(self) -> None:
        """NumPy scalar should be converted to string."""
        assert _to_py_str(np.int32(42)) == "42"


# ─────────────────────────── VocabularyEncoder Tests ───────────────────────────


class TestVocabularyEncoder:
    """Tests for VocabularyEncoder."""

    def test_encode_entity_new(self) -> None:
        """New entities should get sequential indices."""
        encoder = VocabularyEncoder()
        idx1 = encoder.encode_entity("entity_a")
        idx2 = encoder.encode_entity("entity_b")
        assert idx1 == 0
        assert idx2 == 1

    def test_encode_entity_existing(self) -> None:
        """Existing entities should return same index."""
        encoder = VocabularyEncoder()
        idx1 = encoder.encode_entity("entity_a")
        idx2 = encoder.encode_entity("entity_a")
        assert idx1 == idx2

    def test_decode_entity(self) -> None:
        """Decode should return original entity."""
        encoder = VocabularyEncoder()
        idx = encoder.encode_entity("test_entity")
        assert encoder.decode_entity(idx) == "test_entity"

    def test_decode_unknown_entity(self) -> None:
        """Unknown index should return placeholder."""
        encoder = VocabularyEncoder()
        result = encoder.decode_entity(9999)
        assert "unknown" in result.lower()

    def test_encode_relation_new(self) -> None:
        """New relations should get sequential indices."""
        encoder = VocabularyEncoder()
        idx1 = encoder.encode_relation("rel_a")
        idx2 = encoder.encode_relation("rel_b")
        assert idx1 == 0
        assert idx2 == 1

    def test_encode_relation_wildcard(self) -> None:
        """Wildcard relation should return WILDCARD_IDX."""
        encoder = VocabularyEncoder()
        idx = encoder.encode_relation("*")
        assert idx == encoder.WILDCARD_IDX

    def test_decode_relation(self) -> None:
        """Decode should return original relation."""
        encoder = VocabularyEncoder()
        idx = encoder.encode_relation("test_rel")
        assert encoder.decode_relation(idx) == "test_rel"

    def test_decode_wildcard_relation(self) -> None:
        """Wildcard index should decode to '*'."""
        encoder = VocabularyEncoder()
        assert encoder.decode_relation(encoder.WILDCARD_IDX) == "*"

    def test_encode_triples(self) -> None:
        """encode_triples should return (n, 3) int32 array."""
        encoder = VocabularyEncoder()
        triples = [("a", "rel", "b"), ("c", "rel", "d")]
        encoded = encoder.encode_triples(triples)
        assert encoded.shape == (2, 3)
        assert encoded.dtype == np.int32

    def test_encode_triples_numpy_input(self) -> None:
        """encode_triples should handle numpy array input."""
        encoder = VocabularyEncoder()
        triples = np.array([["a", "rel", "b"], ["c", "rel", "d"]])
        encoded = encoder.encode_triples(triples)
        assert encoded.shape == (2, 3)
        assert encoded.dtype == np.int32

    def test_encode_pattern(self) -> None:
        """encode_pattern should return 5-tuple."""
        encoder = VocabularyEncoder()
        pattern = {"predicate": "knows", "args": ["alice", "bob"]}
        result = encoder.encode_pattern(pattern)
        assert len(result) == 5
        # (pred_idx, arg0_idx, arg0_is_var, arg1_idx, arg1_is_var)

    def test_encode_pattern_with_variable(self) -> None:
        """Variables (uppercase) should be marked as variables."""
        encoder = VocabularyEncoder()
        pattern = {"predicate": "knows", "args": ["X", "bob"]}
        pred_idx, arg0_idx, arg0_is_var, arg1_idx, arg1_is_var = encoder.encode_pattern(pattern)
        assert arg0_is_var == 1
        assert arg1_is_var == 0

    def test_encode_pattern_short_args(self) -> None:
        """Pattern with < 2 args should return zeros."""
        encoder = VocabularyEncoder()
        pattern = {"predicate": "knows", "args": ["alice"]}
        pred_idx, arg0_idx, arg0_is_var, arg1_idx, arg1_is_var = encoder.encode_pattern(pattern)
        assert arg0_idx == 0
        assert arg1_idx == 0


# ─────────────────────────── BloomFilter Tests ───────────────────────────


class TestBloomFilter:
    """Tests for BloomFilter."""

    def test_bloom_filter_add_and_check(self) -> None:
        """Added items should be detected."""
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
        bf.add(42)
        assert bf.might_contain(42) is True

    def test_bloom_filter_missing_item(self) -> None:
        """Non-added items should (usually) not be detected."""
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
        bf.add(42)
        missing_detected = sum(1 for i in range(1000, 2000) if bf.might_contain(i))
        assert missing_detected < 50

    def test_bloom_filter_add_batch(self) -> None:
        """add_batch should add multiple items."""
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
        items = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        bf.add_batch(items)
        assert bf.items_added == 5
        for item in items:
            assert bf.might_contain(int(item)) is True

    def test_bloom_filter_size_scales_with_items(self) -> None:
        """Larger expected items should create larger filter."""
        bf_small = BloomFilter(expected_items=100, false_positive_rate=0.01)
        bf_large = BloomFilter(expected_items=10000, false_positive_rate=0.01)
        assert bf_large.size > bf_small.size


# ─────────────────────────── TripleStoreSoA Tests ───────────────────────────


class TestTripleStoreSoA:
    """Tests for TripleStoreSoA."""

    def test_triple_store_initialization(self) -> None:
        """TripleStoreSoA should initialize with correct size."""
        store = TripleStoreSoA(100)
        assert store.n_triples == 100
        assert len(store.subjects) == 100
        assert len(store.predicates) == 100
        assert len(store.objects) == 100

    def test_triple_store_load_from_triples(self) -> None:
        """load_from_triples should populate arrays."""
        triples = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        store = TripleStoreSoA(2)
        store.load_from_triples(triples)
        assert np.array_equal(store.subjects, np.array([1, 4]))
        assert np.array_equal(store.predicates, np.array([2, 5]))
        assert np.array_equal(store.objects, np.array([3, 6]))

    def test_triple_store_load_wrong_size_raises(self) -> None:
        """load_from_triples with wrong size should raise."""
        triples = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        store = TripleStoreSoA(10)  # Wrong size
        with pytest.raises(ValueError, match="Expected 10 triples"):
            store.load_from_triples(triples)

    def test_triple_store_build_indexes(self) -> None:
        """build_indexes should create SPO, POS, OSP indexes."""
        triples = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 7]], dtype=np.int32)
        store = TripleStoreSoA(3)
        store.load_from_triples(triples)
        store.build_indexes()
        assert store._spo_index is not None
        assert store._pos_index is not None
        assert store._osp_index is not None
        assert len(store.spo_index) == 3
        assert len(store.pos_index) == 3
        assert len(store.osp_index) == 3

    def test_triple_store_indexes_lazy_build(self) -> None:
        """Accessing index properties should trigger lazy build."""
        triples = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        store = TripleStoreSoA(2)
        store.load_from_triples(triples)
        assert store._spo_index is None
        _ = store.spo_index  # Should trigger build
        assert store._spo_index is not None

    def test_triple_store_contiguous_arrays(self) -> None:
        """Arrays should be C-contiguous for SIMD optimization."""
        store = TripleStoreSoA(100)
        assert store.subjects.flags["C_CONTIGUOUS"]
        assert store.predicates.flags["C_CONTIGUOUS"]
        assert store.objects.flags["C_CONTIGUOUS"]


# ─────────────────────────── calculate_optimal_batch_size Tests ──────────────


class TestCalculateOptimalBatchSize:
    """Tests for calculate_optimal_batch_size."""

    def test_batch_size_minimum(self) -> None:
        """Batch size should be at least 64."""
        result = calculate_optimal_batch_size(n_features=1, cache_size_mb=1)
        assert result >= 64

    def test_batch_size_aligned_to_64(self) -> None:
        """Batch size should be aligned to 64 for SIMD."""
        result = calculate_optimal_batch_size(n_features=1100)
        assert result % 64 == 0

    def test_batch_size_scales_with_cache(self) -> None:
        """Larger cache should allow larger batch size."""
        small = calculate_optimal_batch_size(n_features=1100, cache_size_mb=8)
        large = calculate_optimal_batch_size(n_features=1100, cache_size_mb=32)
        assert large > small

    def test_batch_size_scales_with_dtype(self) -> None:
        """Larger dtype should reduce batch size."""
        float32 = calculate_optimal_batch_size(n_features=1100, dtype=np.dtype(np.float32))
        float64 = calculate_optimal_batch_size(n_features=1100, dtype=np.dtype(np.float64))
        assert float32 > float64  # float64 is 2x larger


# ─────────────────────────── generate_negative_samples Tests ─────────────────


class TestGenerateNegativeSamples:
    """Tests for generate_negative_samples."""

    def test_negative_samples_shape(self) -> None:
        """Output should have shape (num_negatives, 3)."""
        result = generate_negative_samples(
            num_negatives=10,
            num_entities=100,
            head_idx=5,
            tail_idx=10,
            rel_idx=2,
            seed=42,
        )
        assert result.shape == (10, 3)

    def test_negative_samples_relation_preserved(self) -> None:
        """Relation should be preserved in all negatives."""
        result = generate_negative_samples(
            num_negatives=10,
            num_entities=100,
            head_idx=5,
            tail_idx=10,
            rel_idx=2,
            seed=42,
        )
        assert np.all(result[:, 1] == 2)

    def test_negative_samples_deterministic(self) -> None:
        """Same seed should produce same samples."""
        result1 = generate_negative_samples(
            num_negatives=10,
            num_entities=100,
            head_idx=5,
            tail_idx=10,
            rel_idx=2,
            seed=42,
        )
        result2 = generate_negative_samples(
            num_negatives=10,
            num_entities=100,
            head_idx=5,
            tail_idx=10,
            rel_idx=2,
            seed=42,
        )
        assert np.array_equal(result1, result2)

    def test_negative_samples_different_seeds(self) -> None:
        """Different seeds should produce different samples."""
        result1 = generate_negative_samples(
            num_negatives=10,
            num_entities=100,
            head_idx=5,
            tail_idx=10,
            rel_idx=2,
            seed=42,
        )
        result2 = generate_negative_samples(
            num_negatives=10,
            num_entities=100,
            head_idx=5,
            tail_idx=10,
            rel_idx=2,
            seed=123,
        )
        assert not np.array_equal(result1, result2)


# ─────────────────────────── generate_emu_noise Tests ───────────────────────


class TestGenerateEmuNoise:
    """Tests for generate_emu_noise."""

    def test_emu_noise_shape(self) -> None:
        """Output should have shape (num_samples, embedding_dim)."""
        result = generate_emu_noise(embedding_dim=128, num_samples=10, seed=42)
        assert result.shape == (10, 128)

    def test_emu_noise_dtype(self) -> None:
        """Output should be float32."""
        result = generate_emu_noise(embedding_dim=128, num_samples=10, seed=42)
        assert result.dtype == np.float32

    def test_emu_noise_scale(self) -> None:
        """Noise should be scaled by perturbation_scale."""
        result_small = generate_emu_noise(
            embedding_dim=128, num_samples=100, perturbation_scale=0.1, seed=42
        )
        result_large = generate_emu_noise(
            embedding_dim=128, num_samples=100, perturbation_scale=1.0, seed=42
        )
        assert np.std(result_large) > np.std(result_small)

    def test_emu_noise_deterministic(self) -> None:
        """Same seed should produce same noise."""
        result1 = generate_emu_noise(embedding_dim=128, num_samples=10, seed=42)
        result2 = generate_emu_noise(embedding_dim=128, num_samples=10, seed=42)
        assert np.allclose(result1, result2)


# ─────────────────────────── get_numba_diagnostics Tests ─────────────────────


class TestGetNumbaDiagnostics:
    """Tests for get_numba_diagnostics."""

    def test_diagnostics_returns_dict(self) -> None:
        """Should return a dictionary."""
        result = get_numba_diagnostics()
        assert isinstance(result, dict)

    def test_diagnostics_has_available_key(self) -> None:
        """Should have 'available' key."""
        result = get_numba_diagnostics()
        assert "available" in result

    def test_diagnostics_has_version_key(self) -> None:
        """Should have 'version' key."""
        result = get_numba_diagnostics()
        assert "version" in result

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_diagnostics_with_numba(self) -> None:
        """With Numba, should have additional keys."""
        result = get_numba_diagnostics()
        assert result["available"] is True
        assert "threading_layer" in result
        assert "num_threads" in result


# ─────────────────────────── find_matching_triples_accelerated Tests ─────────


class TestFindMatchingTriplesAccelerated:
    """Tests for find_matching_triples_accelerated."""

    def test_find_matching_basic(self) -> None:
        """Should find matching triples."""
        encoder = VocabularyEncoder()
        triples = [
            ("alice", "knows", "bob"),
            ("bob", "knows", "charlie"),
            ("alice", "likes", "dave"),
        ]
        pattern = {"predicate": "knows", "args": ["alice", "X"]}
        result = find_matching_triples_accelerated(pattern, triples, encoder)
        assert 0 in result  # alice knows bob
        assert 1 not in result  # bob knows charlie

    def test_find_matching_wildcard_predicate(self) -> None:
        """Wildcard predicate should match all."""
        encoder = VocabularyEncoder()
        triples = [("alice", "knows", "bob"), ("alice", "likes", "dave")]
        pattern = {"predicate": "*", "args": ["alice", "X"]}
        result = find_matching_triples_accelerated(pattern, triples, encoder)
        assert 0 in result
        assert 1 in result

    def test_find_matching_no_matches(self) -> None:
        """Should return empty list when no matches."""
        encoder = VocabularyEncoder()
        triples = [("alice", "knows", "bob"), ("bob", "knows", "charlie")]
        pattern = {"predicate": "likes", "args": ["alice", "X"]}
        result = find_matching_triples_accelerated(pattern, triples, encoder)
        assert len(result) == 0


# ─────────────────────────── binary_search_range Tests ───────────────────────


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba required for binary_search_range")
class TestBinarySearchRange:
    """Tests for binary_search_range."""

    def test_binary_search_found(self) -> None:
        """Should find range of matching elements."""
        arr = np.array([1, 2, 2, 2, 3, 4], dtype=np.int32)
        start, end = binary_search_range(arr, 2)
        assert start == 1
        assert end == 4

    def test_binary_search_single_element(self) -> None:
        """Should find single element."""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        start, end = binary_search_range(arr, 3)
        assert start == 2
        assert end == 3

    def test_binary_search_not_found(self) -> None:
        """Should return (0, 0) when not found."""
        arr = np.array([1, 2, 4, 5], dtype=np.int32)
        start, end = binary_search_range(arr, 3)
        assert start == 0
        assert end == 0

    def test_binary_search_first_element(self) -> None:
        """Should find first element."""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        start, end = binary_search_range(arr, 1)
        assert start == 0
        assert end == 1

    def test_binary_search_last_element(self) -> None:
        """Should find last element."""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        start, end = binary_search_range(arr, 5)
        assert start == 4
        assert end == 5

"""Tests for pff_rust numerical kernels.

Tests VocabularyEncoder, BloomFilter, TripleStoreSoA,
negative sampling, and EMU noise generation.
"""

from __future__ import annotations

import numpy as np
from pff_rust import (
    BloomFilter,
    TripleStoreSoA,
    VocabularyEncoder,
    batch_generate_negative_samples,
    find_unique_triples_mask,
    generate_emu_noise,
    generate_negative_samples,
)


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
        """Unknown index should return None."""
        encoder = VocabularyEncoder()
        result = encoder.decode_entity(9999)
        assert result is None

    def test_encode_relation_new(self) -> None:
        """New relations should get sequential indices."""
        encoder = VocabularyEncoder()
        idx1 = encoder.encode_relation("rel_a")
        idx2 = encoder.encode_relation("rel_b")
        assert idx1 == 0
        assert idx2 == 1

    def test_decode_relation(self) -> None:
        """Decode should return original relation."""
        encoder = VocabularyEncoder()
        idx = encoder.encode_relation("test_rel")
        assert encoder.decode_relation(idx) == "test_rel"

    def test_encode_triples(self) -> None:
        """encode_triples should return list of [s, p, o] arrays."""
        encoder = VocabularyEncoder()
        triples = [("a", "rel", "b"), ("c", "rel", "d")]
        encoded = encoder.encode_triples(triples)
        assert len(encoded) == 2
        assert len(encoded[0]) == 3

    def test_encode_pattern_variable(self) -> None:
        """Variables (uppercase-starting) should get high indices."""
        encoder = VocabularyEncoder()
        result = encoder.encode_pattern("X", "knows", "bob")
        assert result[0] >= 1_000_000

    def test_num_entities(self) -> None:
        """num_entities should track encoded count."""
        encoder = VocabularyEncoder()
        encoder.encode_entity("a")
        encoder.encode_entity("b")
        assert encoder.num_entities() == 2

    def test_num_relations(self) -> None:
        """num_relations should track encoded count."""
        encoder = VocabularyEncoder()
        encoder.encode_relation("r1")
        encoder.encode_relation("r2")
        assert encoder.num_relations() == 2


class TestBloomFilter:
    """Tests for BloomFilter."""

    def test_bloom_filter_add_and_check(self) -> None:
        """Added items should be detected."""
        bf = BloomFilter(1000, 0.01)
        bf.add(42)
        assert bf.might_contain(42) is True

    def test_bloom_filter_missing_item(self) -> None:
        """Non-added items should (usually) not be detected."""
        bf = BloomFilter(1000, 0.01)
        bf.add(42)
        missing_detected = sum(1 for i in range(1000, 2000) if bf.might_contain(i))
        assert missing_detected < 50


class TestTripleStoreSoA:
    """Tests for TripleStoreSoA."""

    def test_triple_store_load_and_len(self) -> None:
        """load_from_arrays should populate store."""
        store = TripleStoreSoA()
        s = np.array([1, 4], dtype=np.int32)
        p = np.array([2, 5], dtype=np.int32)
        o = np.array([3, 6], dtype=np.int32)
        store.load_from_arrays(s, p, o)
        assert store.len() == 2

    def test_triple_store_get_arrays(self) -> None:
        """get_subjects/predicates/objects should return loaded arrays."""
        store = TripleStoreSoA()
        s = np.array([1, 4], dtype=np.int32)
        p = np.array([2, 5], dtype=np.int32)
        o = np.array([3, 6], dtype=np.int32)
        store.load_from_arrays(s, p, o)
        assert np.array_equal(store.get_subjects(), s)
        assert np.array_equal(store.get_predicates(), p)
        assert np.array_equal(store.get_objects(), o)

    def test_triple_store_indexes(self) -> None:
        """SPO/POS/OSP indexes should be built on load."""
        store = TripleStoreSoA()
        s = np.array([5, 1, 3], dtype=np.int32)
        p = np.array([2, 2, 2], dtype=np.int32)
        o = np.array([3, 6, 1], dtype=np.int32)
        store.load_from_arrays(s, p, o)
        spo = store.get_spo_index()
        assert len(spo) == 3

    def test_find_matching_wildcard(self) -> None:
        """find_matching with -1 wildcard should match all."""
        store = TripleStoreSoA()
        s = np.array([1, 2, 1], dtype=np.int32)
        p = np.array([10, 10, 20], dtype=np.int32)
        o = np.array([3, 4, 5], dtype=np.int32)
        store.load_from_arrays(s, p, o)
        result = store.find_matching(1, -1, -1)
        assert result.shape[0] == 2

    def test_find_matching_exact(self) -> None:
        """find_matching with exact values should filter correctly."""
        store = TripleStoreSoA()
        s = np.array([1, 2, 1], dtype=np.int32)
        p = np.array([10, 10, 20], dtype=np.int32)
        o = np.array([3, 4, 5], dtype=np.int32)
        store.load_from_arrays(s, p, o)
        result = store.find_matching(1, 10, 3)
        assert result.shape[0] == 1


class TestGenerateNegativeSamples:
    """Tests for generate_negative_samples."""

    def test_negative_samples_shape(self) -> None:
        """Output should have correct shape."""
        h = np.array([5], dtype=np.int64)
        r = np.array([2], dtype=np.int64)
        t = np.array([10], dtype=np.int64)
        result = generate_negative_samples(h, r, t, 100, 10, 42)
        assert result.shape == (10, 3)

    def test_negative_samples_relation_preserved(self) -> None:
        """Relation should be preserved in all negatives."""
        h = np.array([5], dtype=np.int64)
        r = np.array([2], dtype=np.int64)
        t = np.array([10], dtype=np.int64)
        result = generate_negative_samples(h, r, t, 100, 10, 42)
        assert np.all(result[:, 1] == 2)

    def test_negative_samples_deterministic(self) -> None:
        """Same seed should produce same samples."""
        h = np.array([5], dtype=np.int64)
        r = np.array([2], dtype=np.int64)
        t = np.array([10], dtype=np.int64)
        result1 = generate_negative_samples(h, r, t, 100, 10, 42)
        result2 = generate_negative_samples(h, r, t, 100, 10, 42)
        assert np.array_equal(result1, result2)

    def test_negative_samples_different_seeds(self) -> None:
        """Different seeds should produce different samples."""
        h = np.array([5], dtype=np.int64)
        r = np.array([2], dtype=np.int64)
        t = np.array([10], dtype=np.int64)
        result1 = generate_negative_samples(h, r, t, 100, 10, 42)
        result2 = generate_negative_samples(h, r, t, 100, 10, 123)
        assert not np.array_equal(result1, result2)


class TestBatchGenerateNegativeSamples:
    """Tests for batch_generate_negative_samples."""

    def test_batch_shape(self) -> None:
        """Batch output should have correct shape."""
        h = np.array([1, 2, 3], dtype=np.int64)
        r = np.array([10, 10, 10], dtype=np.int64)
        t = np.array([5, 6, 7], dtype=np.int64)
        result = batch_generate_negative_samples(h, r, t, 5, 100, 42)
        assert result.shape == (15, 3)

    def test_batch_deterministic(self) -> None:
        """Same seed should produce same batch."""
        h = np.array([1, 2, 3], dtype=np.int64)
        r = np.array([10, 10, 10], dtype=np.int64)
        t = np.array([5, 6, 7], dtype=np.int64)
        r1 = batch_generate_negative_samples(h, r, t, 5, 100, 42)
        r2 = batch_generate_negative_samples(h, r, t, 5, 100, 42)
        assert np.array_equal(r1, r2)


class TestFindUniqueTriplesMask:
    """Tests for find_unique_triples_mask."""

    def test_all_unique(self) -> None:
        """All-unique triples should produce all-True mask."""
        h = np.array([1, 2, 3], dtype=np.int64)
        r = np.array([1, 1, 1], dtype=np.int64)
        t = np.array([4, 5, 6], dtype=np.int64)
        mask = find_unique_triples_mask(h, r, t)
        assert np.all(mask)

    def test_duplicates(self) -> None:
        """Duplicate triples should be marked False."""
        h = np.array([1, 1, 2], dtype=np.int64)
        r = np.array([1, 1, 1], dtype=np.int64)
        t = np.array([3, 3, 4], dtype=np.int64)
        mask = find_unique_triples_mask(h, r, t)
        assert mask[0] == True  # noqa: E712
        assert mask[1] == False  # noqa: E712
        assert mask[2] == True  # noqa: E712

    def test_empty(self) -> None:
        """Empty arrays should return empty mask."""
        h = np.array([], dtype=np.int64)
        r = np.array([], dtype=np.int64)
        t = np.array([], dtype=np.int64)
        mask = find_unique_triples_mask(h, r, t)
        assert len(mask) == 0


class TestGenerateEmuNoise:
    """Tests for generate_emu_noise."""

    def test_emu_noise_shape(self) -> None:
        """Output should have shape (num_samples, embedding_dim)."""
        result = generate_emu_noise(128, 10, 0.1, 42)
        assert result.shape == (10, 128)

    def test_emu_noise_dtype(self) -> None:
        """Output should be float32."""
        result = generate_emu_noise(128, 10, 0.1, 42)
        assert result.dtype == np.float32

    def test_emu_noise_scale(self) -> None:
        """Noise should be scaled by perturbation_scale."""
        result_small = generate_emu_noise(128, 100, 0.1, 42)
        result_large = generate_emu_noise(128, 100, 1.0, 42)
        assert np.std(result_large) > np.std(result_small)

    def test_emu_noise_deterministic(self) -> None:
        """Same seed should produce same noise."""
        result1 = generate_emu_noise(128, 10, 0.1, 42)
        result2 = generate_emu_noise(128, 10, 0.1, 42)
        assert np.allclose(result1, result2)

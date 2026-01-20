"""
Tests for pff.shared.hash module.

Tests ensure determinism and correctness of stable hash functions.
"""

import pytest

from pff.shared.hash import hash_64bit, hash_bytes, hash_tuple, stable_hash


class TestStableHash:
    """Test suite for stable_hash function."""

    def test_string_hash_determinism(self):
        """Test that string hashing is deterministic across calls."""
        test_string = "test_entity_123"
        hash1 = stable_hash(test_string)
        hash2 = stable_hash(test_string)
        assert hash1 == hash2, "Same string should produce same hash"

    def test_string_hash_different_values(self):
        """Test that different strings produce different hashes."""
        hash1 = stable_hash("entity_a")
        hash2 = stable_hash("entity_b")
        assert hash1 != hash2, "Different strings should produce different hashes"

    def test_int_hash_determinism(self):
        """Test that integer hashing is deterministic."""
        test_int = 42
        hash1 = stable_hash(test_int)
        hash2 = stable_hash(test_int)
        assert hash1 == hash2

    def test_tuple_hash_determinism(self):
        """Test that tuple hashing is deterministic."""
        test_tuple = ("subject", "predicate", "object")
        hash1 = stable_hash(test_tuple)
        hash2 = stable_hash(test_tuple)
        assert hash1 == hash2

    def test_bytes_hash_determinism(self):
        """Test that bytes hashing is deterministic."""
        test_bytes = b"test_data"
        hash1 = hash_bytes(test_bytes)
        hash2 = hash_bytes(test_bytes)
        assert hash1 == hash2

    def test_bytes_vs_string_same_content(self):
        """Test that bytes and string with same content produce same hash."""
        text = "test_data"
        string_hash = stable_hash(text)
        bytes_hash = hash_bytes(text.encode("utf-8"))
        assert string_hash == bytes_hash

    def test_hash_64bit_range(self):
        """Test that hash_64bit returns values within 64-bit range."""
        test_value = "test_entity"
        hash_val = hash_64bit(test_value)
        # 64-bit unsigned integer max value is 2^64 - 1
        assert 0 <= hash_val < 2**64
        assert hash_val == stable_hash(test_value, truncate=16)

    def test_hash_tuple_function(self):
        """Test hash_tuple utility function."""
        items = ("head", "relation", "tail")
        hash1 = hash_tuple(items)
        hash2 = hash_tuple(items)
        assert hash1 == hash2

    def test_empty_string_hash(self):
        """Test that empty string produces a consistent hash."""
        hash1 = stable_hash("")
        hash2 = stable_hash("")
        assert hash1 == hash2

    def test_unicode_string_hash(self):
        """Test that Unicode strings are hashed correctly."""
        unicode_str = "test_ñ_entities_测试_"
        hash1 = stable_hash(unicode_str)
        hash2 = stable_hash(unicode_str)
        assert hash1 == hash2

    def test_list_hash_determinism(self):
        """Test that lists produce deterministic hashes."""
        test_list = [1, 2, 3, "test"]
        hash1 = stable_hash(test_list)
        hash2 = stable_hash(test_list)
        assert hash1 == hash2

    def test_dict_hash_determinism(self):
        """Test that dictionaries produce deterministic hashes."""
        test_dict = {"key1": "value1", "key2": 42}
        hash1 = stable_hash(test_dict)
        hash2 = stable_hash(test_dict)
        assert hash1 == hash2

    def test_hash_algorithm_parameter(self):
        """Test that different hash algorithms work."""
        test_value = "test_data"
        hash_sha1 = stable_hash(test_value, algorithm="sha1")
        hash_md5 = stable_hash(test_value, algorithm="md5")
        assert hash_sha1 != hash_md5, "Different algorithms should produce different hashes"
        # But same algorithm should be deterministic
        hash_sha1_2 = stable_hash(test_value, algorithm="sha1")
        assert hash_sha1 == hash_sha1_2

    def test_hash_truncation(self):
        """Test that hash truncation works correctly."""
        test_value = "test_data_for_truncation"
        stable_hash(test_value, truncate=None)
        truncated_hash = stable_hash(test_value, truncate=16)
        import hashlib

        serialized = str(test_value).encode("utf-8")
        hasher = hashlib.new("sha1")
        hasher.update(serialized)
        full_hash_hex = hasher.hexdigest()
        truncated_hash_hex = full_hash_hex[:16]
        expected_truncated = int(truncated_hash_hex, 16)
        assert truncated_hash == expected_truncated


if __name__ == "__main__":
    pytest.main([__file__])

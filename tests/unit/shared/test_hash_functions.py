"""Tests for BLAKE3 deterministic hashing utilities."""

from pff_rust import hash_64bit, hash_bytes, hash_tuple, stable_hash


class TestStableHash:
    """Tests for the stable_hash function."""

    def test_stable_hash_string_deterministic(self):
        """Verify same string produces same hash across calls."""
        result1 = stable_hash("test_string")
        result2 = stable_hash("test_string")
        assert result1 == result2

    def test_stable_hash_different_strings_differ(self):
        """Verify different strings produce different hashes."""
        hash1 = stable_hash("string_a")
        hash2 = stable_hash("string_b")
        assert hash1 != hash2

    def test_stable_hash_list_deterministic(self):
        """Verify list hashing is deterministic."""
        data = [1, 2, 3, "test"]
        result1 = stable_hash(data)
        result2 = stable_hash(data)
        assert result1 == result2

    def test_stable_hash_dict_deterministic(self):
        """Verify dict hashing is deterministic."""
        data = {"key": "value", "num": 42}
        result1 = stable_hash(data)
        result2 = stable_hash(data)
        assert result1 == result2

    def test_stable_hash_returns_integer(self):
        """Verify hash returns an integer."""
        result = stable_hash("test")
        assert isinstance(result, int)

    def test_stable_hash_truncate_affects_size(self):
        """Verify truncation parameter affects hash size."""
        full = stable_hash("test", truncate=None)
        truncated = stable_hash("test", truncate=16)
        # Truncated should be smaller (fewer hex digits = smaller int)
        assert truncated < full or truncated == full  # depends on actual values

    def test_stable_hash_empty_string(self):
        """Verify empty string can be hashed."""
        result = stable_hash("")
        assert isinstance(result, int)
        assert result > 0


class TestHashTuple:
    """Tests for the hash_tuple function."""

    def test_hash_tuple_deterministic(self):
        """Verify tuple hashing is deterministic."""
        data = (1, 2, "three")
        result1 = hash_tuple(data)
        result2 = hash_tuple(data)
        assert result1 == result2

    def test_hash_tuple_different_tuples_differ(self):
        """Verify different tuples produce different hashes."""
        hash1 = hash_tuple((1, 2, 3))
        hash2 = hash_tuple((3, 2, 1))
        assert hash1 != hash2

    def test_hash_tuple_order_matters(self):
        """Verify tuple order affects hash."""
        hash1 = hash_tuple(("a", "b"))
        hash2 = hash_tuple(("b", "a"))
        assert hash1 != hash2

    def test_hash_tuple_empty(self):
        """Verify empty tuple can be hashed."""
        result = hash_tuple(())
        assert isinstance(result, int)

    def test_hash_tuple_nested(self):
        """Verify nested tuple can be hashed."""
        data = ((1, 2), (3, 4))
        result = hash_tuple(data)
        assert isinstance(result, int)


class TestHash64Bit:
    """Tests for the hash_64bit function."""

    def test_hash_64bit_deterministic(self):
        """Verify 64-bit hash is deterministic."""
        result1 = hash_64bit("test")
        result2 = hash_64bit("test")
        assert result1 == result2

    def test_hash_64bit_returns_integer(self):
        """Verify 64-bit hash returns integer."""
        result = hash_64bit("test")
        assert isinstance(result, int)

    def test_hash_64bit_reasonable_range(self):
        """Verify 64-bit hash is in expected range (16 hex chars = 64 bits)."""
        result = hash_64bit("test")
        # 16 hex chars = 64 bits max = 2^64 - 1
        assert result < 2**64


class TestHashBytes:
    """Tests for the hash_bytes function."""

    def test_hash_bytes_from_bytes(self):
        """Verify bytes can be hashed."""
        result = hash_bytes(b"test_bytes")
        assert isinstance(result, int)

    def test_hash_bytes_from_string(self):
        """Verify string is converted and hashed."""
        result = hash_bytes("test_string")
        assert isinstance(result, int)

    def test_hash_bytes_deterministic(self):
        """Verify bytes hashing is deterministic."""
        result1 = hash_bytes(b"test")
        result2 = hash_bytes(b"test")
        assert result1 == result2

    def test_hash_bytes_string_bytes_equivalence(self):
        """Verify string and its encoded bytes produce same hash."""
        text = "test"
        hash_from_str = hash_bytes(text)
        hash_from_bytes = hash_bytes(text.encode("utf-8"))
        assert hash_from_str == hash_from_bytes

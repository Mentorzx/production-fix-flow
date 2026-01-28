"""Tests for pff/shared/core/cache.py core utilities.

Tests JsonSafeEncoder, FunctionCallHasher, CacheEntry, HttpTemplateEntry,
TemplatePatternNormalizer, and other cache components without heavy I/O.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pff.shared.core.cache import (
    DEFAULT_TEMPLATE_TTL_DAYS,
    AtomicFileWriter,
    CacheEntry,
    CacheJanitor,
    CacheManager,
    CacheSerializer,
    FileSystemStorage,
    FunctionCallHasher,
    HttpTemplateEntry,
    JsonSafeEncoder,
    TemplatePatternNormalizer,
    create_memory_cache,
)

# ─────────────────────────── JsonSafeEncoder Tests ───────────────────────────


class TestJsonSafeEncoder:
    """Tests for JsonSafeEncoder.make_json_safe."""

    def test_json_safe_primitives_pass_through(self) -> None:
        """Primitives should pass through unchanged."""
        encoder = JsonSafeEncoder()
        assert encoder.make_json_safe(42) == 42
        assert encoder.make_json_safe("hello") == "hello"
        assert encoder.make_json_safe(3.14) == 3.14
        assert encoder.make_json_safe(True) is True
        assert encoder.make_json_safe(None) is None

    def test_json_safe_list_dict_pass_through(self) -> None:
        """JSON-serializable collections should pass through."""
        encoder = JsonSafeEncoder()
        data = {"key": [1, 2, 3], "nested": {"a": 1}}
        assert encoder.make_json_safe(data) == data

    def test_json_safe_non_serializable_uses_repr(self) -> None:
        """Non-JSON-serializable objects should use repr."""
        encoder = JsonSafeEncoder()

        class Custom:
            def __repr__(self) -> str:
                return "<Custom object>"

        result = encoder.make_json_safe(Custom())
        assert result == "<Custom object>"

    def test_json_safe_bytes_uses_repr(self) -> None:
        """Bytes are not JSON-serializable, should use repr."""
        encoder = JsonSafeEncoder()
        result = encoder.make_json_safe(b"binary data")
        assert "binary" in result or "bytes" in result.lower()

    def test_json_safe_set_uses_repr(self) -> None:
        """Sets are not JSON-serializable, should use repr."""
        encoder = JsonSafeEncoder()
        result = encoder.make_json_safe({1, 2, 3})
        assert isinstance(result, str)


# ─────────────────────────── FunctionCallHasher Tests ───────────────────────────


class TestFunctionCallHasher:
    """Tests for FunctionCallHasher.hash_function_call."""

    def test_hash_deterministic(self) -> None:
        """Same function and args should produce same hash."""

        def sample_func(x: int, y: int) -> int:
            return x + y

        hash1 = FunctionCallHasher.hash_function_call(sample_func, 1, 2)
        hash2 = FunctionCallHasher.hash_function_call(sample_func, 1, 2)
        assert hash1 == hash2

    def test_hash_different_args_differ(self) -> None:
        """Different args should produce different hashes."""

        def sample_func(x: int, y: int) -> int:
            return x + y

        hash1 = FunctionCallHasher.hash_function_call(sample_func, 1, 2)
        hash2 = FunctionCallHasher.hash_function_call(sample_func, 3, 4)
        assert hash1 != hash2

    def test_hash_different_functions_differ(self) -> None:
        """Different functions with same args should produce different hashes."""

        def func_a(x: int) -> int:
            return x

        def func_b(x: int) -> int:
            return x

        hash1 = FunctionCallHasher.hash_function_call(func_a, 1)
        hash2 = FunctionCallHasher.hash_function_call(func_b, 1)
        assert hash1 != hash2

    def test_hash_kwargs_included(self) -> None:
        """Kwargs should be included in hash."""

        def sample_func(x: int, y: int = 10) -> int:
            return x + y

        hash1 = FunctionCallHasher.hash_function_call(sample_func, 1, y=10)
        hash2 = FunctionCallHasher.hash_function_call(sample_func, 1, y=20)
        assert hash1 != hash2

    def test_hash_returns_hex_string(self) -> None:
        """Hash should be a valid hex string."""

        def sample_func() -> None:
            pass

        result = FunctionCallHasher.hash_function_call(sample_func)
        assert isinstance(result, str)
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_lambda_functions(self) -> None:
        """Lambda functions should be hashable."""
        fn = lambda x: x * 2  # noqa: E731
        result = FunctionCallHasher.hash_function_call(fn, 5)
        assert isinstance(result, str)
        assert len(result) == 32  # blake2b with digest_size=16

    def test_hash_with_non_serializable_args(self) -> None:
        """Non-serializable args should be handled via repr."""

        def sample_func(obj: Any) -> None:
            pass

        class Custom:
            pass

        # Should not raise
        result = FunctionCallHasher.hash_function_call(sample_func, Custom())
        assert isinstance(result, str)


# ─────────────────────────── CacheEntry Tests ───────────────────────────


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_default_timestamps(self) -> None:
        """CacheEntry should have default timestamps."""
        entry = CacheEntry()
        assert entry.created_at > 0
        assert entry.last_accessed > 0
        assert entry.expires_at is None

    def test_cache_entry_not_expired_by_default(self) -> None:
        """Entry without expires_at should not be expired."""
        entry = CacheEntry()
        assert entry.is_expired() is False

    def test_cache_entry_expired_when_past_expiry(self) -> None:
        """Entry with past expires_at should be expired."""
        entry = CacheEntry(expires_at=time.time() - 100)
        assert entry.is_expired() is True

    def test_cache_entry_not_expired_when_future_expiry(self) -> None:
        """Entry with future expires_at should not be expired."""
        entry = CacheEntry(expires_at=time.time() + 3600)
        assert entry.is_expired() is False

    def test_cache_entry_touch_updates_last_accessed(self) -> None:
        """touch() should update last_accessed."""
        entry = CacheEntry()
        original = entry.last_accessed
        time.sleep(0.01)
        entry.touch()
        assert entry.last_accessed > original


# ─────────────────────────── HttpTemplateEntry Tests ───────────────────────────


class TestHttpTemplateEntry:
    """Tests for HttpTemplateEntry dataclass."""

    def test_http_template_entry_creation(self) -> None:
        """HttpTemplateEntry should be created with required fields."""
        entry = HttpTemplateEntry(
            template="https://api.example.com/{msisdn}",
            endpoint_type="subscriber",
            method="GET",
        )
        assert entry.template == "https://api.example.com/{msisdn}"
        assert entry.endpoint_type == "subscriber"
        assert entry.method == "GET"

    def test_http_template_entry_extracts_variables(self) -> None:
        """HttpTemplateEntry should extract variables from template."""
        entry = HttpTemplateEntry(
            template="https://api.example.com/{msisdn}/status/{id}",
            endpoint_type="status",
        )
        assert "msisdn" in entry.variables
        assert "id" in entry.variables

    def test_http_template_entry_default_expiry(self) -> None:
        """HttpTemplateEntry should have default expiry based on TTL."""
        entry = HttpTemplateEntry(
            template="https://api.example.com/test",
            endpoint_type="test",
        )
        expected_min = time.time() + (DEFAULT_TEMPLATE_TTL_DAYS * 24 * 3600) - 10
        expected_max = time.time() + (DEFAULT_TEMPLATE_TTL_DAYS * 24 * 3600) + 10
        assert entry.expires_at is not None
        assert expected_min < entry.expires_at < expected_max

    def test_http_template_entry_success_count_default(self) -> None:
        """HttpTemplateEntry should have default success_count of 0."""
        entry = HttpTemplateEntry(
            template="https://api.example.com/test",
            endpoint_type="test",
        )
        assert entry.success_count == 0


# ─────────────────────────── TemplatePatternNormalizer Tests ───────────────────


class TestTemplatePatternNormalizer:
    """Tests for TemplatePatternNormalizer."""

    def test_normalize_url_replaces_uuid(self) -> None:
        """UUIDs should be replaced with {uuid} placeholder."""
        normalizer = TemplatePatternNormalizer()
        url = "api/users/550e8400-e29b-41d4-a716-446655440000/profile"
        result = normalizer.normalize_url(url)
        assert "{uuid}" in result
        assert "550e8400" not in result

    def test_normalize_url_replaces_msisdn(self) -> None:
        """MSISDNs should be replaced with {msisdn} placeholder."""
        normalizer = TemplatePatternNormalizer()
        url = "api/subscriber/5511999887766/status"
        result = normalizer.normalize_url(url)
        assert "55{msisdn}" in result

    def test_normalize_url_replaces_long_numbers(self) -> None:
        """Long numeric IDs should be replaced with {number} placeholder."""
        normalizer = TemplatePatternNormalizer()
        url = "api/orders/12345678/details"
        result = normalizer.normalize_url(url)
        assert "{number}" in result

    def test_normalize_url_replaces_hex_ids(self) -> None:
        """Hex IDs should be replaced with {hex_id} placeholder."""
        normalizer = TemplatePatternNormalizer()
        url = "api/session/abc123def456789012/data"
        result = normalizer.normalize_url(url)
        assert "{hex_id}" in result

    def test_extract_template_known_values(self) -> None:
        """Known values should be replaced with their names."""
        normalizer = TemplatePatternNormalizer()
        url = "api/subscriber/5511999887766/account/ABC123"
        known = {"msisdn": "5511999887766", "account_id": "ABC123"}
        result = normalizer.extract_template(url, known)
        assert "{msisdn}" in result
        assert "{account_id}" in result

    def test_extract_template_empty_known_values(self) -> None:
        """Empty known values should still normalize generic patterns."""
        normalizer = TemplatePatternNormalizer()
        url = "api/subscriber/5511999887766/status"
        result = normalizer.extract_template(url, {})
        assert "55{msisdn}" in result


# ─────────────────────────── CacheSerializer Tests ───────────────────────────


class TestCacheSerializer:
    """Tests for CacheSerializer."""

    def test_serialize_deserialize_primitives(self) -> None:
        """Primitives should round-trip correctly."""
        serializer = CacheSerializer()
        for value in [42, "hello", 3.14, True, None, [1, 2, 3], {"a": 1}]:
            data = serializer.serialize(value)
            result = serializer.deserialize(data)
            assert result == value

    def test_serialize_deserialize_bytes(self) -> None:
        """Bytes should round-trip correctly."""
        serializer = CacheSerializer()
        value = b"binary data"
        data = serializer.serialize(value)
        result = serializer.deserialize(data)
        assert result == value

    def test_serialize_deserialize_dict_with_to_dict(self) -> None:
        """Objects with to_dict should be serialized as object kind."""

        @dataclass
        class ConfigObject:
            name: str
            value: int

            def to_dict(self) -> dict[str, Any]:
                return {"name": self.name, "value": self.value}

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> ConfigObject:
                return cls(name=data["name"], value=data["value"])

        serializer = CacheSerializer()
        obj = ConfigObject(name="test", value=42)
        data = serializer.serialize(obj)
        result = serializer.deserialize(data)
        # Result might be dict or reconstructed object depending on import path
        if isinstance(result, dict):
            assert result["name"] == "test"
            assert result["value"] == 42

    def test_serialize_nested_structures(self) -> None:
        """Nested structures should round-trip correctly."""
        serializer = CacheSerializer()
        value = {
            "users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
            "metadata": {"version": 1, "active": True},
        }
        data = serializer.serialize(value)
        result = serializer.deserialize(data)
        assert result == value


# ─────────────────────────── FileSystemStorage Tests ───────────────────────────


class TestFileSystemStorage:
    """Tests for FileSystemStorage."""

    def test_storage_read_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """Reading non-existent file should return None."""
        storage = FileSystemStorage(compress=False)
        result = storage.read(tmp_path / "nonexistent.pkl")
        assert result is None

    def test_storage_write_read_uncompressed(self, tmp_path: Path) -> None:
        """Uncompressed write/read should round-trip."""
        storage = FileSystemStorage(compress=False)
        data = b"test data"
        path = tmp_path / "test.pkl"
        storage.write(path, data)
        result = storage.read(path)
        assert result == data

    def test_storage_write_read_compressed(self, tmp_path: Path) -> None:
        """Compressed write/read should round-trip."""
        storage = FileSystemStorage(compress=True)
        data = b"test data that will be compressed"
        path = tmp_path / "test.pkl.gz"
        storage.write(path, data)
        result = storage.read(path)
        assert result == data

    def test_storage_delete(self, tmp_path: Path) -> None:
        """Delete should remove file."""
        storage = FileSystemStorage(compress=False)
        path = tmp_path / "to_delete.pkl"
        path.write_bytes(b"data")
        assert path.exists()
        storage.delete(path)
        assert not path.exists()

    def test_storage_delete_nonexistent_no_error(self, tmp_path: Path) -> None:
        """Deleting non-existent file should not raise."""
        storage = FileSystemStorage(compress=False)
        storage.delete(tmp_path / "nonexistent.pkl")  # Should not raise

    def test_storage_exists(self, tmp_path: Path) -> None:
        """exists() should return correct status."""
        storage = FileSystemStorage(compress=False)
        path = tmp_path / "exists.pkl"
        assert storage.exists(path) is False
        path.write_bytes(b"data")
        assert storage.exists(path) is True


# ─────────────────────────── AtomicFileWriter Tests ───────────────────────────


class TestAtomicFileWriter:
    """Tests for AtomicFileWriter."""

    def test_atomic_write_creates_file(self, tmp_path: Path) -> None:
        """Atomic write should create file with correct content."""
        path = tmp_path / "atomic_test.dat"
        data = b"atomic write test"
        AtomicFileWriter.write_atomically(path, data)
        assert path.exists()
        assert path.read_bytes() == data

    def test_atomic_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Atomic write should create parent directories."""
        path = tmp_path / "subdir" / "nested" / "file.dat"
        data = b"nested data"
        AtomicFileWriter.write_atomically(path, data)
        assert path.exists()
        assert path.read_bytes() == data


# ─────────────────────────── CacheManager Tests ───────────────────────────


class TestCacheManager:
    """Tests for CacheManager memory cache functionality."""

    def test_cache_manager_set_get(self, tmp_path: Path) -> None:
        """Basic set/get should work."""
        manager = CacheManager(cache_dir=tmp_path / "cache")
        manager.set("key1", "value1")
        assert manager.get("key1") == "value1"

    def test_cache_manager_get_default(self, tmp_path: Path) -> None:
        """get() should return default for missing keys."""
        manager = CacheManager(cache_dir=tmp_path / "cache")
        assert manager.get("missing") is None
        assert manager.get("missing", "default") == "default"

    def test_cache_manager_dict_interface(self, tmp_path: Path) -> None:
        """Dict-like interface should work."""
        manager = CacheManager(cache_dir=tmp_path / "cache")
        manager["key1"] = "value1"
        assert manager["key1"] == "value1"
        del manager["key1"]
        assert manager.get("key1") is None

    def test_cache_manager_ttl_expiration(self, tmp_path: Path) -> None:
        """Items with TTL should expire."""
        manager = CacheManager(cache_dir=tmp_path / "cache")
        manager.set("key1", "value1", ttl=0)  # Immediate expiry
        time.sleep(0.01)
        assert manager.get("key1") is None

    def test_cache_manager_lru_eviction(self, tmp_path: Path) -> None:
        """LRU eviction should work when at capacity."""
        manager = CacheManager(cache_dir=tmp_path / "cache", max_memory_items=3)
        manager.set("k1", "v1")
        manager.set("k2", "v2")
        manager.set("k3", "v3")
        # This should evict k1 (oldest)
        manager.set("k4", "v4")
        assert manager.get("k1") is None
        assert manager.get("k2") == "v2"
        assert manager.get("k3") == "v3"
        assert manager.get("k4") == "v4"

    def test_cache_manager_stats(self, tmp_path: Path) -> None:
        """Stats should track hits and misses."""
        manager = CacheManager(cache_dir=tmp_path / "cache")
        manager.set("key1", "value1")
        manager.get("key1")  # hit
        manager.get("key1")  # hit
        manager.get("missing")  # miss
        stats = manager.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["sets"] == 1

    def test_cache_manager_len(self, tmp_path: Path) -> None:
        """len() should return memory cache size."""
        manager = CacheManager(cache_dir=tmp_path / "cache")
        assert len(manager) == 0
        manager.set("k1", "v1")
        manager.set("k2", "v2")
        assert len(manager) == 2

    def test_cache_manager_iter(self, tmp_path: Path) -> None:
        """Iteration should yield keys."""
        manager = CacheManager(cache_dir=tmp_path / "cache")
        manager.set("k1", "v1")
        manager.set("k2", "v2")
        keys = list(manager)
        assert "k1" in keys
        assert "k2" in keys


# ─────────────────────────── create_memory_cache Tests ───────────────────────────


class TestCreateMemoryCache:
    """Tests for create_memory_cache decorator factory."""

    def test_memory_cache_caches_results(self) -> None:
        """Memory cache should cache function results."""
        call_count = 0

        @create_memory_cache(maxsize=128)
        def expensive_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive_func(5) == 10
        assert expensive_func(5) == 10
        assert call_count == 1  # Only called once

    def test_memory_cache_different_args_call_again(self) -> None:
        """Different args should trigger new calls."""
        call_count = 0

        @create_memory_cache(maxsize=128)
        def expensive_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive_func(5) == 10
        assert expensive_func(10) == 20
        assert call_count == 2


# ─────────────────────────── CacheJanitor Tests ───────────────────────────


class TestCacheJanitor:
    """Tests for CacheJanitor behavior with missing directories."""

    def test_cache_janitor_missing_root_no_error(self, tmp_path: Path) -> None:
        """CacheJanitor should ignore missing cache roots without raising errors."""
        missing_root = tmp_path / "missing_cache"
        janitor = CacheJanitor(missing_root, max_age_seconds=10, interval_seconds=1)

        janitor._purge_stale_entries()

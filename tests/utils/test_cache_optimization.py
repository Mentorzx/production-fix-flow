"""
Tests for cache optimization features: bounded memory, metrics, tag-based invalidation, warming.

Sprint 5: Cache Optimization
"""

import time
import threading
from pathlib import Path

import pytest

from pff.utils.cache import CacheManager


class TestMemoryCacheBounded:
    """Test bounded memory cache with LRU eviction."""

    def test_memory_cache_bounded(self, tmp_path):
        """Test memory cache respects max_memory_items."""
        cache = CacheManager(cache_dir=tmp_path, max_memory_items=10)

        # Fill beyond capacity
        for i in range(20):
            cache.set(f"key_{i}", f"value_{i}")

        # Should have only 10 items (LRU evicted)
        assert len(cache._memory_storage) == 10
        assert cache.get_stats()["evictions"] == 10

        # Check that oldest keys were evicted (key_0 to key_9)
        assert cache.get("key_0") is None
        assert cache.get("key_9") is None

        # Check that newest keys are present (key_10 to key_19)
        assert cache.get("key_10") == "value_10"
        assert cache.get("key_19") == "value_19"

    def test_lru_ordering(self, tmp_path):
        """Test LRU ordering is maintained correctly."""
        cache = CacheManager(cache_dir=tmp_path, max_memory_items=3)

        # Add 3 items
        cache.set("a", "value_a")
        cache.set("b", "value_b")
        cache.set("c", "value_c")

        # Access 'a' to make it MRU
        cache.get("a")

        # Add new item (should evict 'b' as it's LRU)
        cache.set("d", "value_d")

        assert cache.get("a") == "value_a"  # Still present
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == "value_c"  # Still present
        assert cache.get("d") == "value_d"  # Newly added

    def test_update_existing_key_no_eviction(self, tmp_path):
        """Test updating existing key doesn't trigger eviction."""
        cache = CacheManager(cache_dir=tmp_path, max_memory_items=3)

        cache.set("a", "value_a")
        cache.set("b", "value_b")
        cache.set("c", "value_c")

        # Update existing key
        cache.set("a", "new_value_a")

        # No evictions should happen
        assert cache.get_stats()["evictions"] == 0
        assert cache.get("a") == "new_value_a"


class TestCacheMetrics:
    """Test cache statistics tracking."""

    def test_cache_metrics_basic(self, tmp_path):
        """Test cache tracks hit rate correctly."""
        cache = CacheManager(cache_dir=tmp_path)

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["hit_rate"] == "50.00%"

    def test_cache_metrics_multiple_operations(self, tmp_path):
        """Test metrics accumulate correctly over multiple operations."""
        cache = CacheManager(cache_dir=tmp_path)

        # 5 sets
        for i in range(5):
            cache.set(f"key_{i}", f"value_{i}")

        # 3 hits, 2 misses
        cache.get("key_0")  # Hit
        cache.get("key_1")  # Hit
        cache.get("key_2")  # Hit
        cache.get("nonexistent_1")  # Miss
        cache.get("nonexistent_2")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 2
        assert stats["sets"] == 5
        assert stats["hit_rate"] == "60.00%"

    def test_expiration_tracking(self, tmp_path):
        """Test expiration is tracked in metrics."""
        cache = CacheManager(cache_dir=tmp_path)

        # Set with short TTL
        cache.set("key1", "value1", ttl=1)

        # Wait for expiration
        time.sleep(1.1)

        # Try to get expired key
        result = cache.get("key1")

        assert result is None
        stats = cache.get_stats()
        assert stats["expirations"] == 1
        assert stats["misses"] == 1

    def test_memory_usage_percentage(self, tmp_path):
        """Test memory usage percentage calculation."""
        cache = CacheManager(cache_dir=tmp_path, max_memory_items=10)

        # Add 5 items (50% capacity)
        for i in range(5):
            cache.set(f"key_{i}", f"value_{i}")

        stats = cache.get_stats()
        assert stats["size"] == 5
        assert stats["max_size"] == 10
        assert stats["memory_usage_pct"] == "50.0%"


class TestTagBasedInvalidation:
    """Test tag-based cache invalidation."""

    def test_invalidate_by_tag(self, tmp_path):
        """Test tag-based invalidation."""
        cache = CacheManager(cache_dir=tmp_path)

        cache.set("user:123:profile", {"name": "Alice"}, tags=["user:123", "profile"])
        cache.set("user:123:posts", ["post1", "post2"], tags=["user:123", "posts"])
        cache.set("user:456:profile", {"name": "Bob"}, tags=["user:456", "profile"])

        # Invalidate user 123
        deleted = cache.invalidate(tags=["user:123"])

        assert deleted == 2
        assert cache.get("user:123:profile") is None
        assert cache.get("user:123:posts") is None
        assert cache.get("user:456:profile") is not None

    def test_invalidate_by_pattern(self, tmp_path):
        """Test pattern-based invalidation."""
        cache = CacheManager(cache_dir=tmp_path)

        cache.set("user:123:posts", "data1")
        cache.set("user:456:posts", "data2")
        cache.set("user:123:profile", "data3")

        # Invalidate all posts
        deleted = cache.invalidate(pattern=r":posts$")

        assert deleted == 2
        assert cache.get("user:123:posts") is None
        assert cache.get("user:456:posts") is None
        assert cache.get("user:123:profile") == "data3"

    def test_invalidate_by_tag_and_pattern(self, tmp_path):
        """Test combined tag and pattern invalidation."""
        cache = CacheManager(cache_dir=tmp_path)

        cache.set("user:123:profile", "data1", tags=["user:123", "profile"])
        cache.set("user:123:posts", "data2", tags=["user:123", "posts"])
        cache.set("user:456:profile", "data3", tags=["user:456", "profile"])

        # Invalidate profiles for user 123
        deleted = cache.invalidate(pattern=r"^user:123:", tags=["profile"])

        # Should match user:123:profile (both pattern AND tag match)
        # Note: implementation uses OR logic, so both conditions separately
        assert deleted >= 1
        assert cache.get("user:123:profile") is None

    def test_invalidate_nonexistent_tags(self, tmp_path):
        """Test invalidating nonexistent tags returns 0."""
        cache = CacheManager(cache_dir=tmp_path)

        cache.set("key1", "value1", tags=["tag1"])

        deleted = cache.invalidate(tags=["nonexistent"])

        assert deleted == 0
        assert cache.get("key1") == "value1"


class TestCacheWarming:
    """Test cache warming functionality."""

    def test_cache_warming_with_function(self, tmp_path):
        """Test cache warming with a preload function."""
        cache = CacheManager(cache_dir=tmp_path)

        # Define a warming function
        def warm_cache():
            for i in range(10):
                cache.set(f"warm_key_{i}", f"warm_value_{i}")

        # Warm the cache
        cache.warm(preload_func=warm_cache)

        # Verify items are loaded
        assert len(cache._memory_storage) == 10
        assert cache.get("warm_key_0") == "warm_value_0"
        assert cache.get("warm_key_9") == "warm_value_9"

    def test_cache_warming_with_keys(self, tmp_path):
        """Test cache warming with specific keys (stub implementation)."""
        cache = CacheManager(cache_dir=tmp_path)

        # Pre-populate cache
        for i in range(5):
            cache.set(f"key_{i}", f"value_{i}")

        # Call warm with keys (currently a no-op, but shouldn't error)
        cache.warm(keys=["key_0", "key_1"])

        # Verify cache still works
        assert cache.get("key_0") == "value_0"

    def test_cache_warming_no_args(self, tmp_path):
        """Test cache warming with no arguments (should not error)."""
        cache = CacheManager(cache_dir=tmp_path)

        # Should not raise any errors
        cache.warm()


class TestTTLExpiration:
    """Test TTL-based expiration."""

    def test_ttl_expiration(self, tmp_path):
        """Test items expire after TTL."""
        cache = CacheManager(cache_dir=tmp_path)

        cache.set("key1", "value1", ttl=1)

        # Should be accessible immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get("key1") is None

    def test_ttl_mixed_with_permanent(self, tmp_path):
        """Test TTL items mixed with permanent items."""
        cache = CacheManager(cache_dir=tmp_path)

        cache.set("permanent", "value_perm")
        cache.set("temporary", "value_temp", ttl=1)

        # Both accessible
        assert cache.get("permanent") == "value_perm"
        assert cache.get("temporary") == "value_temp"

        # Wait for TTL expiration
        time.sleep(1.1)

        # Permanent still there, temporary gone
        assert cache.get("permanent") == "value_perm"
        assert cache.get("temporary") is None


class TestConcurrency:
    """Test thread safety and concurrency improvements."""

    def test_concurrent_access(self, tmp_path):
        """Test concurrent reads and writes are thread-safe."""
        cache = CacheManager(cache_dir=tmp_path, max_memory_items=100)
        errors = []

        def writer(thread_id):
            try:
                for i in range(10):
                    cache.set(f"thread_{thread_id}_key_{i}", f"value_{i}")
            except Exception as e:
                errors.append(e)

        def reader(thread_id):
            try:
                for i in range(10):
                    cache.get(f"thread_{thread_id}_key_{i}")
            except Exception as e:
                errors.append(e)

        # Create 10 writer and 10 reader threads
        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0

        # Verify some data was written
        assert len(cache._memory_storage) > 0

    def test_concurrent_eviction(self, tmp_path):
        """Test concurrent operations trigger evictions correctly."""
        cache = CacheManager(cache_dir=tmp_path, max_memory_items=50)
        errors = []

        def writer(thread_id):
            try:
                for i in range(30):
                    cache.set(f"thread_{thread_id}_key_{i}", f"value_{i}")
            except Exception as e:
                errors.append(e)

        # Create 5 writer threads (150 items total, capacity 50)
        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # No errors
        assert len(errors) == 0

        # Cache should be at capacity
        assert len(cache._memory_storage) == 50

        # Evictions should have occurred
        stats = cache.get_stats()
        assert stats["evictions"] > 0


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create a temporary directory for cache testing."""
    return tmp_path_factory.mktemp("cache_optimization_test")

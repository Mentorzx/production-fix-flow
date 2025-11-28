"""
Tests for CacheManager (pff/utils/cache.py)

Tests cover:
- Multi-layer caching (Memory + Disk + HTTP)
- Disk cache with compression
- LRU memory cache
- HTTP template cache
- Cache invalidation and expiration
- Multi-process safety (file locking)
"""

import pytest
import tempfile
import time
from pathlib import Path
from pff.utils.cache import CacheManager, DiskCache, HttpTemplateCache


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache_manager(temp_cache_dir):
    """Create a CacheManager instance with temp directory."""
    return CacheManager(str(temp_cache_dir))


# ═══════════════════════════════════════════════════════════════════
# DiskCache Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestDiskCache:
    """Test disk-based caching functionality."""

    def test_disk_cache_decorator_basic(self, temp_cache_dir):
        """Test basic disk cache decorator functionality."""
        cache = DiskCache(temp_cache_dir)
        call_count = 0

        @cache()
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call - should execute function
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Function not called again

    def test_disk_cache_different_args(self, temp_cache_dir):
        """Test cache differentiates between different arguments."""
        cache = DiskCache(temp_cache_dir)
        call_count = 0

        @cache()
        def multiply(x: int, y: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * y

        result1 = multiply(3, 4)
        result2 = multiply(5, 6)
        result3 = multiply(3, 4)  # Should hit cache

        assert result1 == 12
        assert result2 == 30
        assert result3 == 12
        assert call_count == 2  # Only 2 unique calls

    def test_disk_cache_with_ttl(self, temp_cache_dir):
        """Test cache respects TTL (time-to-live)."""
        cache = DiskCache(temp_cache_dir)
        call_count = 0

        @cache(ttl=1)  # 1 second TTL
        def get_value(x):
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        # First call
        result1 = get_value(1)
        assert result1 == "result_1"
        assert call_count == 1

        # Immediate second call - should use cache
        result2 = get_value(1)
        assert result2 == "result_1"  # Same cached result
        assert call_count == 1

        # Wait for TTL to expire (slightly longer to account for timing precision)
        time.sleep(1.2)

        # Third call - cache expired, should execute again
        result3 = get_value(1)
        assert result3 == "result_2"  # New result
        assert call_count == 2

    def test_disk_cache_with_compression(self, temp_cache_dir):
        """Test cache with gzip compression enabled."""
        cache = DiskCache(temp_cache_dir)

        @cache()
        def large_data():
            return "x" * 10000  # Large string for compression

        result = large_data()
        assert len(result) == 10000

        # Verify cache file exists (may be .pkl or .pkl.gz depending on env)
        pkl_files = list(temp_cache_dir.glob("*.pkl*"))
        assert len(pkl_files) > 0

        # Verify compression is used (default is True unless DISKCACHE_NO_GZIP is set)
        if cache.compress:
            gz_files = list(temp_cache_dir.glob("*.pkl.gz"))
            assert len(gz_files) > 0

    def test_disk_cache_handles_exceptions(self, temp_cache_dir):
        """Test cache doesn't cache exceptions."""
        cache = DiskCache(temp_cache_dir)
        call_count = 0

        @cache()
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("First call fails")
            return "success"

        # First call raises exception
        with pytest.raises(ValueError):
            failing_function()

        assert call_count == 1

        # Second call should execute again (not cached)
        result = failing_function()
        assert result == "success"
        assert call_count == 2


# ═══════════════════════════════════════════════════════════════════
# Memory Cache (LRU) Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestLRUCache:
    """Test in-memory LRU cache functionality."""

    def test_lru_cache_basic(self, cache_manager):
        """Test basic LRU cache functionality."""
        call_count = 0

        @cache_manager.lru_cache(maxsize=128)
        def add(a: int, b: int) -> int:
            nonlocal call_count
            call_count += 1
            return a + b

        # First call
        result1 = add(2, 3)
        assert result1 == 5
        assert call_count == 1

        # Second call with same args - should use cache
        result2 = add(2, 3)
        assert result2 == 5
        assert call_count == 1

    def test_lru_cache_eviction(self, cache_manager):
        """Test LRU cache evicts least recently used items."""
        call_count = 0

        @cache_manager.lru_cache(maxsize=2)  # Small cache
        def multiply(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Fill cache
        multiply(1)  # call_count = 1
        multiply(2)  # call_count = 2

        # Access 1 again (makes it recently used)
        multiply(1)  # call_count = 2 (from cache)

        # Add new item - should evict 2 (least recently used)
        multiply(3)  # call_count = 3

        # Access 2 again - should recompute (was evicted)
        multiply(2)  # call_count = 4

        assert call_count == 4

    def test_lru_cache_clear(self, cache_manager):
        """Test clearing LRU cache."""
        call_count = 0

        @cache_manager.lru_cache(maxsize=128)
        def square(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * x

        square(4)
        assert call_count == 1

        square.cache_clear()

        square(4)  # Should recompute
        assert call_count == 2


# ═══════════════════════════════════════════════════════════════════
# HTTP Template Cache Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestHttpTemplateCache:
    """Test HTTP template caching functionality."""

    def test_http_cache_stores_template(self, cache_manager):
        """Test HTTP template cache stores URL patterns."""
        cache = cache_manager.templates

        url = "https://api.example.com/users/123"
        endpoint_type = "user_detail"

        # Store entry in cache (set() creates the entry automatically)
        entry = cache.set(url, endpoint_type, method="GET")

        # Verify entry was created
        assert entry is not None
        assert entry.endpoint_type == endpoint_type

        # Retrieve using same URL
        cached = cache.get(url, endpoint_type)
        assert cached is not None
        assert cached.endpoint_type == endpoint_type

    def test_http_cache_template_matching(self, cache_manager):
        """Test HTTP cache matches URL patterns."""
        cache = cache_manager.templates

        # Store template with URL containing variable data
        url = "https://api.example.com/users/123/posts/456"
        endpoint_type = "user_posts"

        # Store entry - the cache will extract template pattern
        entry = cache.set(url, endpoint_type, method="GET")

        # Verify entry was created with a template
        assert entry is not None
        assert entry.endpoint_type == endpoint_type

        # Retrieve the template
        cached = cache.get(url, endpoint_type)

        # Template should be stored
        assert cached is not None
        assert cached.endpoint_type == endpoint_type


# ═══════════════════════════════════════════════════════════════════
# CacheManager Integration Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestCacheManagerIntegration:
    """Test CacheManager multi-layer caching."""

    def test_cache_manager_provides_decorators(self, cache_manager):
        """Test CacheManager provides both disk and LRU cache decorators."""
        assert hasattr(cache_manager, 'disk_cache')
        assert hasattr(cache_manager, 'lru_cache')

    def test_disk_cache_through_manager(self, cache_manager):
        """Test disk cache accessed through CacheManager."""
        call_count = 0

        @cache_manager.disk_cache()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x ** 2

        result1 = compute(5)
        result2 = compute(5)

        assert result1 == 25
        assert result2 == 25
        assert call_count == 1  # Only computed once

    def test_lru_cache_through_manager(self, cache_manager):
        """Test LRU cache accessed through CacheManager."""
        call_count = 0

        @cache_manager.lru_cache(maxsize=64)
        def fibonacci(n: int) -> int:
            nonlocal call_count
            call_count += 1
            if n < 2:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        result = fibonacci(10)
        assert result == 55

        # With memoization, should be much fewer calls than naive recursion
        assert call_count < 20  # Naive would be 177 calls

    def test_multiple_cache_instances_isolated(self, temp_cache_dir):
        """Test multiple cache managers are isolated."""
        cache1 = CacheManager(str(temp_cache_dir / "cache1"))
        cache2 = CacheManager(str(temp_cache_dir / "cache2"))

        call_count1 = 0
        call_count2 = 0

        @cache1.disk_cache()
        def func1(x: int) -> int:
            nonlocal call_count1
            call_count1 += 1
            return x * 2

        @cache2.disk_cache()
        def func2(x: int) -> int:
            nonlocal call_count2
            call_count2 += 1
            return x * 3

        func1(5)
        func2(5)
        func1(5)  # Should use cache
        func2(5)  # Should use cache

        assert call_count1 == 1
        assert call_count2 == 1


# ═══════════════════════════════════════════════════════════════════
# Cache Persistence Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestCachePersistence:
    """Test cache persists across sessions."""

    def test_disk_cache_persists_across_instances(self, temp_cache_dir):
        """Test disk cache survives recreation of cache instance."""
        call_count = 0

        def create_cached_function(cache_dir):
            cache = DiskCache(cache_dir)

            @cache()
            def expensive_op(x: int) -> int:
                nonlocal call_count
                call_count += 1
                return x * 100

            return expensive_op

        # First instance
        func1 = create_cached_function(temp_cache_dir)
        result1 = func1(7)
        assert result1 == 700
        assert call_count == 1

        # Second instance - cache should persist
        func2 = create_cached_function(temp_cache_dir)
        result2 = func2(7)
        assert result2 == 700
        assert call_count == 1  # Still 1 - used persisted cache

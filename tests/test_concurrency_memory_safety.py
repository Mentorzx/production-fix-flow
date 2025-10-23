"""Tests for ConcurrencyManager memory safety features."""

import asyncio
import pytest

from pff.utils.concurrency import ConcurrencyManager


class TestMemorySafety:
    """Test memory safety checks in ConcurrencyManager."""

    def test_memory_safety_check_passes_normal_usage(self):
        """Memory check should pass when RAM usage is normal."""
        cm = ConcurrencyManager(memory_threshold_pct=95.0)  # Very high threshold

        # Should not raise (normal usage should be well below 95%)
        cm._check_memory_safety()

    def test_memory_safety_check_fails_low_threshold(self):
        """Memory check should fail when threshold is very low."""
        cm = ConcurrencyManager(memory_threshold_pct=1.0)  # Impossibly low

        # Should raise MemoryError (current usage > 1%)
        with pytest.raises(MemoryError) as exc_info:
            cm._check_memory_safety()

        assert "RAM usage" in str(exc_info.value)
        assert "threshold" in str(exc_info.value)

    def test_memory_error_includes_helpful_info(self):
        """MemoryError should include RAM stats and recommendations."""
        cm = ConcurrencyManager(memory_threshold_pct=1.0)

        with pytest.raises(MemoryError) as exc_info:
            cm._check_memory_safety()

        error_msg = str(exc_info.value)
        assert "Available:" in error_msg  # Shows available RAM
        assert "GB" in error_msg  # Shows units
        assert "max_workers" in error_msg or "Recomendação" in error_msg

    @pytest.mark.asyncio
    async def test_execute_checks_memory_before_start(self):
        """execute() should check memory before starting workers."""
        cm = ConcurrencyManager(memory_threshold_pct=1.0)  # Will fail

        async def dummy_fn(x):
            return x

        args_list = [(1,), (2,), (3,)]

        # Should raise MemoryError before processing tasks
        with pytest.raises(MemoryError):
            await cm.execute(dummy_fn, args_list, task_type="io_async")


class TestLazyTaskCreation:
    """Test lazy task creation in IoAsyncioStrategy."""

    @pytest.mark.asyncio
    async def test_small_task_list_uses_original_method(self):
        """Small task lists (<100) should use simpler original method."""
        cm = ConcurrencyManager()

        async def add_one(x):
            return x + 1

        # 50 tasks (< 100 threshold)
        args_list = [(i,) for i in range(50)]

        results = await cm.execute(
            add_one, args_list, task_type="io_async", max_workers=4
        )

        assert len(results) == 50
        assert results == list(range(1, 51))

    @pytest.mark.asyncio
    async def test_large_task_list_uses_bounded_queue(self):
        """Large task lists (>=100) should use lazy task creation."""
        cm = ConcurrencyManager()

        async def add_one(x):
            await asyncio.sleep(0.001)  # Simulate async work
            return x + 1

        # 200 tasks (>= 100 threshold)
        args_list = [(i,) for i in range(200)]

        results = await cm.execute(
            add_one, args_list, task_type="io_async", max_workers=4
        )

        # Results should be correct and in order
        assert len(results) == 200
        assert results == list(range(1, 201))

    @pytest.mark.asyncio
    async def test_lazy_creation_maintains_order(self):
        """Lazy task creation should preserve result order."""
        cm = ConcurrencyManager()

        async def double(x):
            # Introduce variable delay to test ordering
            await asyncio.sleep(0.001 * (10 - x % 10))
            return x * 2

        args_list = [(i,) for i in range(150)]

        results = await cm.execute(
            double, args_list, task_type="io_async", max_workers=8
        )

        # Order should be preserved despite variable delays
        assert results == [i * 2 for i in range(150)]

    @pytest.mark.asyncio
    async def test_lazy_creation_handles_errors(self):
        """Lazy task creation should handle task errors gracefully."""
        cm = ConcurrencyManager()

        async def fail_on_five(x):
            if x == 5:
                raise ValueError(f"Failed on {x}")
            return x

        args_list = [(i,) for i in range(10)]

        # Should propagate error
        with pytest.raises(ValueError) as exc_info:
            await cm.execute(fail_on_five, args_list, task_type="io_async")

        assert "Failed on 5" in str(exc_info.value)

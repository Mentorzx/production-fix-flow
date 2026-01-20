"""Tests for pff/shared/acceleration/concurrency.py.

Tests GlobalLock, ExecutorFactory, _format_time, progress_bar utilities,
and executor implementations without heavy parallel operations.
"""

from __future__ import annotations

import threading
import time

import pytest

from pff.shared.acceleration.concurrency import (
    BaseExecutor,
    ExecutorFactory,
    GlobalLock,
    GPUInfo,
    HardwareManager,
    ThreadExecutor,
    _format_time,
    get_lock,
)

# ─────────────────────────── GlobalLock Tests ───────────────────────────


class TestGlobalLock:
    """Tests for GlobalLock wrapper."""

    def test_global_lock_creation(self) -> None:
        """GlobalLock should be creatable."""
        lock = GlobalLock()
        assert lock is not None
        assert hasattr(lock, "_lock")

    def test_global_lock_context_manager(self) -> None:
        """GlobalLock should work as context manager."""
        lock = GlobalLock()
        counter = 0

        with lock:
            counter += 1

        assert counter == 1

    def test_global_lock_acquire_release(self) -> None:
        """GlobalLock should support explicit acquire/release."""
        lock = GlobalLock()
        result = lock.acquire(blocking=True)
        assert result is True
        lock.release()

    def test_global_lock_acquire_non_blocking(self) -> None:
        """Non-blocking acquire on locked lock should return False."""
        lock = GlobalLock()
        lock.acquire(blocking=True)

        # Try to acquire again non-blocking
        result = lock.acquire(blocking=False)
        assert result is False

        lock.release()

    def test_global_lock_thread_safety(self) -> None:
        """GlobalLock should provide thread safety."""
        lock = GlobalLock()
        counter = [0]  # Use list to allow mutation in threads
        iterations = 100

        def increment():
            for _ in range(iterations):
                with lock:
                    counter[0] += 1

        threads = [threading.Thread(target=increment) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter[0] == 500  # 5 threads × 100 iterations

    def test_get_lock_returns_global_lock(self) -> None:
        """get_lock() should return a GlobalLock instance."""
        lock = get_lock()
        assert isinstance(lock, GlobalLock)

    def test_get_lock_returns_new_instance(self) -> None:
        """get_lock() should return new instance each call."""
        lock1 = get_lock()
        lock2 = get_lock()
        assert lock1 is not lock2


# ─────────────────────────── _format_time Tests ───────────────────────────


class TestFormatTime:
    """Tests for _format_time utility."""

    def test_format_time_negative(self) -> None:
        """Negative seconds should return placeholder."""
        assert _format_time(-1) == "--:--"
        assert _format_time(-100) == "--:--"

    def test_format_time_zero(self) -> None:
        """Zero seconds should return 00:00."""
        assert _format_time(0) == "00:00"

    def test_format_time_seconds_only(self) -> None:
        """Less than a minute should show MM:SS."""
        assert _format_time(30) == "00:30"
        assert _format_time(45) == "00:45"

    def test_format_time_minutes_and_seconds(self) -> None:
        """Minutes and seconds should show MM:SS."""
        assert _format_time(90) == "01:30"
        assert _format_time(125) == "02:05"
        assert _format_time(3599) == "59:59"

    def test_format_time_hours(self) -> None:
        """Hours should show HH:MM:SS."""
        assert _format_time(3600) == "01:00:00"
        assert _format_time(3661) == "01:01:01"
        assert _format_time(7323) == "02:02:03"

    def test_format_time_float(self) -> None:
        """Float seconds should be truncated."""
        assert _format_time(30.9) == "00:30"
        assert _format_time(90.5) == "01:30"


# ─────────────────────────── ExecutorFactory Tests ───────────────────────────


class TestExecutorFactory:
    """Tests for ExecutorFactory."""

    def test_create_thread_executor(self) -> None:
        """Factory should create ThreadExecutor."""
        executor = ExecutorFactory.create("thread", max_workers=2)
        try:
            assert isinstance(executor, ThreadExecutor)
        finally:
            executor.shutdown()

    def test_create_thread_executor_case_insensitive(self) -> None:
        """Factory should handle case-insensitive kind."""
        executor = ExecutorFactory.create("THREAD", max_workers=2)
        try:
            assert isinstance(executor, ThreadExecutor)
        finally:
            executor.shutdown()

    def test_create_unknown_raises(self) -> None:
        """Factory should raise for unknown executor kind."""
        with pytest.raises(ValueError, match="desconhecido"):
            ExecutorFactory.create("unknown_executor")

    def test_create_process_executor(self) -> None:
        """Factory should create ProcessExecutor (or fallback)."""
        executor = ExecutorFactory.create("process", max_workers=2)
        try:
            # Should be either ProcessExecutor or ThreadExecutor (fallback)
            assert isinstance(executor, BaseExecutor)
        finally:
            executor.shutdown()


# ─────────────────────────── ThreadExecutor Tests ───────────────────────────


class TestThreadExecutor:
    """Tests for ThreadExecutor."""

    def test_thread_executor_map_empty(self) -> None:
        """map with empty args should return empty list."""
        executor = ThreadExecutor(max_workers=2)
        try:
            result = executor.map(lambda x: x * 2, [])
            assert result == []
        finally:
            executor.shutdown()

    def test_thread_executor_map_single(self) -> None:
        """map with single item should work."""
        executor = ThreadExecutor(max_workers=2)
        try:
            result = executor.map(lambda x: x * 2, [(5,)])
            assert result == [10]
        finally:
            executor.shutdown()

    def test_thread_executor_map_multiple(self) -> None:
        """map should process multiple items in parallel."""
        executor = ThreadExecutor(max_workers=4)
        try:
            args = [(i,) for i in range(10)]
            result = executor.map(lambda x: x * 2, args)
            assert result == [i * 2 for i in range(10)]
        finally:
            executor.shutdown()

    def test_thread_executor_preserves_order(self) -> None:
        """map should preserve order of results."""
        executor = ThreadExecutor(max_workers=4)
        try:

            def slow_fn(x):
                time.sleep(0.01 * (10 - x))  # Earlier items take longer
                return x

            args = [(i,) for i in range(10)]
            result = executor.map(slow_fn, args)
            assert result == list(range(10))
        finally:
            executor.shutdown()

    def test_thread_executor_submit(self) -> None:
        """submit should return a future."""
        executor = ThreadExecutor(max_workers=2)
        try:
            future = executor.submit(lambda x: x * 2, 5)
            result = future.result()
            assert result == 10
        finally:
            executor.shutdown()


# ─────────────────────────── GPUInfo Tests ───────────────────────────


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_gpu_info_creation(self) -> None:
        """GPUInfo should be creatable with all fields."""
        gpu = GPUInfo(
            id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024 * 1024 * 1024,  # 24 GB
            compute_capability=(8, 9),
            uuid="GPU-12345678",
        )
        assert gpu.id == 0
        assert gpu.name == "NVIDIA RTX 4090"
        assert gpu.compute_capability == (8, 9)

    def test_gpu_info_fields(self) -> None:
        """GPUInfo should have expected fields."""
        gpu = GPUInfo(
            id=1,
            name="Test GPU",
            memory_total=8 * 1024 * 1024 * 1024,
            compute_capability=(7, 5),
            uuid="GPU-TEST",
        )
        assert hasattr(gpu, "id")
        assert hasattr(gpu, "name")
        assert hasattr(gpu, "memory_total")
        assert hasattr(gpu, "compute_capability")
        assert hasattr(gpu, "uuid")


# ─────────────────────────── HardwareManager Tests ───────────────────────────


class TestHardwareManager:
    """Tests for HardwareManager."""

    def test_hardware_manager_creation(self) -> None:
        """HardwareManager should detect CPU cores."""
        hw = HardwareManager()
        assert hw.physical_cores >= 1
        assert hw.logical_cores >= 1
        assert hw.logical_cores >= hw.physical_cores

    def test_hardware_manager_gpus_is_list(self) -> None:
        """HardwareManager.gpus should be a list."""
        hw = HardwareManager()
        assert isinstance(hw.gpus, list)
        # GPUs may or may not be present

    def test_hardware_manager_context_manager(self) -> None:
        """HardwareManager should work as context manager."""
        with HardwareManager() as hw:
            assert hw.physical_cores >= 1

    def test_hardware_manager_getstate_setstate(self) -> None:
        """HardwareManager should be picklable."""

        hw = HardwareManager()
        state = hw.__getstate__()
        assert "physical_cores" in state
        assert "logical_cores" in state
        assert "gpus" in state

        # Test round-trip
        hw2 = HardwareManager()
        hw2.__setstate__(state)
        assert hw2.physical_cores == hw.physical_cores
        assert hw2.logical_cores == hw.logical_cores


# ─────────────────────────── Integration Tests ───────────────────────────


class TestConcurrencyIntegration:
    """Integration tests for concurrency utilities."""

    def test_executor_with_exception(self) -> None:
        """Executor should propagate exceptions."""
        executor = ThreadExecutor(max_workers=2)

        def failing_fn(x):
            if x == 5:
                raise ValueError("Test error")
            return x

        try:
            with pytest.raises(ValueError, match="Test error"):
                executor.map(failing_fn, [(i,) for i in range(10)])
        finally:
            executor.shutdown()

    def test_executor_with_varying_durations(self) -> None:
        """Executor should handle tasks with varying durations."""
        executor = ThreadExecutor(max_workers=4)
        try:

            def variable_fn(x):
                time.sleep(0.001 * x)
                return x * 2

            args = [(i,) for i in range(20)]
            result = executor.map(variable_fn, args)
            assert result == [i * 2 for i in range(20)]
        finally:
            executor.shutdown()

    def test_multiple_executors(self) -> None:
        """Multiple executors should work independently."""
        exec1 = ThreadExecutor(max_workers=2)
        exec2 = ThreadExecutor(max_workers=2)

        try:
            result1 = exec1.map(lambda x: x + 1, [(i,) for i in range(5)])
            result2 = exec2.map(lambda x: x * 2, [(i,) for i in range(5)])

            assert result1 == [1, 2, 3, 4, 5]
            assert result2 == [0, 2, 4, 6, 8]
        finally:
            exec1.shutdown()
            exec2.shutdown()

"""
Tests for LoopAccelerator - Generic loop acceleration framework.

Tests cover:
- Strategy selection (Numba/Vectorized/Parallel/Python)
- Basic map() functionality
- Batch processing
- Error handling and fallback
- Performance statistics

Author: PFF Team
Date: 2025-10-31
"""

import pytest
import numpy as np

from pff.utils import (
    LoopAccelerator,
    AcceleratorConfig,
    AcceleratorBackend,
    accelerate_loop,
)


def square(x):
    """Simple test function."""
    return x ** 2


def add_one(x):
    """Simple test function."""
    return x + 1


class TestLoopAcceleratorBasics:
    """Basic functionality tests."""

    def test_simple_map(self):
        """Test basic map operation."""
        accelerator = LoopAccelerator()
        items = list(range(10))
        results = accelerator.map(square, items)

        assert len(results) == 10
        assert results[0] == 0
        assert results[5] == 25
        assert results[9] == 81

    def test_empty_input(self):
        """Test with empty input."""
        accelerator = LoopAccelerator()
        results = accelerator.map(square, [])

        assert results == []

    def test_convenience_function(self):
        """Test accelerate_loop() convenience function."""
        items = list(range(5))
        results = accelerate_loop(square, items)

        assert len(results) == 5
        assert results == [0, 1, 4, 9, 16]


class TestBackendSelection:
    """Test different acceleration backends."""

    def test_numba_backend(self):
        """Test Numba backend (if available)."""
        config = AcceleratorConfig(backend=AcceleratorBackend.NUMBA)
        accelerator = LoopAccelerator(config)

        items = list(range(100))
        results = accelerator.map(square, items)

        assert len(results) == 100
        assert results[50] == 2500

    def test_python_backend(self):
        """Test pure Python backend."""
        config = AcceleratorConfig(backend=AcceleratorBackend.PYTHON)
        accelerator = LoopAccelerator(config)

        items = list(range(10))
        results = accelerator.map(square, items)

        assert len(results) == 10
        assert results[5] == 25

    def test_parallel_backend(self):
        """Test parallel backend."""
        config = AcceleratorConfig(backend=AcceleratorBackend.PARALLEL)
        accelerator = LoopAccelerator(config)

        items = list(range(50))
        results = accelerator.map(add_one, items)

        assert len(results) == 50
        assert results[0] == 1
        assert results[49] == 50


class TestStatistics:
    """Test performance statistics."""

    def test_stats_tracking(self):
        """Test that stats are tracked."""
        config = AcceleratorConfig(profile=True)
        accelerator = LoopAccelerator(config)

        items = list(range(100))
        accelerator.map(square, items)

        stats = accelerator.get_stats()
        assert stats["calls"] == 1
        assert stats["items_processed"] == 100
        assert stats["total_time"] > 0

    def test_stats_reset(self):
        """Test stats reset."""
        accelerator = LoopAccelerator()
        items = list(range(10))

        accelerator.map(square, items)
        accelerator.reset_stats()

        stats = accelerator.get_stats()
        assert stats["calls"] == 0
        assert stats["items_processed"] == 0


class TestBatchProcessing:
    """Test batch processing functionality."""

    def process_batch(self, batch):
        """Process a batch of items."""
        return [x * 2 for x in batch]

    def test_map_batch(self):
        """Test batch processing."""
        accelerator = LoopAccelerator()
        items = list(range(20))

        results = accelerator.map_batch(
            self.process_batch,
            items,
            batch_size=5
        )

        assert len(results) == 20
        assert results[0] == 0
        assert results[10] == 20
        assert results[19] == 38


class TestErrorHandling:
    """Test error handling and fallback."""

    def failing_function(self, x):
        """Function that fails for certain inputs."""
        if x == 5:
            raise ValueError("Test error")
        return x * 2

    def test_partial_failure(self):
        """Test handling of partial failures."""
        # Note: This test depends on implementation
        # Currently LoopAccelerator may not handle partial failures gracefully
        # This is a placeholder for future enhancement
        pass


class TestNumpyIntegration:
    """Test with NumPy arrays."""

    def test_numpy_input(self):
        """Test with NumPy array input."""
        accelerator = LoopAccelerator()
        items = np.arange(10)

        results = accelerator.map(square, items.tolist())

        assert len(results) == 10
        assert results[5] == 25

    def test_numpy_output(self):
        """Test converting results to NumPy array."""
        accelerator = LoopAccelerator()
        items = list(range(10))

        results = accelerator.map(square, items)
        results_array = np.array(results)

        assert results_array.shape == (10,)
        assert results_array[5] == 25


@pytest.mark.slow
class TestPerformance:
    """Performance tests (marked as slow)."""

    def test_large_dataset(self):
        """Test with large dataset."""
        accelerator = LoopAccelerator()
        items = list(range(10000))

        results = accelerator.map(square, items)

        assert len(results) == 10000
        assert results[5000] == 25000000

    def test_numba_speedup(self):
        """Test that Numba provides speedup (if available)."""
        # This is a qualitative test - just ensure it completes
        config = AcceleratorConfig(backend=AcceleratorBackend.NUMBA)
        accelerator = LoopAccelerator(config)

        items = list(range(1000))
        results = accelerator.map(square, items)

        assert len(results) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

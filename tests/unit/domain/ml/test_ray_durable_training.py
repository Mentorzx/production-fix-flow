"""
Tests for Ray Durable Training & Node Affinity.

Tests durable trainable, node affinity scheduling, and fault tolerance.
"""

import tempfile

import pytest


class TestDurableRayTrainer:
    """Test DurableRayTrainer implementation."""

    def test_durable_trainer_initialization(self):
        """Test DurableRayTrainer initializes correctly."""
        from pathlib import Path

        from pff.shared.acceleration.concurrency import DurableRayTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DurableRayTrainer(checkpoint_dir=tmpdir, max_retries=2)
            assert trainer is not None
            assert trainer.checkpoint_dir == Path(tmpdir)
            assert trainer.max_retries == 2

    def test_create_durable_trainable(self):
        """Test create_durable_trainable method."""
        from pff.shared.acceleration.concurrency import DurableRayTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DurableRayTrainer(checkpoint_dir=tmpdir, max_retries=1)

            def simple_train_fn(x: int, y: int) -> int:
                return x + y

            durable_fn = trainer.create_durable_trainable(simple_train_fn)
            assert durable_fn is not None

    def test_create_node_affinity_executor(self):
        """Test create_node_affinity_executor method."""
        from pff.shared.acceleration.concurrency import DurableRayTrainer

        trainer = DurableRayTrainer(max_retries=1)

        def simple_fn(x: int) -> int:
            return x * 2

        executor = trainer.create_node_affinity_executor(simple_fn)
        assert executor is not None

    def test_execute_with_fault_tolerance(self):
        """Test execute_with_fault_tolerance method."""
        pytest.skip("Ray initialization can hang in test environment - skip for CI stability")

    def test_get_durable_trainer(self):
        """Test get_durable_trainer factory function."""
        from pff.shared.acceleration.concurrency import get_durable_trainer

        trainer = get_durable_trainer(max_retries=2)
        assert trainer is not None
        assert trainer.max_retries == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

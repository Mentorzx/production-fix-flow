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
                """Execute simple train fn.



                Args:

                    x: Input value used by this callable.

                    y: Input value used by this callable.



                Returns:

                    Return value produced by the callable.



                Notes:

                    Keep behavior deterministic and free of hidden side effects.

                """

                return x + y

            durable_fn = trainer.create_durable_trainable(simple_train_fn)
            assert durable_fn is not None

    def test_create_node_affinity_executor(self):
        """Test create_node_affinity_executor method."""
        from pff.shared.acceleration.concurrency import DurableRayTrainer

        trainer = DurableRayTrainer(max_retries=1)

        def simple_fn(x: int) -> int:
            """Execute simple fn.



            Args:

                x: Input value used by this callable.



            Returns:

                Return value produced by the callable.

            """

            return x * 2

        executor = trainer.create_node_affinity_executor(simple_fn)
        assert executor is not None

    def test_execute_with_fault_tolerance(self):
        """Test durable wrapper uses Ray retry metadata without starting a cluster."""
        from pff.shared.acceleration.concurrency import DurableRayTrainer

        remote_calls = []

        class _FakeRay:
            def remote(self, *args, **kwargs):
                remote_calls.append(kwargs)

                def _decorate(fn):
                    return fn

                return _decorate

        trainer = DurableRayTrainer(max_retries=4)
        trainer._ray = _FakeRay()

        def simple_train_fn(value: int) -> int:
            return value + 1

        durable_fn = trainer.create_durable_trainable(simple_train_fn)

        assert durable_fn(1) == 2
        assert remote_calls[0]["max_retries"] == 4

    def test_get_durable_trainer(self):
        """Test get_durable_trainer factory function."""
        from pff.shared.acceleration.concurrency import get_durable_trainer

        trainer = get_durable_trainer(max_retries=2)
        assert trainer is not None
        assert trainer.max_retries == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

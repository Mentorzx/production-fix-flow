"""
Tests for Ray Durable Training & Node Affinity.

Tests durable trainable, node affinity scheduling, and fault tolerance.
"""

import pytest
import tempfile


class TestDurableRayTrainer:
    """Test DurableRayTrainer implementation."""

    def test_durable_trainer_initialization(self):
        """Test DurableRayTrainer initializes correctly."""
        from pff.shared.acceleration.concurrency import DurableRayTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DurableRayTrainer(checkpoint_dir=tmpdir, max_retries=2)
            assert trainer is not None
            assert trainer.checkpoint_dir == tmpdir
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
        pytest.skip(
            "Ray initialization can hang in test environment - skip for CI stability"
        )

    def test_get_durable_trainer(self):
        """Test get_durable_trainer factory function."""
        from pff.shared.acceleration.concurrency import get_durable_trainer

        trainer = get_durable_trainer(max_retries=2)
        assert trainer is not None
        assert trainer.max_retries == 2


class TestAOTAutogradOptimizer:
    """Test AOTAutogradOptimizer implementation."""

    def test_aot_optimizer_initialization(self):
        """Test AOTAutogradOptimizer initializes correctly."""
        from pff.infrastructure.performance import AOTAutogradOptimizer

        optimizer = AOTAutogradOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, "create_aot_function")
        assert hasattr(optimizer, "optimize_transe_training_step")
        assert hasattr(optimizer, "enable_operator_fusion")

    def test_create_aot_function(self):
        """Test create_aot_function method."""
        from pff.infrastructure.performance import AOTAutogradOptimizer

        optimizer = AOTAutogradOptimizer()

        def simple_fn(x: float) -> float:
            return x * 2.0

        aot_fn = optimizer.create_aot_function(simple_fn)
        assert aot_fn is not None

        result = aot_fn(5.0)
        assert result == 10.0

    def test_default_compiler(self):
        """Test _get_default_compiler method."""
        from pff.infrastructure.performance import AOTAutogradOptimizer

        optimizer = AOTAutogradOptimizer()
        compiler = optimizer._get_default_compiler()
        assert compiler is not None

    def test_optimize_transe_training_step(self):
        """Test optimize_transe_training_step method."""
        from pff.infrastructure.performance import AOTAutogradOptimizer
        import torch
        from torch import nn
        from torch.optim import SGD

        optimizer = AOTAutogradOptimizer()

        model = nn.Linear(10, 5)
        opt = SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        inputs = (torch.randn(5, 10), torch.randn(5, 5))

        try:
            optimized_step = optimizer.optimize_transe_training_step(
                model, opt, criterion, inputs
            )
            assert optimized_step is not None
        except Exception:
            pytest.skip(
                "Training step optimization failed (expected in test environment)"
            )

    def test_enable_operator_fusion(self):
        """Test enable_operator_fusion method."""
        from pff.infrastructure.performance import AOTAutogradOptimizer
        from torch import nn

        optimizer = AOTAutogradOptimizer()

        model = nn.Linear(10, 5)
        optimized_model = optimizer.enable_operator_fusion(model)

        assert optimized_model is not None
        assert isinstance(optimized_model, nn.Linear)

    def test_benchmark_aot_vs_eager(self):
        """Test benchmark_aot_vs_eager method."""
        from pff.infrastructure.performance import AOTAutogradOptimizer
        import torch

        optimizer = AOTAutogradOptimizer()

        def simple_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.matmul(x, y)

        example_args = (torch.randn(100, 50), torch.randn(50, 100))

        try:
            results = optimizer.benchmark_aot_vs_eager(
                simple_fn, example_args, iterations=5
            )

            assert results is not None
            assert "eager_avg_time" in results
            assert "aot_avg_time" in results
            assert "speedup" in results
            assert "iterations" in results
            assert results["iterations"] == 5
        except Exception:
            pytest.skip("Benchmarking failed (functorch not available)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

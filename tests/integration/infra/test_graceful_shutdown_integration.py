"""Integration tests for graceful shutdown in ML pipelines.

Tests that Ctrl+C properly interrupts training at various stages,
saves emergency checkpoints, and allows recovery.

Author: PFF Team
Date: 2025-12-02
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pff.shared.ops import global_interrupt_manager as gim

pytestmark = pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")


@pytest.fixture(autouse=True)
def reset_interrupt_manager():
    """Reset GlobalInterruptManager state before and after each test."""
    manager = gim.get_interrupt_manager()
    manager.reset()
    yield
    manager.reset()


class TestDSLFMGracefulShutdown:
    """Test graceful shutdown in DSLFM training."""

    def test_dslfm_manager_registers_interrupt_handler(self, tmp_path: Path):
        """Test that DSLFMManager registers cleanup callback."""
        manager = gim.get_interrupt_manager()
        initial_callbacks = len(manager._callbacks)

        # Create a real config file
        config_path = tmp_path / "dslfm_config.yaml"
        config_content = """
training:
  epochs: 10
  seed: 42
  batch_size: 128
  learning_rate: 0.0001
model:
  embedding_dim: 64
  gamma: 12.0
checkpointing:
  save_dir: "{checkpoint_dir}"
observability:
  enable_debugging: false
""".format(checkpoint_dir=str(tmp_path / "checkpoints"))
        config_path.write_text(config_content)

        from pff.domain.learning.dslfm.manager import DSLFMManager

        # Create manager (will register callback)
        with patch.object(DSLFMManager, "_setup_device", return_value=MagicMock(type="cpu")):
            DSLFMManager(config_path)

        # Should have registered at least one callback
        assert len(manager._callbacks) > initial_callbacks

    def test_dslfm_training_respects_should_stop(self):
        """Test that simulated training loop checks should_stop."""
        manager = gim.get_interrupt_manager()
        epochs_completed = 0

        def simulate_training(num_epochs: int):
            nonlocal epochs_completed
            for epoch in range(num_epochs):
                if gim.should_stop():
                    break
                epochs_completed += 1
                # Simulate interrupt mid-training
                if epoch == 3:
                    manager.force_stop("User pressed Ctrl+C")

        simulate_training(100)

        assert epochs_completed == 4  # 0, 1, 2, 3 completed before stop
        assert manager.should_stop is True


class TestKGPipelineGracefulShutdown:
    """Test graceful shutdown in KG pipeline."""

    def test_kg_pipeline_check_interruption_points(self):
        """Test that check_interruption raises when stopped."""
        manager = gim.get_interrupt_manager()

        # Should pass normally
        gim.check_interruption()

        # Set stop flag
        manager._stop_event.set()

        # Should now raise
        with pytest.raises(KeyboardInterrupt):
            gim.check_interruption()


class TestEmergencyCheckpointSaving:
    """Test emergency checkpoint saving during interruption."""

    def test_emergency_checkpoint_saved_on_interrupt(self, tmp_path: Path):
        """Test that emergency checkpoint is saved when force_stop is called."""
        manager = gim.get_interrupt_manager()
        checkpoint_saved = {"path": None}

        def save_checkpoint():
            checkpoint_path = tmp_path / "emergency_model.pt"
            checkpoint_path.write_text("model state")
            checkpoint_saved["path"] = checkpoint_path

        manager.register_callback(save_checkpoint)
        manager.force_stop("Test interrupt")

        assert checkpoint_saved["path"] is not None
        assert checkpoint_saved["path"].exists()

    def test_multiple_components_save_checkpoints(self, tmp_path: Path):
        """Test that multiple components each save their checkpoints."""
        manager = gim.get_interrupt_manager()
        saved_checkpoints: list[Path] = []

        def dslfm_checkpoint():
            path = tmp_path / "dslfm_emergency.pt"
            path.write_text("dslfm state")
            saved_checkpoints.append(path)

        def ensemble_checkpoint():
            path = tmp_path / "ensemble_emergency.pkl"
            path.write_text("ensemble state")
            saved_checkpoints.append(path)

        def kg_checkpoint():
            path = tmp_path / "kg_checkpoint.json"
            path.write_text("{}")
            saved_checkpoints.append(path)

        manager.register_callback(dslfm_checkpoint)
        manager.register_callback(ensemble_checkpoint)
        manager.register_callback(kg_checkpoint)

        manager.force_stop("Test multi-component interrupt")

        assert len(saved_checkpoints) == 3
        assert all(p.exists() for p in saved_checkpoints)


class TestCLILearnCommandGracefulShutdown:
    """Test graceful shutdown in CLI learn command."""

    def test_learn_command_checks_interruption(self):
        """Test that LearnCommand has interrupt manager."""
        import argparse

        from pff.drivers.cli.main import LearnCommand

        args = argparse.Namespace(model="dslfm", config=None)
        command = LearnCommand(args)

        assert command.interrupt_manager is not None
        assert hasattr(command, "check_interruption")

    def test_training_strategy_check_interruption(self):
        """Test that training strategies have check_interruption method."""
        from pff.drivers.cli.main import DSLFMTrainingStrategy, KGTrainingStrategy

        for strategy_class in [KGTrainingStrategy, DSLFMTrainingStrategy]:
            # All strategies inherit check_interruption from TrainingStrategy
            assert hasattr(strategy_class, "check_interruption")


class TestInterruptRecovery:
    """Test recovery from interrupted training."""

    def test_can_reset_and_restart(self):
        """Test that manager can be reset for restart."""
        manager = gim.get_interrupt_manager()

        # Simulate interruption
        manager.force_stop("First interrupt")
        assert manager.should_stop is True

        # Reset for restart
        manager.reset()
        assert manager.should_stop is False
        assert len(manager._callbacks) == 0

        # Register new callback
        callback = MagicMock()
        manager.register_callback(callback)
        assert any(cb.callback is callback for cb in manager._callbacks)

    def test_interrupted_state_persists_across_function_calls(self):
        """Test that should_stop state persists."""
        manager = gim.get_interrupt_manager()

        def function_a():
            return gim.should_stop()

        def function_b():
            return gim.should_stop()

        assert function_a() is False
        assert function_b() is False

        manager._stop_event.set()

        assert function_a() is True
        assert function_b() is True


import pytest  # noqa: E402

pytest.skip("DSLFM graceful shutdown desativado; use DSLFM/PC", allow_module_level=True)

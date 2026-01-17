"""Tests for graceful shutdown via GlobalInterruptManager.

This module tests the graceful shutdown utility that allows Ctrl+C
to cleanly terminate any part of the PFF pipeline with checkpoint saving.

Design Pattern: Observer Pattern - tests callback invocation on interrupt.

Author: PFF Team
Date: 2025-12-02
"""

from __future__ import annotations

import asyncio
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from pff.shared.acceleration.concurrency import ConcurrencyManager
from pff.shared.core.file_manager import FileManager
from pff.shared.ops import global_interrupt_manager as gim


@pytest.fixture(autouse=True)
def reset_interrupt_manager():
    """Reset GlobalInterruptManager state before and after each test."""
    manager = gim.get_interrupt_manager()
    manager.reset()
    yield
    manager.reset()


class TestGlobalInterruptManagerBasics:
    """Test basic functionality of GlobalInterruptManager."""

    def test_singleton_pattern(self):
        """Verify GlobalInterruptManager uses singleton pattern."""
        manager1 = gim.get_interrupt_manager()
        manager2 = gim.get_interrupt_manager()
        assert manager1 is manager2

    def test_initial_state_is_not_stopped(self):
        """Verify initial state has should_stop=False."""
        manager = gim.get_interrupt_manager()
        assert manager.should_stop is False
        assert manager.signal_received is False

    def test_threading_event_replaces_bool(self):
        """Ensure stop flag is backed by threading.Event for safe waits."""
        manager = gim.get_interrupt_manager()
        assert hasattr(manager, "_stop_event")
        assert manager.should_stop is False
        manager._stop_event.set()
        assert manager.should_stop is True

    def test_wait_for_stop_semantics_preserved(self):
        """wait_for_stop should reflect the same semantics as should_stop."""
        manager = gim.get_interrupt_manager()
        assert manager.wait_for_stop(timeout=0.01) is False
        manager.force_stop("wait-check")
        assert manager.wait_for_stop(timeout=0.01) is True

    def test_should_stop_helper_function(self):
        """Test should_stop() helper function."""
        assert gim.should_stop() is False
        manager = gim.get_interrupt_manager()
        manager.force_stop("test-run")
        assert gim.should_stop() is True

    def test_reset_clears_state(self):
        """Test that reset() clears all state."""
        manager = gim.get_interrupt_manager()
        manager.force_stop("reset-state")
        manager.register_callback(lambda: None)

        manager.reset()

        assert manager.should_stop is False
        assert manager.signal_received is False
        assert manager._callback_counter == 0
        assert len(manager._callbacks) == 0


class TestCallbackRegistration:
    """Test callback registration and invocation."""

    def test_register_callback(self):
        """Test callback registration."""
        manager = gim.get_interrupt_manager()
        callback = MagicMock()

        label = manager.register_callback(callback)

        assert any(cb.callback is callback for cb in manager._callbacks)
        assert label == "callback_0"

    def test_callback_invoked_on_force_stop(self):
        """Test callbacks are invoked when force_stop is called."""
        manager = gim.get_interrupt_manager()
        callback1 = MagicMock()
        callback2 = MagicMock()

        manager.register_callback(callback1)
        manager.register_callback(callback2)
        manager.force_stop("test reason")

        callback1.assert_called_once()
        callback2.assert_called_once()
        assert manager.should_stop is True

    def test_callback_error_does_not_block_others(self):
        """Test that a failing callback doesn't prevent others from running."""
        manager = gim.get_interrupt_manager()
        error_callback = MagicMock(side_effect=RuntimeError("test error"))
        success_callback = MagicMock()

        manager.register_callback(error_callback)
        manager.register_callback(success_callback)

        # Should not raise despite error_callback failing
        manager.force_stop("test")

        error_callback.assert_called_once()
        success_callback.assert_called_once()

    def test_register_with_label(self):
        """Callbacks can be registered with custom labels."""
        manager = gim.get_interrupt_manager()
        callback = MagicMock()

        label = manager.register_callback(callback, label="custom_label")

        assert label == "custom_label"
        assert any(cb.label == "custom_label" for cb in manager._callbacks)

    def test_unregister_by_callback(self):
        """Unregister a callback by reference."""
        manager = gim.get_interrupt_manager()
        callback = MagicMock()
        manager.register_callback(callback, label="to_remove")

        removed = manager.unregister_callback(callback)

        assert removed is True
        assert len(manager._callbacks) == 0

    def test_unregister_by_label(self):
        """Unregister a callback by label."""
        manager = gim.get_interrupt_manager()
        callback = MagicMock()
        label = manager.register_callback(callback, label="label_remove")

        removed = manager.unregister_callback(label)

        assert removed is True
        assert len(manager._callbacks) == 0

    def test_unregister_idempotent(self):
        """Unregister returns False when callback is not present."""
        manager = gim.get_interrupt_manager()
        assert manager.unregister_callback("missing") is False


class TestCallbackOrdering:
    """Test callback ordering by priority and stability."""

    def test_priority_ordering_stable(self):
        """Callbacks run by ascending priority then registration order."""
        manager = gim.get_interrupt_manager()
        executed: list[str] = []

        manager.register_callback(
            lambda: executed.append("normal"),
            priority=gim.PRIORITY_NORMAL,
            label="normal",
        )
        manager.register_callback(
            lambda: executed.append("critical"),
            priority=gim.PRIORITY_CRITICAL,
            label="critical",
        )
        manager.register_callback(
            lambda: executed.append("high"),
            priority=gim.PRIORITY_HIGH,
            label="high",
        )

        manager.force_stop("order-test")

        assert executed == ["critical", "high", "normal"]

    def test_same_priority_preserves_registration_order(self):
        """Callbacks keep registration order when priority ties."""
        manager = gim.get_interrupt_manager()
        executed: list[str] = []

        manager.register_callback(
            lambda: executed.append("first"), priority=gim.PRIORITY_NORMAL
        )
        manager.register_callback(
            lambda: executed.append("second"), priority=gim.PRIORITY_NORMAL
        )
        manager.register_callback(
            lambda: executed.append("third"), priority=gim.PRIORITY_NORMAL
        )

        manager.force_stop("order-stability")

        assert executed == ["first", "second", "third"]


class TestCallbackLogging:
    """Test logging content during callback execution."""

    def test_error_logging_includes_label_and_priority(self, monkeypatch):
        """Errors should mention label and priority for debugging."""
        messages: list[str] = []

        class DummyLogger:
            def error(self, msg, *args, **kwargs):
                messages.append(str(msg))

            def warning(self, msg, *args, **kwargs):
                pass

            def info(self, msg, *args, **kwargs):
                pass

            def debug(self, msg, *args, **kwargs):
                pass

        monkeypatch.setattr(gim, "logger", DummyLogger())
        manager = gim.get_interrupt_manager()
        manager.register_callback(
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            priority=gim.PRIORITY_HIGH,
            label="failing_callback",
        )

        manager.force_stop("log-test")

        assert any("failing_callback" in msg for msg in messages)
        assert any("priority=10" in msg for msg in messages)


class TestCheckInterruption:
    """Test check_interruption() behavior."""

    def test_check_interruption_raises_when_stopped(self):
        """Test check_interruption raises KeyboardInterrupt when stopped."""
        manager = gim.get_interrupt_manager()
        manager._stop_event.set()

        with pytest.raises(KeyboardInterrupt):
            gim.check_interruption()

    def test_check_interruption_passes_when_not_stopped(self):
        """Test check_interruption passes silently when not stopped."""
        # Should not raise
        gim.check_interruption()

    def test_check_interruption_logs_warning_in_english(self, monkeypatch):
        """Verify warning message is in English per AGENTS.md policy."""
        messages: list[str] = []

        class MockLogger:
            def warning(self, msg, *args, **kwargs):
                messages.append(str(msg))

            def info(self, msg, *args, **kwargs):
                pass

            def debug(self, msg, *args, **kwargs):
                pass

        monkeypatch.setattr(gim, "logger", MockLogger())
        manager = gim.get_interrupt_manager()
        manager._stop_event.set()

        with pytest.raises(KeyboardInterrupt):
            gim.check_interruption()

        assert any("interrupted" in msg.lower() for msg in messages)
        # Ensure English (no Portuguese characters)
        for msg in messages:
            assert "interrompida" not in msg.lower()


class TestInterruptibleDecorator:
    """Test @interruptible decorator."""

    def test_interruptible_decorator_passes_when_not_stopped(self):
        """Test decorated function executes normally when not stopped."""

        @gim.interruptible
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)
        assert result == 10

    def test_interruptible_decorator_raises_when_stopped(self):
        """Test decorated function raises when should_stop is True."""
        manager = gim.get_interrupt_manager()
        manager._stop_event.set()

        @gim.interruptible
        def my_function():
            return "should not reach"

        with pytest.raises(KeyboardInterrupt):
            my_function()

    def test_interruptible_preserves_function_behavior(self):
        """Test decorator preserves function's normal behavior."""

        @gim.interruptible
        def add_numbers(a: int, b: int) -> int:
            return a + b

        assert add_numbers(3, 4) == 7
        assert add_numbers(a=10, b=20) == 30

    def test_wraps_preserves_metadata(self):
        """interruptible should preserve metadata like __name__ and __doc__."""

        @gim.interruptible
        def sample_function():
            """Sample docstring."""
            return True

        assert sample_function.__name__ == "sample_function"
        assert sample_function.__doc__ == "Sample docstring."


class TestSignalHandling:
    """Test signal handler behavior."""

    def test_signal_handler_fallback_no_loop(self, monkeypatch):
        """Fallback to signal.signal when no asyncio loop is running."""
        manager = gim.get_interrupt_manager()
        manager.reset()
        registered: dict[int, Any] = {}
        originals: dict[int, str] = {}

        monkeypatch.setattr(
            asyncio,
            "get_running_loop",
            MagicMock(side_effect=RuntimeError("no loop")),
        )

        def fake_signal(sig, handler):
            registered[sig] = handler
            originals[int(sig)] = f"orig-{sig}"
            return originals[int(sig)]

        monkeypatch.setattr(signal, "signal", fake_signal)

        manager._setup_signal_handlers()

        assert set(registered.keys()) == {signal.SIGINT, signal.SIGTERM}
        assert manager._original_handlers == originals
        registered[signal.SIGINT](signal.SIGINT, None)
        assert manager.should_stop is True

    def test_signal_handler_fallback_windows(self, monkeypatch):
        """Always use signal.signal on Windows."""
        manager = gim.get_interrupt_manager()
        manager.reset()
        registered: dict[int, Any] = {}
        originals: dict[int, str] = {}

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            asyncio,
            "get_running_loop",
            MagicMock(side_effect=AssertionError("should not call get_running_loop")),
        )

        def fake_signal(sig, handler):
            registered[sig] = handler
            originals[int(sig)] = f"orig-{sig}"
            return originals[int(sig)]

        monkeypatch.setattr(signal, "signal", fake_signal)

        manager._setup_signal_handlers()

        assert set(registered.keys()) == {signal.SIGINT, signal.SIGTERM}
        assert manager._original_handlers == originals

    def test_signal_handler_asyncio_when_available(self, monkeypatch):
        """Use asyncio event loop when available."""
        manager = gim.get_interrupt_manager()
        manager.reset()
        calls: list[tuple[int, Any]] = []
        loop = MagicMock()

        def add_handler(sig, cb):
            calls.append((sig, cb))

        loop.add_signal_handler.side_effect = add_handler
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(asyncio, "get_running_loop", MagicMock(return_value=loop))

        manager._setup_signal_handlers()

        assert {sig for sig, _ in calls} == {signal.SIGINT, signal.SIGTERM}
        assert manager._original_handlers == {}
        for _, cb in calls:
            manager.reset()
            cb()
            assert manager.should_stop is True

    def test_multiple_signals_are_idempotent(self):
        """Test that multiple signals don't cause issues."""
        manager = gim.get_interrupt_manager()
        callback_count = {"count": 0}

        def counting_callback():
            callback_count["count"] += 1

        manager.register_callback(counting_callback)
        manager.force_stop("first")
        manager.force_stop("second")

        # Callback should be called twice (once per force_stop)
        assert callback_count["count"] == 2
        assert manager.should_stop is True


class TestEmergencyCheckpoint:
    """Test emergency checkpoint saving during interruption."""

    def test_emergency_checkpoint_callback(self, tmp_path: Path):
        """Test that emergency checkpoint callback is invoked."""
        manager = gim.get_interrupt_manager()
        checkpoint_saved = {"saved": False, "path": None}

        def save_emergency_checkpoint():
            checkpoint_path = tmp_path / "emergency_checkpoint.pt"
            checkpoint_path.write_text("mock checkpoint data")
            checkpoint_saved["saved"] = True
            checkpoint_saved["path"] = checkpoint_path

        manager.register_callback(save_emergency_checkpoint)
        manager.force_stop("test interruption")

        assert checkpoint_saved["saved"] is True
        assert checkpoint_saved["path"].exists()

    def test_checkpoint_callback_with_model_state(self, tmp_path: Path):
        """Test saving model state in emergency checkpoint."""
        manager = gim.get_interrupt_manager()
        model_state = {"epoch": 42, "best_mrr": 0.456, "weights": [1, 2, 3]}
        saved_state: dict[str, Any] = {}

        def save_model_checkpoint():
            saved_state.update(model_state)
            saved_state["emergency"] = True

        manager.register_callback(save_model_checkpoint)
        manager.force_stop("Ctrl+C received")

        assert saved_state["epoch"] == 42
        assert saved_state["emergency"] is True


class TestThreadSafety:
    """Test thread-safety of GlobalInterruptManager."""

    def test_concurrent_callback_registration(self):
        """Test that concurrent callback registration is thread-safe."""
        manager = gim.get_interrupt_manager()
        callbacks_registered = []

        def register_callback(idx: int):
            callback = MagicMock(name=f"callback_{idx}")
            manager.register_callback(callback)
            callbacks_registered.append(callback)

        threads = [
            threading.Thread(target=register_callback, args=(i,)) for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(manager._callbacks) == 10

    def test_concurrent_should_stop_access(self):
        """Test concurrent read/write of should_stop."""
        manager = gim.get_interrupt_manager()
        results = []

        def reader():
            for _ in range(100):
                results.append(manager.should_stop)
                time.sleep(0.001)

        def writer():
            time.sleep(0.05)
            manager._stop_event.set()

        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)

        reader_thread.start()
        writer_thread.start()
        reader_thread.join()
        writer_thread.join()

        # Eventually should see True after writer sets it
        assert True in results


class TestPipelineIntegration:
    """Test integration with pipeline components."""

    def test_pipeline_respects_should_stop(self):
        """Test that a simulated pipeline loop respects should_stop."""
        manager = gim.get_interrupt_manager()
        iterations_completed = 0

        def simulate_training_loop():
            nonlocal iterations_completed
            for epoch in range(100):
                if gim.should_stop():
                    break
                iterations_completed += 1
                if epoch == 5:
                    manager.force_stop("Early stop for test")

        simulate_training_loop()

        # Should have stopped after epoch 5
        assert iterations_completed == 6
        assert manager.should_stop is True

    def test_nested_check_interruption_calls(self):
        """Test nested functions using check_interruption."""

        def outer_function():
            gim.check_interruption()
            return inner_function()

        def inner_function():
            gim.check_interruption()
            return "success"

        # Should pass normally
        result = outer_function()
        assert result == "success"

        # Now stop and verify both levels raise
        manager = gim.get_interrupt_manager()
        manager._stop_event.set()

        with pytest.raises(KeyboardInterrupt):
            outer_function()


class TestCleanup:
    """Test cleanup and resource management."""

    def test_restore_original_handlers(self):
        """Test that original signal handlers can be restored."""
        manager = gim.get_interrupt_manager()

        # Store current handlers
        original_sigint = signal.getsignal(signal.SIGINT)

        # Restore
        manager.restore_original_handlers()

        # Handler should be restored (may be different from our test handler)
        restored_sigint = signal.getsignal(signal.SIGINT)
        assert restored_sigint == original_sigint

        # Reset manager to re-install handlers for other tests
        manager._initialized = False
        manager.__init__()

    def test_reset_allows_reuse(self):
        """Test that reset() allows reusing the manager."""
        manager = gim.get_interrupt_manager()

        # First use
        callback1 = MagicMock()
        manager.register_callback(callback1)
        manager.force_stop("first")
        assert manager.should_stop is True

        # Reset and reuse
        manager.reset()
        assert manager.should_stop is False
        assert len(manager._callbacks) == 0

        # Second use
        callback2 = MagicMock()
        manager.register_callback(callback2)
        assert any(cb.callback is callback2 for cb in manager._callbacks)


class TestUtilsIntegrations:
    """Smoke tests for integrations with utils layer components."""

    def test_concurrency_manager_registers_callback(self):
        """ConcurrencyManager should register shutdown callback on init."""
        manager = gim.get_interrupt_manager()
        assert len(manager._callbacks) == 0

        ConcurrencyManager()

        assert any(
            cb.label == "concurrency_manager_shutdown" for cb in manager._callbacks
        )

    def test_concurrency_manager_checks_should_stop(self):
        """ConcurrencyManager must raise when interrupted before execution."""
        manager = gim.get_interrupt_manager()
        manager.force_stop("pre-execution-stop")
        concurrency_manager = ConcurrencyManager()

        with pytest.raises(KeyboardInterrupt):
            concurrency_manager.execute_sync(lambda x: x, args_list=[(1,)])

    def test_file_manager_registers_callback(self):
        """FileManager registers emergency flush callback on init."""
        manager = gim.get_interrupt_manager()
        assert len(manager._callbacks) == 0

        FileManager()

        assert any(cb.label == "file_manager_flush" for cb in manager._callbacks)

    def test_file_manager_skips_save_on_interrupt(self, tmp_path: Path):
        """Save operations should short-circuit when interrupted."""
        manager = gim.get_interrupt_manager()
        manager.force_stop("skip-save")
        target = tmp_path / "data.json"

        result = FileManager.save({"a": 1}, target)

        assert result is None
        assert target.exists() is False

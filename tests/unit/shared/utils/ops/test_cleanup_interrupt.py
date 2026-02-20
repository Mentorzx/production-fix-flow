"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/ops/test_cleanup_interrupt.py

"""

import pytest

from pff.infrastructure.cleanup import CleanupCommand, CleanupEngine
from pff.shared.ops.global_interrupt_manager import PRIORITY_HIGH, get_interrupt_manager


class DummyCommand:
    """Represent DummyCommand."""

    def __init__(self, label: str, sink: list[str], side_effect=None):
        """Execute init.



        Args:

            label: Input value used by this callable.

            sink: Input value used by this callable.

            side_effect: Optional input value.

        """

        self.label = label
        self._sink = sink
        self._side_effect = side_effect

    def execute(self) -> None:
        """Execute execute.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._sink.append(self.label)
        if self._side_effect:
            self._side_effect()


class DummyStrategy:
    """Represent DummyStrategy.



    Notes:

        Encapsulates behavior while preserving architecture boundaries.

    """

    def __init__(self, commands: list[CleanupCommand]):
        """Execute init.



        Args:

            commands: Input value used by this callable.

        """

        self._commands = commands

    def build_commands(self, **kwargs) -> list[CleanupCommand]:
        """Execute build commands.



        Args:

            **kwargs: Additional keyword arguments.



        Returns:

            Return value produced by the callable.

        """

        return self._commands


@pytest.fixture(autouse=True)
def cleanup_engine_callback_cleanup():
    """Execute cleanup engine callback cleanup."""

    manager = get_interrupt_manager()
    yield
    manager.unregister_callback("cleanup_engine_emergency")


def test_cleanup_engine_registers_callback():
    """Execute test cleanup engine registers callback."""

    manager = get_interrupt_manager()
    manager.unregister_callback("cleanup_engine_emergency")

    CleanupEngine(DummyStrategy([]), auto_yes=True, dry_run=True)

    registered = [cb for cb in manager._callbacks if cb.label == "cleanup_engine_emergency"]
    assert registered and registered[0].priority == PRIORITY_HIGH


@pytest.mark.asyncio
async def test_cleanup_aborts_on_interrupt(monkeypatch):
    """Execute test cleanup aborts on interrupt.



    Args:

        monkeypatch: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    executed: list[str] = []
    command = DummyCommand("file", executed)
    engine = CleanupEngine(DummyStrategy([command]), auto_yes=True, dry_run=False)

    async def fake_filter():
        """Execute fake filter.



        Returns:

            Return value produced by the callable.

        """

        return [(command, 1)]

    monkeypatch.setattr(engine, "_filter_commands", fake_filter)
    engine._should_stop = lambda: True  # type: ignore[assignment]

    await engine.run(confirm=False)

    assert executed == []


@pytest.mark.asyncio
async def test_parallel_execution_respects_interrupt(monkeypatch):
    """Execute test parallel execution respects interrupt.



    Args:

        monkeypatch: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    executed: list[str] = []
    stop_flag = {"stop": False}

    def should_stop():
        """Execute should stop.



        Returns:

            Return value produced by the callable.

        """

        return stop_flag["stop"]

    def mark_stop():
        """Execute mark stop."""

        stop_flag["stop"] = True
        import time

        time.sleep(0.1)

    cmd_first = DummyCommand("first", executed, side_effect=mark_stop)
    # Add many more commands to ensure they are not all submitted instantly
    # although with only 10 they might be, but the engine checks _should_stop
    # before launching if we were doing it sequentially.
    # In parallel, the ConcurrencyManager will submit them to ThreadExecutor.
    others = [DummyCommand(f"other_{i}", executed) for i in range(20)]

    engine = CleanupEngine(DummyStrategy([cmd_first] + others), auto_yes=True, dry_run=False)
    engine._should_stop = should_stop  # type: ignore[assignment]

    async def fake_filter():
        """Execute fake filter.



        Returns:

            Return value produced by the callable.

        """

        return [(cmd_first, 1)] + [(c, 1) for c in others]

    monkeypatch.setattr(engine, "_filter_commands", fake_filter)

    await engine.run(confirm=False)

    assert "first" in executed
    # We expect some of the others to NOT be executed because first marks stop=True
    # and the engine or executor should catch it.
    assert len(executed) < 21

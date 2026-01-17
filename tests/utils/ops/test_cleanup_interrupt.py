import pytest

from pff.infrastructure.cleanup import CleanupCommand, CleanupEngine
from pff.shared.ops.global_interrupt_manager import PRIORITY_HIGH, get_interrupt_manager


class DummyCommand:
    def __init__(self, label: str, sink: list[str], side_effect=None):
        self.label = label
        self._sink = sink
        self._side_effect = side_effect

    def execute(self) -> None:
        self._sink.append(self.label)
        if self._side_effect:
            self._side_effect()


class DummyStrategy:
    def __init__(self, commands: list[CleanupCommand]):
        self._commands = commands

    def build_commands(self) -> list[CleanupCommand]:
        return self._commands


@pytest.fixture(autouse=True)
def cleanup_engine_callback_cleanup():
    manager = get_interrupt_manager()
    yield
    manager.unregister_callback("cleanup_engine_emergency")


def test_cleanup_engine_registers_callback():
    manager = get_interrupt_manager()
    manager.unregister_callback("cleanup_engine_emergency")

    CleanupEngine(DummyStrategy([]), auto_yes=True, dry_run=True)

    registered = [
        cb for cb in manager._callbacks if cb.label == "cleanup_engine_emergency"
    ]
    assert registered and registered[0].priority == PRIORITY_HIGH


@pytest.mark.asyncio
async def test_cleanup_aborts_on_interrupt(monkeypatch):
    executed: list[str] = []
    command = DummyCommand("file", executed)
    engine = CleanupEngine(DummyStrategy([command]), auto_yes=True, dry_run=False)

    async def fake_filter():
        return [(command, 1)]

    monkeypatch.setattr(engine, "_filter_commands", fake_filter)
    engine._should_stop = lambda: True  # type: ignore[assignment]

    await engine.run(confirm=False)

    assert executed == []


@pytest.mark.asyncio
async def test_parallel_execution_respects_interrupt(monkeypatch):
    executed: list[str] = []
    stop_flag = {"stop": False}

    def should_stop():
        return stop_flag["stop"]

    def mark_stop():
        stop_flag["stop"] = True

    cmd_one = DummyCommand("first", executed, side_effect=mark_stop)
    cmd_two = DummyCommand("second", executed)
    engine = CleanupEngine(
        DummyStrategy([cmd_one, cmd_two]), auto_yes=True, dry_run=False
    )
    engine._should_stop = should_stop  # type: ignore[assignment]

    async def fake_filter():
        return [(cmd_one, 1), (cmd_two, 1)]

    monkeypatch.setattr(engine, "_filter_commands", fake_filter)

    await engine.run(confirm=False)

    assert "first" in executed
    assert "second" not in executed

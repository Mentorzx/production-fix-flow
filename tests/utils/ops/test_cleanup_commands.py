from pff.infrastructure.cleanup.commands.base import (
    CompositeCommand,
    TransparentCompositeCommand,
)


class DummyCommand:
    def __init__(self, label: str):
        self.label = label
        self.executed = False

    def execute(self) -> None:
        self.executed = True


def test_composite_executes_children():
    a = DummyCommand("a")
    b = DummyCommand("b")
    comp = CompositeCommand("root", [a, b])

    comp.execute()

    assert a.executed and b.executed


def test_transparent_composite_flattens_children():
    a = DummyCommand("a")
    b = DummyCommand("b")
    comp = TransparentCompositeCommand("root", [a, b])

    leaves = comp.get_all_leaf_commands()

    assert leaves == [a, b]

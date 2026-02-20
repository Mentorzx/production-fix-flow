"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/ops/test_cleanup_commands.py

"""

from pff.infrastructure.cleanup.commands.base import (
    CompositeCommand,
    TransparentCompositeCommand,
)


class DummyCommand:
    """Represent DummyCommand."""

    def __init__(self, label: str):
        """Execute init.



        Args:

            label: Input value used by this callable.

        """

        self.label = label
        self.executed = False

    def execute(self) -> None:
        """Execute execute."""

        self.executed = True


def test_composite_executes_children():
    """Execute test composite executes children."""

    a = DummyCommand("a")
    b = DummyCommand("b")
    comp = CompositeCommand("root", [a, b])

    comp.execute()

    assert a.executed and b.executed


def test_transparent_composite_flattens_children():
    """Execute test transparent composite flattens children."""

    a = DummyCommand("a")
    b = DummyCommand("b")
    comp = TransparentCompositeCommand("root", [a, b])

    leaves = comp.get_all_leaf_commands()

    assert leaves == [a, b]

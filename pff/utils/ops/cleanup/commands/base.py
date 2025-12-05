"""Base classes and protocols for cleanup commands.

Design Patterns:
    - Protocol: Defines the CleanupCommand interface.
    - Composite: CompositeCommand and TransparentCompositeCommand aggregate commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol


class CleanupCommand(Protocol):
    """Protocol defining the interface for cleanup commands.

    All cleanup commands must implement this protocol to be executable
    by the CleanupEngine.

    Attributes:
        label: Human-readable description displayed in UI during execution.
    """

    label: str

    def execute(self) -> None:
        """Execute the cleanup operation.

        Implementations should handle their own error logging and recovery.
        """
        ...


@dataclass
class CompositeCommand:
    """Composite command that executes child cleanup commands sequentially.

    Design Pattern: Composite. Allows treating individual commands and
    compositions uniformly.

    Args:
        label: Human-readable description for the command group.
        children: List of child commands to execute.
        transparent: If True, `get_leaf_commands()` returns children instead of self.

    Attributes:
        label: Display label for UI.
        children: Child commands.
        transparent: Whether to expose children for flattening.
    """

    label: str
    children: list[CleanupCommand] = field(default_factory=list)
    transparent: bool = False

    def execute(self) -> None:
        """Execute all child commands sequentially."""
        for cmd in self.children:
            cmd.execute()

    def get_leaf_commands(self) -> list[CleanupCommand]:
        """Flatten and return executable commands.

        Returns:
            list[CleanupCommand]: If transparent, returns flattened children;
                otherwise returns self as a single-item list.
        """
        if not self.transparent:
            return [self]
        leaves: list[CleanupCommand] = []
        for child in self.children:
            if isinstance(child, CompositeCommand):
                leaves.extend(child.get_leaf_commands())
            else:
                leaves.append(child)
        return leaves


class TransparentCompositeCommand:
    """Composite that always exposes leaf commands for flattening.

    Unlike CompositeCommand with `transparent=True`, this class is always
    transparent and provides `get_all_leaf_commands()` for recursive flattening.

    Args:
        label: Human-readable description for the command group.
        children: Iterable of child commands to execute.

    Attributes:
        label: Display label for UI.
    """

    def __init__(self, label: str, children: Iterable[CleanupCommand]):
        """Initialize with label and child commands.

        Args:
            label: Human-readable description for the command group.
            children: Iterable of child commands to execute.
        """
        self.label = label
        self._children = list(children)

    def execute(self) -> None:
        """Execute all child commands sequentially."""
        for cmd in self._children:
            cmd.execute()

    def get_all_leaf_commands(self) -> list[CleanupCommand]:
        """Return all non-composite commands recursively.

        Traverses the command tree and collects all leaf commands that
        can be executed individually.

        Returns:
            list[CleanupCommand]: Flattened list of executable commands.
        """
        leaf_commands = []
        for child in self._children:
            if isinstance(child, TransparentCompositeCommand):
                leaf_commands.extend(child.get_all_leaf_commands())
            elif isinstance(child, CompositeCommand):
                for subchild in child.children:
                    if isinstance(subchild, TransparentCompositeCommand):
                        leaf_commands.extend(subchild.get_all_leaf_commands())
                    else:
                        leaf_commands.append(subchild)
            else:
                leaf_commands.append(child)
        return leaf_commands


__all__ = ["CleanupCommand", "CompositeCommand", "TransparentCompositeCommand"]

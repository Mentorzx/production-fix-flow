"""Base cleanup strategy module defining the Strategy protocol.

This module defines the CleanupStrategy Protocol that all cleanup
strategies must implement, following the Strategy design pattern.

Design Patterns:
    - Strategy: Defines interface for interchangeable cleanup algorithms.
    - Protocol: Uses structural subtyping for type safety.
"""
from __future__ import annotations

from typing import Protocol

from ..commands.base import CleanupCommand


class CleanupStrategy(Protocol):
    """Protocol for strategies that build cleanup commands.

    Defines the interface for cleanup strategy implementations.
    Each strategy builds a specific set of cleanup commands based
    on the cleanup scope (standard, deep, ML-focused, shutdown).
    """

    def build_commands(self) -> list[CleanupCommand]:
        """Build and return the list of cleanup commands.

        Returns:
            List of CleanupCommand instances to be executed.
        """
        ...


__all__ = ["CleanupStrategy"]

"""Factory protocol utilities shared across the codebase."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

T = TypeVar("T")


class GenericFactory(Protocol[T]):
    """Protocol for registry-backed factories.

    Implementations should expose both registration and creation methods,
    enabling consistent factory usage across modules.
    """

    def create(self, key: str, *args, **kwargs) -> T:
        """Create an instance by key."""

    def register(self, key: str, factory: Callable[..., T] | None = None):
        """Register a factory or act as a decorator."""

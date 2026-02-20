"""Factory protocol utilities shared across the codebase."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

T = TypeVar("T", covariant=True)


class GenericFactory(Protocol[T]):
    """Protocol for registry-backed factories.

    Implementations should expose both registration and creation methods,
    enabling consistent factory usage across modules.
    """

    def create(self, key: str, *args: Any, **kwargs: Any) -> T:
        """Create an instance by key."""
        ...

    def register(self, key: str, factory: Any = None) -> Any:
        """Register a factory or act as a decorator."""
        ...

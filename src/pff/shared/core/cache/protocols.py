"""Cache module protocols and interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class Serializer(Protocol):
    """Protocol for object serialization."""

    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to bytes."""
        ...

    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to an object."""
        ...


class StorageBackend(Protocol):
    """Protocol for storage operations."""

    def read(self, path: Path) -> bytes | None:
        """Read data from the given path."""
        ...

    def write(self, path: Path, data: bytes) -> None:
        """Write data to the given path."""
        ...

    def delete(self, path: Path) -> None:
        """Delete the file at the given path."""
        ...

    def exists(self, path: Path) -> bool:
        """Check if a file exists at the given path."""
        ...


class CacheKeyGenerator(Protocol):
    """Protocol for generating cache keys."""

    def generate_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate a unique cache key."""
        ...

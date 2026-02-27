"""File manager ports for PFF.

This module defines the interface for filesystem operations.
Patterns: Port/Adapter.
"""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FileManagerPort(Protocol):
    """Protocol for file management operations."""

    def exists(self, path: Path | str) -> bool:
        """Check if path exists."""
        ...

    def mkdir(
        self, path: Path | str, parents: bool = True, exist_ok: bool = True
    ) -> None:
        """Create directory."""
        ...

    def save(self, data: Any, path: Path | str, **kwargs: Any) -> None:
        """Save data to path."""
        ...

    def read(self, path: Path | str, **kwargs: Any) -> Any:
        """Read data from path."""
        ...

    def delete_file(self, path: Path | str, ignore_errors: bool = True) -> None:
        """Delete a file."""
        ...

    def glob(self, directory: Path | str, pattern: str) -> list[Path]:
        """Find files matching pattern."""
        ...

    def read_bytes(self, path: Path | str) -> bytes:
        """Read raw bytes from path."""
        ...

    def write_bytes(self, data: bytes, path: Path | str) -> None:
        """Write raw bytes to path."""
        ...

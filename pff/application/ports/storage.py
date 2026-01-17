"""Storage port interfaces for application layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class StoragePort(ABC):
    """Port for writing artifacts to storage backends."""

    @abstractmethod
    def ensure_dir(self, path: Path) -> None:
        """Ensure a directory exists.

        Args:
            path: Directory path.
        """
        raise NotImplementedError

    @abstractmethod
    def save_json(self, payload: dict[str, Any], path: Path) -> Path:
        """Persist a JSON payload.

        Args:
            payload: JSON-serializable dictionary.
            path: Destination path.

        Returns:
            Path to the written file.
        """
        raise NotImplementedError

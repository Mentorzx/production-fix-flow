"""Config loader port for application-layer dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class ConfigLoaderPort(Protocol):
    """Port for loading config dictionaries by path."""

    def __call__(self, path: Path) -> dict[str, Any]:
        """Load a configuration mapping from a file path."""

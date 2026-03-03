"""Line API endpoint provider port for application layer."""

from __future__ import annotations

from typing import Any, Protocol


class LineApiPort(Protocol):
    """Protocol for endpoint provider used by LineService."""

    def __getattr__(self, name: str) -> Any:
        """Return endpoint builders/constants by attribute name."""

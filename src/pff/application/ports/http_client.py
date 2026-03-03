"""HTTP client port interfaces for application layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HttpClientPort(Protocol):
    """Protocol for async HTTP operations used by application services."""

    async def close(self) -> None:
        """Close underlying HTTP resources."""
        ...

    async def make_request(
        self, endpoint_config: dict[str, Any], subscriber_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Execute one HTTP request and return parsed payload."""
        ...

    def _generate_unique_path(self, folder: Path, stem: str, suffix: str) -> Path:
        """Return a non-conflicting file path for persisted artifacts."""
        ...

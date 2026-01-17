"""FileHandler abstract base class (Strategy pattern).

All file format handlers implement this interface to provide
consistent read/write/async operations across formats.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class FileHandler(ABC):
    """Base class for file format handlers (Strategy pattern).

    Handlers MUST be stateless. Do not store mutable state in handler instances.
    If state is needed, use thread-local storage or pass it via kwargs.

    Each handler provides:
    - read(): Synchronous file reading
    - save(): Synchronous file writing
    - async_read(): Asynchronous file reading
    - async_save(): Asynchronous file writing
    - load_bytes(): Load from raw bytes (convenience method)
    """

    @abstractmethod
    def read(self, path: Path | io.BytesIO, **kwargs: Any) -> Any:
        """Read and parse file content.

        Args:
            path: File path or in-memory buffer.
            **kwargs: Handler-specific options.

        Returns:
            Parsed content (type depends on handler).
        """
        ...

    @abstractmethod
    def save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Save object to file.

        Args:
            obj: Object to save.
            path: Destination file path.
            **kwargs: Handler-specific options.
        """
        ...

    @abstractmethod
    async def async_read(self, path: Path, **kwargs: Any) -> Any:
        """Asynchronously read and parse file content.

        Args:
            path: File path.
            **kwargs: Handler-specific options.

        Returns:
            Parsed content.
        """
        ...

    @abstractmethod
    async def async_save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Asynchronously save object to file.

        Args:
            obj: Object to save.
            path: Destination file path.
            **kwargs: Handler-specific options.
        """
        ...

    def load_bytes(self, raw: bytes, **kwargs: Any) -> Any:
        """Load content from raw bytes.

        Default implementation wraps bytes in BytesIO and calls read().
        Subclasses may override for optimized byte loading.

        Args:
            raw: Raw file bytes.
            **kwargs: Handler-specific options.

        Returns:
            Parsed content.
        """
        return self.read(io.BytesIO(raw), **kwargs)

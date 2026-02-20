"""Cache storage backends."""

from __future__ import annotations

from pathlib import Path

from ..logging import logger
from .constants import GZIP_COMPRESSION_LEVEL, GZIP_MAGIC_BYTES
from .utils import AtomicFileWriter


class FileSystemStorage:
    """File system based storage backend with optional compression."""

    def __init__(self, compress: bool = True):
        """
        Initialize the file system storage.

        Args:
            compress: Whether to use gzip compression
        """
        self.compress = compress
        self._writer = AtomicFileWriter()

    def read(self, path: Path) -> bytes | None:
        """Read data from a file, handling both compressed and uncompressed formats."""
        import gzip

        if not path.exists():
            return None

        try:
            content = path.read_bytes()
            if content.startswith(GZIP_MAGIC_BYTES):
                return gzip.decompress(content)
            return content

        except Exception as error:
            logger.warning(f"Failed to read cache file [{path.name}]: {error}", exc_info=True)
            return None

    def write(self, path: Path, data: bytes) -> None:
        """Write data to a file with optional compression."""
        import gzip

        if self.compress:
            data = gzip.compress(data, compresslevel=GZIP_COMPRESSION_LEVEL)

        self._writer.write_atomically(path, data)

    def delete(self, path: Path) -> None:
        """Delete a file, ignoring if it doesn't exist."""
        try:
            path.unlink(missing_ok=True)
        except Exception as error:
            logger.warning(f"Failed to delete file [{path.name}]: {error}", exc_info=True)

    def exists(self, path: Path) -> bool:
        """Check if a file exists."""
        return path.exists()

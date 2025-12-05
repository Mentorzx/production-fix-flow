from __future__ import annotations

import os
import shutil
from pathlib import Path

from .logger import logger
from ..ops.global_interrupt_manager import should_stop


class FileOps:
    """Filesystem helpers routed through the utils layer.

    Design Pattern: Facade. Centralizes destructive I/O with interrupt awareness.
    """

    @staticmethod
    def rmtree_sync(path: Path, ignore_errors: bool = True) -> bool:
        """Remove a directory tree synchronously with interrupt checks.

        Args:
            path: Directory to remove.
            ignore_errors: Whether to suppress errors during removal.

        Returns:
            bool: True when removed or absent; False when skipped due to interrupt.
        """
        if should_stop():
            logger.warning(f"rmtree skipped due to interrupt: {path}")
            return False
        try:
            shutil.rmtree(path, ignore_errors=ignore_errors)
            return True
        except Exception as exc:  # noqa: BLE001
            if not ignore_errors:
                raise
            logger.debug(f"rmtree error (ignored): {path} - {exc}")
            return False

    @staticmethod
    async def rmtree_async(path: Path, ignore_errors: bool = True) -> bool:
        """Remove a directory tree in an async context with interrupt checks.

        Args:
            path: Directory to remove.
            ignore_errors: Whether to suppress errors during removal.

        Returns:
            bool: True when removed or absent; False when skipped due to interrupt.
        """
        return FileOps.rmtree_sync(path, ignore_errors=ignore_errors)

    @staticmethod
    def calculate_size(path: Path) -> int:
        """Calculate directory size using os.scandir for efficiency.

        Args:
            path: Directory to measure.

        Returns:
            int: Total size in bytes.
        """
        total = 0
        try:
            with os.scandir(path) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                        elif entry.is_dir(follow_symlinks=False):
                            total += FileOps.calculate_size(Path(entry.path))
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError):
            return total
        return total

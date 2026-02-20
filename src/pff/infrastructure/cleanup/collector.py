"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/infrastructure/cleanup/collector.py

"""

from __future__ import annotations

import os
from pathlib import Path

from pff.shared.core.config import settings


class CleanupScanCollector:
    """Consolidates all filesystem scans into a single traversal for efficiency.

    Acts as a shared cache for cleanup commands that need to scan the project tree.
    """

    def __init__(self, root_dir: Path | None = None):
        """Execute init.



        Args:

            root_dir: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.root_dir = root_dir or settings.ROOT_DIR
        self.ignored_dirs = {
            ".git",
            ".venv",
            "node_modules",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
        }
        self._cache: dict[str, list[Path]] = {}
        self._size_cache: dict[str, int] = {}
        self._scanned = False

    def scan(self, target_dirnames: set[str]) -> None:
        """Perform a single walk to find all target directories."""
        if self._scanned:
            return

        for root, dirs, _ in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs or d in target_dirnames]

            for target in target_dirnames:
                if target in dirs:
                    path = Path(root) / target
                    if target not in self._cache:
                        self._cache[target] = []
                    self._cache[target].append(path)

        self._scanned = True

    def get_paths(self, dirname: str) -> list[Path]:
        """Execute get paths.



        Args:

            dirname: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return self._cache.get(dirname, [])

    def get_size(self, dirname: str) -> int:
        """Execute get size.



        Args:

            dirname: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return self._size_cache.get(dirname, 0)

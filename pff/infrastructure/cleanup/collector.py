from __future__ import annotations
import os
from pathlib import Path
from pff import settings
from pff.infrastructure.cleanup.file_ops import FileOps


class CleanupScanCollector:
    """Consolidates all filesystem scans into a single traversal for efficiency.

    Acts as a shared cache for cleanup commands that need to scan the project tree.
    """

    def __init__(self, root_dir: Path | None = None):
        self.root_dir = root_dir or settings.ROOT_DIR
        self.ignored_dirs = {
            ".git",
            ".venv",
            "node_modules",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
        }
        self._cache: dict[str, list[Path]] = {}  # dirname -> list of paths
        self._size_cache: dict[str, int] = {}  # dirname -> total size
        self._scanned = False

    def scan(self, target_dirnames: set[str]) -> None:
        """Perform a single walk to find all target directories."""
        if self._scanned:
            return

        for root, dirs, _ in os.walk(self.root_dir):
            # Prune ignored directories
            dirs[:] = [
                d for d in dirs if d not in self.ignored_dirs or d in target_dirnames
            ]

            for target in target_dirnames:
                if target in dirs:
                    path = Path(root) / target
                    if target not in self._cache:
                        self._cache[target] = []
                    self._cache[target].append(path)
                    # Pre-calculate size for previews
                    self._size_cache[target] = self._size_cache.get(
                        target, 0
                    ) + FileOps.calculate_size(path)

        self._scanned = True

    def get_paths(self, dirname: str) -> list[Path]:
        return self._cache.get(dirname, [])

    def get_size(self, dirname: str) -> int:
        return self._size_cache.get(dirname, 0)

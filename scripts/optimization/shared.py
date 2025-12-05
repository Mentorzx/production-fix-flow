"""
Shared Resources Module for HPO Pipeline

Design Patterns:
- Singleton Pattern: Single shared FileManager instance
- Dependency Injection: Enables testability via get_file_manager()

This module provides shared, lazily-initialized resources to avoid
creating multiple FileManager instances across the optimization pipeline.

Usage:
    from scripts.optimization.shared import get_file_manager
    fm = get_file_manager()  # Returns singleton instance
"""

from __future__ import annotations

from functools import lru_cache

from pff.utils.core.file_manager import FileManager


@lru_cache(maxsize=1)
def get_file_manager() -> FileManager:
    """
    Get the shared FileManager singleton instance.

    Uses lru_cache to ensure only one instance is created and reused
    across all modules in the optimization pipeline.

    Returns:
        FileManager: Shared FileManager instance.

    Example:
        >>> from scripts.optimization.shared import get_file_manager
        >>> fm = get_file_manager()
        >>> fm.save(data, path)
    """
    return FileManager()


def reset_file_manager() -> None:
    """
    Reset the FileManager singleton (useful for testing).

    Clears the cached instance so the next get_file_manager() call
    creates a fresh instance.
    """
    get_file_manager.cache_clear()

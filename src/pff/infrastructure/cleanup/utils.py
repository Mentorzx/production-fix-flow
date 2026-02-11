"""Utility functions for cleanup operations.

This module provides helper functions used across the cleanup package.
"""

from __future__ import annotations

import math


def format_size(size_bytes: int) -> str:
    """Format a size in bytes to a human-readable string.

    Converts byte counts to appropriate units (B, KB, MB, GB, TB)
    with two decimal places of precision.

    Args:
        size_bytes: Size in bytes to format.

    Returns:
        Human-readable size string (e.g., "1.5 GB").
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    idx = int(math.floor(math.log(size_bytes, 1024)))
    power = math.pow(1024, idx)
    scaled = round(size_bytes / power, 2)
    return f"{scaled} {size_name[idx]}"


__all__ = ["format_size"]

"""Settings port for application-layer dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class SettingsPort(Protocol):
    """Protocol exposing the subset of settings paths used by application services."""

    DATA_DIR: Path
    OUTPUTS_DIR: Path
    CACHE_DIR: Path
    PATTERNS_DIR: Path

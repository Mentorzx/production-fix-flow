"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/core/test_settings_paths.py

"""

from __future__ import annotations

from pff.shared.core.config import settings


def test_settings_patterns_dir_points_to_repo_patterns() -> None:
    """Execute test settings patterns dir points to repo patterns.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    assert settings.PATTERNS_DIR.is_dir()
    assert (settings.PATTERNS_DIR / "manual_rules.json").is_file()
    assert (settings.PATTERNS_DIR / "schema.json").is_file()


def test_settings_utils_dir_points_to_repo_shared() -> None:
    """Execute test settings utils dir points to repo shared."""

    assert settings.UTILS_DIR.is_dir()

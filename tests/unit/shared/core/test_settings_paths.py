from __future__ import annotations

from pff.shared.core.config import settings


def test_settings_patterns_dir_points_to_repo_patterns() -> None:
    assert settings.PATTERNS_DIR.is_dir()
    assert (settings.PATTERNS_DIR / "manual_rules.json").is_file()
    assert (settings.PATTERNS_DIR / "schema.json").is_file()


def test_settings_utils_dir_points_to_repo_shared() -> None:
    assert settings.UTILS_DIR.is_dir()

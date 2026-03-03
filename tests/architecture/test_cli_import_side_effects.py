"""Architecture guardrail for CLI import-time side effects."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_MAIN_PATH = REPO_ROOT / "src" / "pff" / "drivers" / "cli" / "main.py"


def test_cli_main_must_not_apply_clean_env_at_import_time() -> None:
    """Ensure clean-mode environment is not mutated at module import time."""
    content = CLI_MAIN_PATH.read_text(encoding="utf-8")

    assert "if _is_clean_command(sys.argv):" not in content
    assert "_apply_clean_mode_env(argv)" in content
    assert "def cli_entrypoint()" in content

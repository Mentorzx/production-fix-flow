"""Regression tests for rust-analyzer linker environment settings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _load_vscode_settings() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    settings_path = repo_root / ".vscode" / "settings.json"
    if not settings_path.exists():
        pytest.skip("VSCode settings are not committed in this environment.")
    return json.loads(settings_path.read_text(encoding="utf-8"))


def test_rust_analyzer_cargo_extra_env_is_configured() -> None:
    settings = _load_vscode_settings()

    assert "rust-analyzer.cargo.extraEnv" in settings
    cargo_env = settings["rust-analyzer.cargo.extraEnv"]
    assert isinstance(cargo_env, dict)

    cargo_path = cargo_env.get("PATH")
    assert isinstance(cargo_path, str)
    assert "/usr/bin" in cargo_path
    assert "/home/lira/.local/bin" in cargo_path

    linker = cargo_env.get("CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER")
    assert linker == "/usr/bin/gcc"


def test_rust_analyzer_server_path_includes_usr_bin() -> None:
    settings = _load_vscode_settings()

    assert "rust-analyzer.server.extraEnv" in settings
    server_env = settings["rust-analyzer.server.extraEnv"]
    assert isinstance(server_env, dict)

    server_path = server_env.get("PATH")
    assert isinstance(server_path, str)
    assert "/usr/bin" in server_path

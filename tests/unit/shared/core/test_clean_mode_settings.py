"""Tests for settings bootstrap in clean mode."""

from __future__ import annotations

import importlib


def test_clean_mode_allows_placeholder_secrets(monkeypatch) -> None:
    """Clean mode should not require production secrets during CLI bootstrap."""
    monkeypatch.setenv("PFF_CLEAN_MODE", "1")
    monkeypatch.setenv("SECRET_KEY", "CHANGE_ME_SURELY_IN_PRODUCTION_32_CHAR_MIN")
    monkeypatch.setenv("API_KEY", "CHANGE_ME_SURELY_16_CHAR_MIN")
    monkeypatch.setenv("POSTGRES_PASSWORD", "CHANGE_ME_PASSWORD")

    config_module = importlib.import_module("pff.shared.core.config")
    reloaded = importlib.reload(config_module)

    assert reloaded.settings.SECRET_KEY == "CHANGE_ME_SURELY_IN_PRODUCTION_32_CHAR_MIN"
    assert reloaded.settings.API_KEY == "CHANGE_ME_SURELY_16_CHAR_MIN"
    assert reloaded.settings.POSTGRES_PASSWORD == "CHANGE_ME_PASSWORD"

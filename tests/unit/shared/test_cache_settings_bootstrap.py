"""Tests for explicit cache settings bootstrap behavior."""

from __future__ import annotations

from pff.shared.core.cache import constants as cache_constants


def test_apply_cache_settings_from_config_is_idempotent(monkeypatch) -> None:
    """Settings should apply once unless forced."""
    monkeypatch.setattr(cache_constants, "_CACHE_SETTINGS_APPLIED", False)
    monkeypatch.setattr(cache_constants, "DEFAULT_LRU_SIZE", 128)
    monkeypatch.setattr(
        cache_constants,
        "_load_cache_settings",
        lambda: {"lru_size": 256},
    )

    assert cache_constants.apply_cache_settings_from_config() is True
    assert cache_constants.DEFAULT_LRU_SIZE == 256

    monkeypatch.setattr(
        cache_constants,
        "_load_cache_settings",
        lambda: {"lru_size": 512},
    )
    assert cache_constants.apply_cache_settings_from_config() is False
    assert cache_constants.DEFAULT_LRU_SIZE == 256

    assert cache_constants.apply_cache_settings_from_config(force=True) is True
    assert cache_constants.DEFAULT_LRU_SIZE == 512


def test_apply_cache_settings_from_config_keeps_flag_false_without_payload(
    monkeypatch,
) -> None:
    """No payload must not mark cache settings as applied."""
    monkeypatch.setattr(cache_constants, "_CACHE_SETTINGS_APPLIED", False)
    monkeypatch.setattr(cache_constants, "_load_cache_settings", lambda: {})

    assert cache_constants.apply_cache_settings_from_config() is False
    assert cache_constants._CACHE_SETTINGS_APPLIED is False

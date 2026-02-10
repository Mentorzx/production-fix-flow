"""Tests for compile flag plumbing in HPO defaults."""

from __future__ import annotations

from pff.infrastructure.hpo import config_loader


def test_load_hpo_defaults_reads_use_compile_from_training(monkeypatch) -> None:
    """`use_compile` should be loaded from dslfm_kgc.training in optimization config."""
    fake_cfg = {
        "dslfm_kgc": {
            "training": {"use_compile": True},
        }
    }
    monkeypatch.setattr(config_loader, "load_optimization_config", lambda *_a, **_k: fake_cfg)

    defaults = config_loader.load_hpo_defaults()

    assert defaults["use_compile"] is True


def test_load_hpo_defaults_use_compile_defaults_to_false(monkeypatch) -> None:
    """When absent, `use_compile` should default to False for HPO safety."""
    fake_cfg = {
        "dslfm_kgc": {
            "training": {},
        }
    }
    monkeypatch.setattr(config_loader, "load_optimization_config", lambda *_a, **_k: fake_cfg)

    defaults = config_loader.load_hpo_defaults()

    assert defaults["use_compile"] is False

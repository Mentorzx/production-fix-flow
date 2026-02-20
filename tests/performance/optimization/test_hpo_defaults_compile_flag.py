"""Tests for compile flag plumbing in HPO defaults."""

from __future__ import annotations

from pff.infrastructure.hpo import config_loader


def test_load_hpo_defaults_reads_use_compile_from_training(monkeypatch) -> None:
    """`use_compile` should be loaded from dslfm_kgc.training in optimization config."""
    fake_cfg = {
        "dslfm_kgc": {
            "training": {"use_compile": True},
            "pc": {
                "pruning_threshold_low": 1e-4,
                "pruning_threshold_high": 1e-1,
                "rebuild_every_low": 0,
                "rebuild_every_high": 10,
            },
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
            "pc": {
                "pruning_threshold_low": 1e-4,
                "pruning_threshold_high": 1e-1,
                "rebuild_every_low": 0,
                "rebuild_every_high": 10,
            },
        }
    }
    monkeypatch.setattr(config_loader, "load_optimization_config", lambda *_a, **_k: fake_cfg)

    defaults = config_loader.load_hpo_defaults()

    assert defaults["use_compile"] is False


def test_load_hpo_defaults_reads_pc_thresholds(monkeypatch) -> None:
    """PC pruning/rebuild defaults should be loaded from optimization config."""
    fake_cfg = {
        "dslfm_kgc": {
            "training": {},
            "pc": {
                "pruning_threshold_low": 1e-4,
                "pruning_threshold_high": 0.1,
                "rebuild_every_low": 5,
                "rebuild_every_high": 25,
            },
        }
    }
    monkeypatch.setattr(config_loader, "load_optimization_config", lambda *_a, **_k: fake_cfg)

    defaults = config_loader.load_hpo_defaults()

    assert defaults["pruning_threshold_low"] == 1e-4
    assert defaults["pruning_threshold_high"] == 0.1
    assert defaults["rebuild_every_low"] == 5
    assert defaults["rebuild_every_high"] == 25

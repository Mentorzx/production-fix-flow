"""Tests for multi-objective HPO config parsing."""

from __future__ import annotations


def test_load_multi_objective_defaults():
    """Defaults should disable multi-objective and use MOTPE."""
    from pff.infrastructure.hpo import config_loader

    settings = config_loader.load_multi_objective_settings()

    assert settings["enabled"] is False
    assert settings["sampler"] == "motpe"
    assert settings["directions"] == ["maximize", "maximize", "minimize"]
    assert settings["secondary_metric"] == "mcc"


def test_load_multi_objective_custom(monkeypatch):
    """Custom settings should override defaults."""
    from pff.infrastructure.hpo import config_loader

    def _fake_loader(*_args, **_kwargs):
        return {
            "multi_objective": {
                "enabled": True,
                "sampler": "nsga2",
                "directions": ["maximize", "minimize"],
                "secondary_metric": "auc",
                "population_size": 80,
                "mutation_prob": 0.2,
                "crossover_prob": 0.7,
            }
        }

    monkeypatch.setattr(config_loader, "load_optimization_config", _fake_loader)

    settings = config_loader.load_multi_objective_settings()

    assert settings["enabled"] is True
    assert settings["sampler"] == "nsga2"
    assert settings["secondary_metric"] == "auc"
    assert settings["population_size"] == 80
    assert settings["mutation_prob"] == 0.2
    assert settings["crossover_prob"] == 0.7

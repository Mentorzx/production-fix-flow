"""Tests for live plot config parsing."""

from __future__ import annotations


def test_load_live_plot_settings_defaults(monkeypatch):
    """Default live plot settings should be enabled with standard dashboard defaults."""
    from pff.infrastructure.hpo import config_loader

    monkeypatch.setattr(config_loader, "_read_hpo_config", lambda *_a, **_k: {})

    settings = config_loader.load_live_plot_settings()

    assert settings["enabled"] is True
    assert settings["max_trials_axis"] == 50
    assert settings["output_subdir"] == "optimization/plots/live"
    assert settings["enable_optuna_dashboard"] is False
    assert settings["dashboard_interval"] == 5
    assert settings["dashboard_top_n"] == 12
    assert settings["dashboard_data_path"] is None
    assert settings["dashboard_debug_mode"] is False


def test_load_live_plot_settings_custom(monkeypatch):
    """Custom live plot settings should override defaults."""
    from pff.infrastructure.hpo import config_loader

    def _fake_loader(*_args, **_kwargs):
        return {
            "live_plots": {
                "enabled": False,
                "max_trials_axis": 120,
                "output_subdir": "optimization/plots/live_custom",
                "enable_optuna_dashboard": True,
                "dashboard_interval": 15,
                "dashboard_top_n": 7,
                "dashboard_data_path": "outputs/.cache/hpo/dashboard_data.json",
                "dashboard_debug_mode": True,
            }
        }

    monkeypatch.setattr(config_loader, "load_optimization_config", _fake_loader)

    settings = config_loader.load_live_plot_settings()

    assert settings["enabled"] is False
    assert settings["max_trials_axis"] == 120
    assert settings["output_subdir"] == "optimization/plots/live_custom"
    assert settings["enable_optuna_dashboard"] is True
    assert settings["dashboard_interval"] == 15
    assert settings["dashboard_top_n"] == 7
    assert settings["dashboard_data_path"] == "outputs/.cache/hpo/dashboard_data.json"
    assert settings["dashboard_debug_mode"] is True

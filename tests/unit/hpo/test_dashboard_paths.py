"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/hpo/test_dashboard_paths.py

"""

from __future__ import annotations

import importlib

from pff.shared.core.file_manager import FileManager


def test_collect_dashboard_data_paths_includes_cache_and_live_plot(tmp_path, monkeypatch):
    """Execute test collect dashboard data paths includes cache and live plot.



    Args:

        tmp_path: Input value used by this callable.

        monkeypatch: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    outputs_dir = tmp_path / "outputs"
    cache_dir = outputs_dir / ".cache"
    live_plot_dir = outputs_dir / "optimization" / "plots"
    study_cache_dir = cache_dir / "hpo" / "study_a"

    FileManager.ensure_dir(live_plot_dir)
    FileManager.ensure_dir(study_cache_dir)

    FileManager().save({"trials": []}, live_plot_dir / "dashboard_data.json")
    FileManager().save({"trials": []}, study_cache_dir / "dashboard_data.json")

    server = importlib.import_module("pff.infrastructure.hpo.dashboard.server")

    monkeypatch.setattr(server, "BASE_DIR", tmp_path)
    monkeypatch.setattr(server.settings, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(server.settings, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "DATA_CACHE_PATH", cache_dir / "hpo" / "dashboard_data.json")
    monkeypatch.setattr(
        server,
        "load_live_plot_settings",
        lambda: {"output_subdir": "optimization/plots"},
    )

    paths = server._collect_dashboard_data_paths()

    assert (live_plot_dir / "dashboard_data.json") in paths
    assert (study_cache_dir / "dashboard_data.json") in paths


def test_collect_dashboard_data_paths_uses_cache(tmp_path, monkeypatch):
    """Execute test collect dashboard data paths uses cache.



    Args:

        tmp_path: Input value used by this callable.

        monkeypatch: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    outputs_dir = tmp_path / "outputs"
    cache_dir = outputs_dir / ".cache"
    live_plot_dir = outputs_dir / "optimization" / "plots"

    FileManager.ensure_dir(live_plot_dir)

    server = importlib.import_module("pff.infrastructure.hpo.dashboard.server")

    monkeypatch.setattr(server, "BASE_DIR", tmp_path)
    monkeypatch.setattr(server.settings, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(server.settings, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "DATA_CACHE_PATH", cache_dir / "hpo" / "dashboard_data.json")

    calls = {"count": 0}

    def _fake_settings():
        calls["count"] += 1
        return {"output_subdir": "optimization/plots"}

    monkeypatch.setattr(server, "load_live_plot_settings", _fake_settings)
    monkeypatch.setattr(server.time, "time", lambda: 1234.0)

    server._reset_dashboard_paths_cache()
    server._collect_dashboard_data_paths()
    server._collect_dashboard_data_paths()

    assert calls["count"] == 1

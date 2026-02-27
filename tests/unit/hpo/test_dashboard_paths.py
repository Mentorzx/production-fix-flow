"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/hpo/test_dashboard_paths.py

"""

from __future__ import annotations

import importlib
import os

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


def test_load_raw_dashboard_data_prefers_most_recent_payload_when_study_hint_is_missing(
    tmp_path, monkeypatch
):
    """Execute test load raw dashboard data prefers most recent payload when no study hint.



    Args:

        tmp_path: Input value used by this callable.

        monkeypatch: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    outputs_dir = tmp_path / "outputs"
    cache_dir = outputs_dir / ".cache"
    canonical_path = cache_dir / "hpo" / "dashboard_data.json"
    nested_path = cache_dir / "hpo" / "study_a" / "dashboard_data.json"

    FileManager.ensure_dir(canonical_path.parent)
    FileManager.ensure_dir(nested_path.parent)

    FileManager().save(
        {"studyName": "canonical", "totalTrials": 50, "updatedAt": "2026-02-26T23:36:20+00:00"},
        canonical_path,
    )
    FileManager().save(
        {
            "studyName": "nested_newer",
            "totalTrials": 12,
            "updatedAt": "2026-02-26T23:36:23+00:00",
        },
        nested_path,
    )
    os.utime(canonical_path, (100.0, 100.0))
    os.utime(nested_path, (200.0, 200.0))

    server = importlib.import_module("pff.infrastructure.hpo.dashboard.server")
    monkeypatch.setattr(server, "BASE_DIR", tmp_path)
    monkeypatch.setattr(server.settings, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(server.settings, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "DATA_CACHE_PATH", canonical_path)
    monkeypatch.setattr(
        server,
        "load_live_plot_settings",
        lambda: {
            "output_subdir": "optimization/plots",
            "dashboard_data_path": str(canonical_path),
        },
    )
    server._reset_dashboard_paths_cache()

    payload = server._load_raw_dashboard_data()

    assert payload.get("studyName") == "nested_newer"
    assert payload.get("totalTrials") == 12


def test_load_raw_dashboard_data_prefers_active_study_even_if_another_is_newer(
    tmp_path, monkeypatch
):
    """Execute test load raw dashboard data prefers active study payload when available."""
    outputs_dir = tmp_path / "outputs"
    cache_dir = outputs_dir / ".cache"
    canonical_path = cache_dir / "hpo" / "dashboard_data.json"
    nested_path = cache_dir / "hpo" / "study_b" / "dashboard_data.json"

    FileManager.ensure_dir(canonical_path.parent)
    FileManager.ensure_dir(nested_path.parent)

    FileManager().save(
        {
            "studyName": "study_a",
            "totalTrials": 50,
            "updatedAt": "2026-02-26T23:36:20+00:00",
        },
        canonical_path,
    )
    FileManager().save(
        {
            "studyName": "study_b",
            "totalTrials": 12,
            "updatedAt": "2026-02-26T23:36:23+00:00",
        },
        nested_path,
    )
    os.utime(canonical_path, (100.0, 100.0))
    os.utime(nested_path, (200.0, 200.0))

    server = importlib.import_module("pff.infrastructure.hpo.dashboard.server")
    monkeypatch.setattr(server, "BASE_DIR", tmp_path)
    monkeypatch.setattr(server.settings, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(server.settings, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(server, "DATA_CACHE_PATH", canonical_path)
    monkeypatch.setattr(
        server,
        "load_live_plot_settings",
        lambda: {
            "output_subdir": "optimization/plots",
            "dashboard_data_path": str(canonical_path),
        },
    )
    server._reset_dashboard_paths_cache()

    payload = server._load_raw_dashboard_data(active_study_name="study_a")

    assert payload.get("studyName") == "study_a"
    assert payload.get("totalTrials") == 50


def test_apply_study_defaults_normalizes_total_trials_and_count_fields():
    """Execute test apply study defaults normalizes total trials and count fields."""
    server = importlib.import_module("pff.infrastructure.hpo.dashboard.server")

    raw_data = {
        "studyName": "study_a",
        "totalTrials": "invalid",
        "trials": [
            {"id": 1, "state": "COMPLETE", "warmstart": True},
            {"id": 2, "state": "COMPLETE", "warmstart": False},
            {"id": 3, "state": "RUNNING", "warmstart": False},
        ],
    }
    server._apply_study_defaults(raw_data)

    assert raw_data["totalTrials"] == 50
    assert raw_data["total_trials_target"] == 50
    assert raw_data["completed_trials_all"] == 2
    assert raw_data["completed_trials_non_warmstart"] == 1
    assert raw_data["warmstart_trials"] == 1


def test_apply_study_defaults_uses_total_trials_target_as_source():
    """Execute test apply study defaults uses total trials target as source."""
    server = importlib.import_module("pff.infrastructure.hpo.dashboard.server")

    raw_data = {
        "studyName": "study_b",
        "totalTrials": 12,
        "total_trials_target": 37,
        "trials": [],
    }
    server._apply_study_defaults(raw_data)

    assert raw_data["total_trials_target"] == 37
    assert raw_data["totalTrials"] == 37

import importlib

from pff.shared.core.file_manager import FileManager


def test_collect_dashboard_data_paths_includes_cache_and_live_plot(
    tmp_path, monkeypatch
):
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
    monkeypatch.setattr(
        server, "DATA_CACHE_PATH", cache_dir / "hpo" / "dashboard_data.json"
    )
    monkeypatch.setattr(
        server,
        "load_live_plot_settings",
        lambda: {"output_subdir": "optimization/plots"},
    )

    paths = server._collect_dashboard_data_paths()

    assert (live_plot_dir / "dashboard_data.json") in paths
    assert (study_cache_dir / "dashboard_data.json") in paths

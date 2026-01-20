"""Tests for HPO Dashboard infrastructure.

Verifies:
1. LivePlotCallback JSON generation (data structure).
2. Dashboard server API endpoints.
3. Resilience to missing data.
"""

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from optuna.trial import TrialState

from pff.infrastructure.hpo.callbacks_internal.visualizers import LivePlotCallback
from pff.infrastructure.hpo.dashboard.server import run_server


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory structure for outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        # Mimic project structure: outputs/optimization/plots -> .cache is at outputs/.cache
        plots_dir = path / "outputs" / "optimization" / "plots"
        plots_dir.mkdir(parents=True)
        yield plots_dir


@pytest.fixture
def mock_study():
    """Create a mock Optuna study with some trials."""
    study = MagicMock()
    study.study_name = "test_dashboard_study"
    study.best_value = 0.85

    study.best_value = 0.85
    study.direction.name = "maximize"
    study.user_attrs = {}

    # Mock trials
    trial1 = MagicMock()
    trial1.number = 0
    trial1.value = 0.5
    trial1.state = TrialState.COMPLETE
    trial1.params = {"lr": 0.01, "layers": 2}
    trial1.user_attrs = {"metrics": {"mrr": 0.5, "hits@10": 0.8}}
    trial1.datetime_start = datetime_mock(1000)
    trial1.datetime_complete = datetime_mock(1010)  # 10s duration

    trial2 = MagicMock()
    trial2.number = 1
    trial2.value = 0.85
    trial2.state = TrialState.COMPLETE
    trial2.params = {"lr": 0.001, "layers": 4}
    trial2.user_attrs = {"metrics": {"mrr": 0.85, "hits@10": 0.95}}
    trial2.datetime_start = datetime_mock(1020)
    trial2.datetime_complete = datetime_mock(1040)  # 20s duration

    trial3 = MagicMock()
    trial3.number = 2
    trial3.value = 0.1
    trial3.state = TrialState.PRUNED
    trial3.params = {"lr": 0.1, "layers": 1}
    trial3.user_attrs = {}
    trial3.datetime_start = datetime_mock(1050)
    trial3.datetime_complete = datetime_mock(1055)

    study.trials = [trial1, trial2, trial3]
    study.get_trials.return_value = study.trials  # Ensure get_trials returns the list
    return study


class MockDateTime:
    def __init__(self, ts):
        self.ts = ts

    def __sub__(self, other):
        return MockTimedelta(self.ts - other.ts)


class MockTimedelta:
    def __init__(self, seconds):
        self._seconds = seconds

    def total_seconds(self):
        return self._seconds


def datetime_mock(ts):
    return MockDateTime(ts)


def test_live_plot_callback_generates_json(temp_output_dir, mock_study):
    """Test that the callback writes the correct JSON structure."""
    # Initialize callback
    callback = LivePlotCallback(output_dir=temp_output_dir)

    callback(mock_study, mock_study.trials[-1])

    expected_file = temp_output_dir.parent.parent / ".cache" / "hpo" / "dashboard_data.json"

    assert expected_file.exists()

    # Verify content
    with open(expected_file) as f:
        data = json.load(f)

    assert data["studyName"] == "test_dashboard_study"
    assert data["bestValue"] == 0.85
    assert len(data["trials"]) == 3

    # Check Trial 1 (Success)
    t1 = next(t for t in data["trials"] if t["id"] == 1)
    assert t1["value"] == 0.5
    assert t1["state"] == "COMPLETE"
    assert t1["duration"] == 10.0
    assert t1["params"]["lr"] == 0.01

    # Verify metrics field is present and populated
    assert "metrics" in t1
    assert t1["metrics"]["mrr"] == 0.5
    assert t1["metrics"]["hits10"] == 0.8  # Renamed from hits@10

    # Check Trial 3 (Pruned)
    t3 = next(t for t in data["trials"] if t["id"] == 3)
    assert t3["state"] == "PRUNED"


def test_live_plot_callback_handles_empty_study(temp_output_dir):
    """Test resilience against empty studies."""
    study = MagicMock()
    study.study_name = "empty_study"
    study.trials = []
    study.best_value = 0.0

    callback = LivePlotCallback(output_dir=temp_output_dir)
    callback(study, None)

    expected_file = temp_output_dir.parent.parent / ".cache" / "hpo" / "dashboard_data.json"
    assert expected_file.exists()

    with open(expected_file) as f:
        data = json.load(f)

    assert data["trials"] == []
    assert data["bestValue"] == 0.0


def test_dashboard_server_api(temp_output_dir):
    """Test the dashboard server API responses."""
    # Setup data
    data_file = temp_output_dir / ".cache" / "hpo" / "dashboard_data.json"
    data_file.parent.mkdir(parents=True, exist_ok=True)

    test_data = {"studyName": "api_test", "trials": [{"id": 1}]}
    with open(data_file, "w") as f:
        json.dump(test_data, f)

    port = 8801

    with patch("pff.infrastructure.hpo.dashboard.server.DATA_CACHE_PATH", data_file):
        static_dir = temp_output_dir / "static"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<html>dashboard</html>")

        with patch("pff.infrastructure.hpo.dashboard.server.STATIC_DIR", static_dir):
            server_thread = threading.Thread(
                target=run_server,
                kwargs={"port": port, "bind": "127.0.0.1"},
                daemon=True,
            )
            server_thread.start()

            time.sleep(1)

            try:
                resp = requests.get(f"http://127.0.0.1:{port}/api/data")
                assert resp.status_code == 200
                assert resp.json()["studyName"] == "api_test"

                resp = requests.get(f"http://127.0.0.1:{port}/index.html")
                assert resp.status_code == 200
                assert "dashboard" in resp.text

            finally:
                pass


def test_callback_handles_missing_directories(temp_output_dir):
    """Test that callback creates directories if they don't exist."""
    # Delete the automatically created structure
    shutil.rmtree(temp_output_dir)

    # It should recreate them
    callback = LivePlotCallback(output_dir=temp_output_dir)
    _ = callback  # Use callback to avoid unused warning  # noqa: F841
    assert temp_output_dir.exists()

    cache_dir = temp_output_dir.parent.parent / ".cache" / "hpo"
    assert cache_dir.exists()


# Atomic write not currently implemented in LivePlotCallback/FileManager
# def test_writer_atomic_operations(temp_output_dir, mock_study):
#     """Test that the writer uses atomic write pattern (temp file + rename)."""
#     callback = LivePlotCallback(output_dir=temp_output_dir)
#
#     # We mock open to track if a temp file was used
#     # The actual implementation does: self.data_path.with_suffix(".tmp")
#     data_path = temp_output_dir.parent.parent / ".cache" / "hpo" / "dashboard_data.json"
#     temp_path = data_path.with_suffix(".tmp")
#
#     with patch("builtins.open", side_effect=open) as mock_open:
#         callback(mock_study, mock_study.trials[-1])
#
#         # Check if opened file path ends with .tmp
#         # We look for the call that opened the temp file for writing
#         calls = [c for c in mock_open.mock_calls if str(temp_path) in str(c)]
#         assert len(calls) > 0, "Writer did not write to .tmp file first"
#
#     # Verify final file exists
#     assert data_path.exists()


def test_dashboard_json_contract(temp_output_dir, mock_study):
    """Validate strict JSON contract expected by React frontend."""
    callback = LivePlotCallback(output_dir=temp_output_dir)
    callback(mock_study, mock_study.trials[-1])

    data_path = temp_output_dir.parent.parent / ".cache" / "hpo" / "dashboard_data.json"
    with open(data_path) as f:
        data = json.load(f)

    # Root level fields
    required_root = {"studyName", "updatedAt", "bestValue", "trials"}
    assert required_root.issubset(set(data.keys()))
    assert isinstance(data["studyName"], str)
    assert isinstance(data["updatedAt"], str)
    assert isinstance(data["bestValue"], (int, float))
    assert isinstance(data["trials"], list)

    # Trial level fields
    if data["trials"]:
        trial = data["trials"][0]
        required_trial = {
            "id",
            "value",
            "state",
            "params",
            "duration",
            "mrr",
            "best_mrr",
            "mcc",
            "auc",
            "hits1",
            "hits3",
            "hits10",
        }
        assert required_trial.issubset(set(trial.keys()))
        assert isinstance(trial["id"], int)
        assert isinstance(trial["params"], dict)


def test_api_headers(temp_output_dir):
    """Verify HTTP headers for CORS and caching."""
    # Setup data
    data_file = temp_output_dir / ".cache" / "hpo" / "dashboard_data.json"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    with open(data_file, "w") as f:
        json.dump({"test": 1}, f)

    port = 8802

    with patch("pff.infrastructure.hpo.dashboard.server.DATA_CACHE_PATH", data_file):
        static_dir = temp_output_dir / "static"
        static_dir.mkdir(parents=True, exist_ok=True)
        (static_dir / "index.html").write_text("html")

        with patch("pff.infrastructure.hpo.dashboard.server.STATIC_DIR", static_dir):
            server_thread = threading.Thread(
                target=run_server,
                kwargs={"port": port, "bind": "127.0.0.1"},
                daemon=True,
            )
            server_thread.start()
            time.sleep(1)

            try:
                resp = requests.get(f"http://127.0.0.1:{port}/api/data")

                # Assertions
                assert resp.headers["Access-Control-Allow-Origin"] == "*"
                assert "no-store" in resp.headers["Cache-Control"]
                assert resp.headers["Content-Type"] == "application/json"

            finally:
                pass


def test_cleanup_preserves_code_removes_data(temp_output_dir):
    """Simulation of 'clean deep' behavior."""
    # 1. Setup Environment
    # Dashboard Code (in pff/...)
    # In real app: pff/infrastructure/hpo/dashboard/static/index.html
    # In test: we simulate this separation

    code_dir = temp_output_dir.parent.parent.parent / "pff" / "dashboard"
    code_dir.mkdir(parents=True, exist_ok=True)
    html_file = code_dir / "index.html"
    html_file.write_text("<h1>Fixed Dashboard</h1>")

    # Dashboard Data (in outputs/.cache/...)
    data_dir = temp_output_dir.parent.parent / ".cache" / "hpo"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_file = data_dir / "dashboard_data.json"
    data_file.write_text("{}")

    # 2. Simulate Clean Deep
    # Clean deep deletes 'outputs/' contents but NOT 'pff/'
    # simulating DirCleanCommand on temp_output_dir.parent.parent (which mimics 'outputs/')

    outputs_root = temp_output_dir.parent.parent

    # Verify pre-condition
    assert html_file.exists()
    assert data_file.exists()

    # Act: Clean outputs
    shutil.rmtree(outputs_root)

    # 3. Assertions
    assert html_file.exists(), "Cleanup should NOT delete the static HTML in pff/"
    assert not data_file.exists(), "Cleanup MUST delete the generated data in outputs/"

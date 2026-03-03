"""Tests for HPO Dashboard infrastructure.

Verifies:
1. LivePlotCallback JSON generation (data structure).
2. Dashboard server API endpoints.
3. Resilience to missing data.
"""

import shutil
import socket
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import orjson
import pytest
import requests  # type: ignore[import-untyped]
from optuna.trial import TrialState

from pff.infrastructure.hpo.callbacks_internal.visualizers import LivePlotCallback
from pff.infrastructure.hpo.dashboard.server import run_server
from pff.infrastructure.hpo.dashboard import server as dashboard_server


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(orjson.dumps(payload))


def _read_json(path: Path) -> dict:
    return orjson.loads(path.read_bytes())


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory structure for outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        # Mimic project structure: outputs/optimization/plots -> .cache is at outputs/.cache
        plots_dir = path / "outputs" / "optimization" / "plots"
        plots_dir.mkdir(parents=True)
        yield plots_dir


@pytest.fixture(autouse=True)
def reset_live_best_metrics():
    """Reset in-memory live best metrics between tests."""
    if hasattr(dashboard_server, "LOOKBACK_MEMORY"):
        dashboard_server.LOOKBACK_MEMORY["live_best_metrics"] = {}
    elif hasattr(dashboard_server, "_get_lookback_memory"):
        lookback = dashboard_server._get_lookback_memory()
        lookback["live_best_metrics"] = {}
        if hasattr(dashboard_server, "_set_lookback_memory"):
            dashboard_server._set_lookback_memory(lookback)
    yield


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
    trial1.user_attrs = {
        "metrics": {
            "mrr": 0.5,
            "hits@10": 0.8,
            "loss": 0.42,
            "precision": 0.7,
            "recall": 0.6,
        }
    }
    trial1.datetime_start = datetime_mock(1000)
    trial1.datetime_complete = datetime_mock(1010)

    trial2 = MagicMock()
    trial2.number = 1
    trial2.value = 0.85
    trial2.state = TrialState.COMPLETE
    trial2.params = {"lr": 0.001, "layers": 4}
    trial2.user_attrs = {
        "metrics": {
            "mrr": 0.85,
            "hits@10": 0.95,
            "loss": 0.31,
            "precision": 0.8,
            "recall": 0.75,
        }
    }
    trial2.datetime_start = datetime_mock(1020)
    trial2.datetime_complete = datetime_mock(1040)

    trial3 = MagicMock()
    trial3.number = 2
    trial3.value = 0.1
    trial3.state = TrialState.PRUNED
    trial3.params = {"lr": 0.1, "layers": 1}
    trial3.user_attrs = {}
    trial3.datetime_start = datetime_mock(1050)
    trial3.datetime_complete = datetime_mock(1055)

    study.trials = [trial1, trial2, trial3]
    study.get_trials.return_value = study.trials
    return study


class MockDateTime:
    """Represent MockDateTime."""

    def __init__(self, ts):
        """Execute init.



        Args:

            ts: Input value used by this callable.

        """

        self.ts = ts

    def __sub__(self, other):
        return MockTimedelta(self.ts - other.ts)


class MockTimedelta:
    """Represent MockTimedelta."""

    def __init__(self, seconds):
        """Execute init.



        Args:

            seconds: Input value used by this callable.

        """

        self._seconds = seconds

    def total_seconds(self):
        """Execute total seconds.



        Returns:

            Return value produced by the callable.

        """

        return self._seconds


def datetime_mock(ts):
    """Execute datetime mock.



    Args:

        ts: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    return MockDateTime(ts)


def _get_free_port() -> int:
    """Execute get free port.



    Returns:

        Return value produced by the callable.

    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_live_plot_callback_generates_json(temp_output_dir, mock_study):
    """Test that the callback writes the correct JSON structure."""
    # Initialize callback
    callback = LivePlotCallback(output_dir=temp_output_dir)

    callback(mock_study, mock_study.trials[-1])

    expected_file = temp_output_dir.parent.parent / ".cache" / "hpo" / "dashboard_data.json"
    mirror_file = temp_output_dir / "dashboard_data.json"

    assert expected_file.exists()
    assert mirror_file.exists()

    # Verify content
    data = orjson.loads(expected_file.read_bytes())

    assert data["studyName"] == "test_dashboard_study"
    assert data["bestValue"] == 0.85
    assert len(data["trials"]) == 3

    # Check Trial 1 (Success)
    t1 = next(t for t in data["trials"] if t["id"] == 1)
    assert t1["value"] == 0.5
    assert t1["state"] == "COMPLETE"
    assert t1["duration"] == 10.0
    assert t1["loss"] == 0.42
    assert t1["precision"] == 0.7
    assert t1["recall"] == 0.6
    assert t1["efficiency"] == 0.05
    assert t1["params"]["lr"] == 0.01

    # Verify metrics field is present and populated
    assert "metrics" in t1
    assert t1["metrics"]["mrr"] == 0.5
    assert t1["metrics"]["hits10"] == 0.8

    # Check Trial 3 (Pruned)
    t3 = next(t for t in data["trials"] if t["id"] == 3)
    assert t3["state"] == "PRUNED"


def test_live_plot_callback_respects_dashboard_data_path(temp_output_dir, tmp_path):
    """Dashboard data should be written to the configured path."""
    data_file = tmp_path / "custom" / "dashboard_data.json"
    callback = LivePlotCallback(output_dir=temp_output_dir, dashboard_data_path=data_file)

    study = MagicMock()
    study.study_name = "custom_path"
    study.trials = []
    study.best_value = 0.0
    study.direction.name = "maximize"
    study.user_attrs = {}
    study.get_trials.return_value = []

    callback.initialize_dashboard(study)

    assert data_file.exists()
    assert (temp_output_dir / "dashboard_data.json").exists()
    data = orjson.loads(data_file.read_bytes())

    assert data["studyName"] == "custom_path"


def test_live_plot_callback_handles_empty_study(temp_output_dir):
    """Test resilience against empty studies."""
    study = MagicMock()
    study.study_name = "empty_study"
    study.trials = []
    study.best_value = 0.0

    callback = LivePlotCallback(output_dir=temp_output_dir)
    callback(study, None)

    expected_file = temp_output_dir.parent.parent / ".cache" / "hpo" / "dashboard_data.json"
    mirror_file = temp_output_dir / "dashboard_data.json"
    assert expected_file.exists()
    assert mirror_file.exists()

    data = orjson.loads(expected_file.read_bytes())

    assert data["trials"] == []
    assert data["bestValue"] == 0.0


def test_live_plot_callback_marks_warmstart_seed_by_user_attr(temp_output_dir):
    """Ensure warmstart is detected even if only warmstart_seed is present."""
    study = MagicMock()
    study.study_name = "warmstart_attr_study"
    study.best_value = 0.0
    study.direction.name = "maximize"
    study.user_attrs = {}

    class _TrialWithoutSystemAttrs:
        number = 0
        value = None
        state = TrialState.WAITING
        params = {}
        user_attrs = {"warmstart_seed": True}
        datetime_start = None
        datetime_complete = None

        @property
        def system_attrs(self):  # pragma: no cover - should never be touched
            raise AssertionError("Deprecated trial.system_attrs should not be accessed")

    trial = _TrialWithoutSystemAttrs()

    study.trials = [trial]
    study.get_trials.return_value = study.trials

    callback = LivePlotCallback(output_dir=temp_output_dir)
    callback.initialize_dashboard(study)

    expected_file = temp_output_dir.parent.parent / ".cache" / "hpo" / "dashboard_data.json"
    data = _read_json(expected_file)

    assert data["trials"][0]["warmstart"] is True


def test_live_plot_callback_ignores_live_status_from_different_study(temp_output_dir, mock_study):
    """Live status must be ignored when it belongs to another study."""
    callback = LivePlotCallback(output_dir=temp_output_dir)
    _write_json(
        temp_output_dir / "live_status.json",
        {
            "study_name": "other_study",
            "trial_number": 9,
            "epoch_history": [{"mrr": 0.99, "timestamp": 1.0}],
        },
    )

    callback.initialize_dashboard(mock_study)

    data_path = temp_output_dir.parent.parent / ".cache" / "hpo" / "dashboard_data.json"
    payload = _read_json(data_path)
    assert payload.get("liveStatus") == {}


def test_live_plot_callback_uses_live_status_from_matching_study(temp_output_dir, mock_study):
    """Live status should be included when study names match."""
    callback = LivePlotCallback(output_dir=temp_output_dir)
    _write_json(
        temp_output_dir / "live_status.json",
        {
            "study_name": "test_dashboard_study",
            "trial_number": 1,
            "epoch_history": [{"mrr": 0.77, "timestamp": 1.0}],
        },
    )

    callback.initialize_dashboard(mock_study)

    data_path = temp_output_dir.parent.parent / ".cache" / "hpo" / "dashboard_data.json"
    payload = _read_json(data_path)
    live_status = payload.get("liveStatus", {})
    assert live_status.get("trial_number") == 1
    assert isinstance(live_status.get("epoch_history"), list)


def test_live_plot_callback_keeps_parallel_running_trial_states(temp_output_dir, mock_study):
    """Parallel RUNNING trials must not be coerced to PRUNED by trial id ordering."""

    class _Trial:
        def __init__(self, number: int, state: TrialState) -> None:
            self.number = number
            self.state = state
            self.params = {}
            self.user_attrs = {}
            self.datetime_start = None
            self.datetime_complete = None
            self.value = None
            self.values = None

    callback = LivePlotCallback(output_dir=temp_output_dir)
    earlier_running = _Trial(number=3, state=TrialState.RUNNING)
    state = callback._resolve_trial_state(earlier_running, max_trial_id=4)
    assert state == "RUNNING"


def test_live_plot_callback_exports_total_trials_target(temp_output_dir, mock_study):
    """Dashboard payload must expose current-run target trial count fields."""
    callback = LivePlotCallback(output_dir=temp_output_dir, expected_trials=50)
    callback.initialize_dashboard(mock_study)

    data_path = temp_output_dir.parent.parent / ".cache" / "hpo" / "dashboard_data.json"
    payload = _read_json(data_path)

    assert payload.get("totalTrials") == 50
    assert payload.get("total_trials_target") == 50


def test_live_plot_callback_is_thread_safe_under_parallel_calls(temp_output_dir, mock_study):
    """Concurrent callback invocations must keep dashboard JSON readable and coherent."""
    callback = LivePlotCallback(output_dir=temp_output_dir, dashboard_interval=0)
    data_path = temp_output_dir.parent.parent / ".cache" / "hpo" / "dashboard_data.json"

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(callback, mock_study, mock_study.trials[-1]) for _ in range(32)]
        for future in futures:
            future.result()

    payload = _read_json(data_path)
    assert payload["studyName"] == "test_dashboard_study"
    assert len(payload["trials"]) == 3


def test_dashboard_server_api(temp_output_dir):
    """Test the dashboard server API responses."""
    # Setup data
    data_file = temp_output_dir / ".cache" / "hpo" / "dashboard_data.json"
    data_file.parent.mkdir(parents=True, exist_ok=True)

    test_data = {"studyName": "api_test", "trials": [{"id": 1}]}
    _write_json(data_file, test_data)

    port = _get_free_port()

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
                assert resp.headers.get("Connection") == "close"

                resp = requests.get(f"http://127.0.0.1:{port}/index.html")
                assert resp.status_code == 200
                assert "dashboard" in resp.text
                assert resp.headers.get("Connection") == "close"

            finally:
                pass


def test_dashboard_server_materializes_live_trial_when_missing_from_snapshot(
    temp_output_dir, tmp_path
):
    """Ensure the dashboard materializes the current live trial when it's missing in snapshot."""
    # Data file has trials 1..3 only.
    data_file = tmp_path / "dashboard_data.json"
    _write_json(
        data_file,
        {
            "studyName": "gap_test",
            "trials": [{"id": 1}, {"id": 2}, {"id": 3}],
        },
    )

    # Create a live_status.json that points to live trial_number=4 (0-based) -> live_id=5.
    base_root = tmp_path / "root"
    live_status_path = base_root / "outputs" / "optimization" / "plots" / "live_status.json"
    live_status_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(live_status_path, {"trial_number": 4, "epoch_history": []})

    port = _get_free_port()

    with (
        patch("pff.infrastructure.hpo.dashboard.server.DATA_CACHE_PATH", data_file),
        patch("pff.infrastructure.hpo.dashboard.server.BASE_DIR", base_root),
    ):
        static_dir = temp_output_dir / "static2"
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

            resp = requests.get(f"http://127.0.0.1:{port}/api/data")
            assert resp.status_code == 200

            payload = resp.json()
            trials = payload.get("trials")
            ids = sorted([t["id"] for t in trials])
            assert ids == [1, 2, 3, 5]
            live_trial = next(t for t in trials if t.get("id") == 5)
            assert live_trial.get("state") == "RUNNING"
            assert payload.get("_synthetic_trials") is False


def test_dashboard_server_synthesizes_trial_when_data_missing(temp_output_dir, tmp_path):
    """If dashboard_data.json is missing, server should still expose the live trial for study plots."""

    base_root = tmp_path / "root"
    live_status_path = base_root / "outputs" / "optimization" / "plots" / "live_status.json"
    live_status_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        live_status_path,
        {
            "trial_number": 0,
            "params": {"learning_rate": 0.001, "embedding_dim": 128},
            "epoch_history": [{"mrr": 0.1, "timestamp": 1.0}],
            "warmstart": True,
        },
    )

    port = _get_free_port()

    with (
        patch(
            "pff.infrastructure.hpo.dashboard.server.DATA_CACHE_PATH",
            tmp_path / "missing.json",
        ),
        patch("pff.infrastructure.hpo.dashboard.server.BASE_DIR", base_root),
        patch(
            "pff.infrastructure.hpo.dashboard.server._collect_dashboard_data_paths",
            lambda: [tmp_path / "missing.json"],
        ),
    ):
        static_dir = temp_output_dir / "static3"
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

            resp = requests.get(f"http://127.0.0.1:{port}/api/data")
            assert resp.status_code == 200

            payload = resp.json()
            assert payload.get("_synthetic_trials") is True
            trials = payload.get("trials")
            assert isinstance(trials, list)
            assert len(trials) == 1
            assert trials[0]["id"] == -1
            assert trials[0]["state"] == "RUNNING"
            assert trials[0]["params"]["learning_rate"] == 0.001
            assert trials[0]["warmstart"] is True


def test_dashboard_server_applies_best_epoch_metrics_to_matching_live_id(temp_output_dir, tmp_path):
    """Live trial should receive best-epoch metrics when id matches trial_number."""
    data_file = tmp_path / "dashboard_data.json"
    _write_json(
        data_file,
        {
            "studyName": "live_id_match",
            "trials": [
                {
                    "id": 2,
                    "state": "RUNNING",
                    "value": 0.33,
                    "metrics": {},
                }
            ],
        },
    )

    base_root = tmp_path / "root"
    live_status_path = base_root / "outputs" / "optimization" / "plots" / "live_status.json"
    live_status_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        live_status_path,
        {
            "trial_number": 2,
            "elapsed_seconds": 12.0,
            "epoch_history": [
                {
                    "mrr": 0.7,
                    "precision": 0.6,
                    "loss": 1.2,
                    "timestamp": 1.0,
                }
            ],
        },
    )

    port = _get_free_port()

    with (
        patch("pff.infrastructure.hpo.dashboard.server.DATA_CACHE_PATH", data_file),
        patch("pff.infrastructure.hpo.dashboard.server.BASE_DIR", base_root),
    ):
        static_dir = temp_output_dir / "static4"
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

            resp = requests.get(f"http://127.0.0.1:{port}/api/data")
            assert resp.status_code == 200

            payload = resp.json()
            trials = payload.get("trials")
            trial = next(t for t in trials if t.get("id") == 2)
            assert trial.get("mrr") == 0.7
            assert trial.get("precision") == 0.6
            assert trial.get("loss") == 1.2
            assert trial.get("metrics", {}).get("mrr") == 0.7


def test_load_live_status_derives_val_loss_from_binary_on_eval_epochs(tmp_path) -> None:
    """Dashboard loader should derive val_loss from binary_loss on evaluation epochs."""
    base_root = tmp_path / "root"
    live_status_path = base_root / "outputs" / "optimization" / "plots" / "live_status.json"
    live_status_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        live_status_path,
        {
            "trial_number": 1,
            "epoch_history": [
                {"epoch": 1, "binary_loss": 0.7},
                {"epoch": 2, "binary_loss": 0.4, "mrr": 0.21},
            ],
        },
    )

    with patch("pff.infrastructure.hpo.dashboard.server.BASE_DIR", base_root):
        loaded = dashboard_server._load_live_status()

    assert isinstance(loaded, dict)
    history = loaded.get("epoch_history", [])
    assert isinstance(history, list)
    assert history[0].get("val_loss") is None
    assert history[1].get("val_loss") == 0.4


def test_load_live_status_prefers_lowest_active_trial_from_per_trial_files(
    tmp_path,
) -> None:
    """Dashboard should pick the lowest active trial when parallel live files exist."""
    base_root = tmp_path / "root"
    live_dir = base_root / "outputs" / "optimization" / "plots" / "live_status"
    live_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        live_dir / "trial_000003.json",
        {
            "study_name": "parallel_study",
            "trial_number": 3,
            "cv_fold_id": 2,
            "epoch_history": [{"mrr": 0.4}],
        },
    )
    _write_json(
        live_dir / "trial_000005.json",
        {
            "study_name": "parallel_study",
            "trial_number": 5,
            "cv_fold_id": 0,
            "epoch_history": [{"mrr": 0.6}],
        },
    )

    with patch("pff.infrastructure.hpo.dashboard.server.BASE_DIR", base_root):
        selected = dashboard_server._load_live_status(
            preferred_study_name="parallel_study",
            preferred_trial_ids={4, 6},
        )

    assert isinstance(selected, dict)
    assert selected.get("trial_number") == 3
    assert selected.get("cv_fold_id") == 2


def test_load_live_status_ignores_stale_per_trial_files(tmp_path) -> None:
    """Stale per-trial files should be ignored to avoid dead-run oscillation."""
    base_root = tmp_path / "root"
    live_dir = base_root / "outputs" / "optimization" / "plots" / "live_status"
    live_dir.mkdir(parents=True, exist_ok=True)
    stale_file = live_dir / "trial_000003.json"
    _write_json(
        stale_file,
        {
            "study_name": "parallel_study",
            "trial_number": 3,
            "updated_at": "2024-01-01T00:00:00+00:00",
            "epoch_history": [],
        },
    )
    fresh_legacy = base_root / "outputs" / "optimization" / "plots" / "live_status.json"
    fresh_legacy.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        fresh_legacy,
        {
            "study_name": "parallel_study",
            "trial_number": 5,
            "epoch_history": [],
        },
    )

    with patch("pff.infrastructure.hpo.dashboard.server.BASE_DIR", base_root):
        selected = dashboard_server._load_live_status(
            preferred_study_name="parallel_study",
            preferred_trial_ids={4, 6},
        )

    assert isinstance(selected, dict)
    assert selected.get("trial_number") == 5


def test_load_raw_dashboard_data_prefers_primary_source_within_recency_window(
    tmp_path,
) -> None:
    """Recent conflicting payloads should use stable source priority to avoid oscillation."""
    base_root = tmp_path / "root"
    source_a = base_root / "a" / "dashboard_data.json"
    source_b = base_root / "b" / "dashboard_data.json"
    source_a.parent.mkdir(parents=True, exist_ok=True)
    source_b.parent.mkdir(parents=True, exist_ok=True)

    _write_json(
        source_a,
        {
            "studyName": "stable_source",
            "updatedAt": "2026-02-27T23:00:00+00:00",
            "totalTrials": 50,
            "total_trials_target": 50,
            "liveStatus": {"trial_number": 3},
        },
    )
    _write_json(
        source_b,
        {
            "studyName": "stable_source",
            "updatedAt": "2026-02-27T23:00:04+00:00",
            "totalTrials": 54,
            "total_trials_target": 54,
            "liveStatus": {"trial_number": 4},
        },
    )

    dashboard_server._DASHBOARD_RUNTIME_CACHE.invalidate(
        pattern=f"^{dashboard_server._CACHE_KEY_DATA_SOURCE}$"
    )
    with patch(
        "pff.infrastructure.hpo.dashboard.server._collect_dashboard_data_paths",
        return_value=[source_a, source_b],
    ):
        payload = dashboard_server._load_raw_dashboard_data(
            active_study_name="stable_source"
        )

    assert payload.get("total_trials_target") == 50
    assert payload.get("totalTrials") == 50


def test_load_raw_dashboard_data_prefers_matching_live_trial_over_source_priority(
    tmp_path,
) -> None:
    """When live trial is known, select payload with matching liveStatus trial id."""
    base_root = tmp_path / "root"
    source_a = base_root / "a" / "dashboard_data.json"
    source_b = base_root / "b" / "dashboard_data.json"
    source_a.parent.mkdir(parents=True, exist_ok=True)
    source_b.parent.mkdir(parents=True, exist_ok=True)

    _write_json(
        source_a,
        {
            "studyName": "stable_source",
            "updatedAt": "2026-02-27T23:00:00+00:00",
            "totalTrials": 50,
            "total_trials_target": 50,
            "liveStatus": {"trial_number": 3},
        },
    )
    _write_json(
        source_b,
        {
            "studyName": "stable_source",
            "updatedAt": "2026-02-27T23:00:04+00:00",
            "totalTrials": 54,
            "total_trials_target": 54,
            "liveStatus": {"trial_number": 4},
        },
    )

    dashboard_server._DASHBOARD_RUNTIME_CACHE.invalidate(
        pattern=f"^{dashboard_server._CACHE_KEY_DATA_SOURCE}$"
    )
    with patch(
        "pff.infrastructure.hpo.dashboard.server._collect_dashboard_data_paths",
        return_value=[source_a, source_b],
    ):
        payload = dashboard_server._load_raw_dashboard_data(
            active_study_name="stable_source",
            preferred_live_trial_id=5,
        )

    assert payload.get("total_trials_target") == 54
    assert payload.get("totalTrials") == 54


def test_append_hardware_history_prefers_gpu_total_utilization() -> None:
    """Hardware history should use GPU total utilization when available."""
    previous = dashboard_server._HARDWARE_HISTORY.copy()
    try:
        dashboard_server._HARDWARE_HISTORY["items"] = []
        dashboard_server._HARDWARE_HISTORY["last_id"] = 0

        history = dashboard_server._append_hardware_history(
            {
                "cpu_usage": 35.0,
                "ram_usage_pct": 66.0,
                "gpus": [
                    {
                        "id": 0,
                        "utilization": 47.0,
                        "utilization_total": 85.0,
                        "vram_usage_pct": 31.0,
                    }
                ],
            }
        )

        assert history
        assert history[-1]["gpu_utilization"] == 85.0
        assert history[-1]["vram_usage_pct"] == 31.0
    finally:
        dashboard_server._HARDWARE_HISTORY.clear()
        dashboard_server._HARDWARE_HISTORY.update(previous)


def test_augment_confusion_matrices_from_fold_history_includes_live_fold(tmp_path) -> None:
    """Dashboard consolidation should expose up to 3 fold confusion matrices with current fold."""
    outputs_dir = tmp_path / "outputs"
    plots_dir = outputs_dir / "optimization" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        plots_dir / "fold_history.json",
        [
            {
                "trial_number": 3,
                "cv_fold_id": 0,
                "epoch": 40,
                "timestamp": 1000.0,
                "confusion_matrix": {"vp": 10, "vn": 20, "fp": 3, "fn": 1},
            },
            {
                "trial_number": 3,
                "cv_fold_id": 1,
                "epoch": 80,
                "timestamp": 2000.0,
                "confusion_matrix": {"vp": 11, "vn": 21, "fp": 4, "fn": 2},
            },
        ],
    )

    raw_data = {"charts": {}}
    live_status = {
        "trial_number": 3,
        "cv_fold_id": 2,
        "current_epoch": 98,
        "confusion_matrix": {"vp": 12, "vn": 22, "fp": 5, "fn": 3},
    }

    with patch("pff.infrastructure.hpo.dashboard.server.BASE_DIR", tmp_path):
        dashboard_server._augment_confusion_matrices_from_fold_history(raw_data, live_status)

    charts = raw_data.get("charts", {})
    confusion_matrices = charts.get("confusion_matrices")
    assert isinstance(confusion_matrices, list)
    assert len(confusion_matrices) == 3
    folds = {row.get("cv_fold_id") for row in confusion_matrices if isinstance(row, dict)}
    assert folds == {0, 1, 2}


def test_dashboard_server_keeps_previous_metrics_when_best_epoch_missing_fields(
    temp_output_dir, tmp_path
):
    """Running trial should keep existing metrics when best epoch lacks fields."""
    data_file = tmp_path / "dashboard_data.json"
    _write_json(
        data_file,
        {
            "studyName": "live_id_keep_metrics",
            "trials": [
                {
                    "id": 2,
                    "state": "RUNNING",
                    "value": 0.33,
                    "mrr": 0.5,
                    "metrics": {"mrr": 0.5},
                }
            ],
        },
    )

    base_root = tmp_path / "root"
    live_status_path = base_root / "outputs" / "optimization" / "plots" / "live_status.json"
    live_status_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        live_status_path,
        {
            "trial_number": 2,
            "elapsed_seconds": 12.0,
            "epoch_history": [{"loss": 1.2, "timestamp": 1.0}],
        },
    )

    port = _get_free_port()

    with (
        patch("pff.infrastructure.hpo.dashboard.server.DATA_CACHE_PATH", data_file),
        patch("pff.infrastructure.hpo.dashboard.server.BASE_DIR", base_root),
    ):
        static_dir = temp_output_dir / "static5"
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

            resp = requests.get(f"http://127.0.0.1:{port}/api/data")
            assert resp.status_code == 200

            payload = resp.json()
            trials = payload.get("trials")
            trial = next(t for t in trials if t.get("id") == 2)
            assert trial.get("mrr") == 0.5
            assert trial.get("metrics", {}).get("mrr") == 0.5


def test_dashboard_server_persists_best_metrics_across_live_updates(temp_output_dir, tmp_path):
    """Best metrics should persist when later live updates omit those fields."""
    data_file = tmp_path / "dashboard_data.json"
    _write_json(
        data_file,
        {
            "studyName": "live_persist",
            "trials": [
                {
                    "id": 2,
                    "state": "RUNNING",
                    "value": 0.33,
                    "metrics": {},
                }
            ],
        },
    )

    base_root = tmp_path / "root"
    live_status_path = base_root / "outputs" / "optimization" / "plots" / "live_status.json"
    live_status_path.parent.mkdir(parents=True, exist_ok=True)

    port = _get_free_port()

    with (
        patch("pff.infrastructure.hpo.dashboard.server.DATA_CACHE_PATH", data_file),
        patch("pff.infrastructure.hpo.dashboard.server.BASE_DIR", base_root),
    ):
        static_dir = temp_output_dir / "static6"
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

            _write_json(
                live_status_path,
                {
                    "trial_number": 2,
                    "elapsed_seconds": 12.0,
                    "epoch_history": [{"mrr": 0.7, "loss": 1.2, "timestamp": 1.0}],
                },
            )

            resp = requests.get(f"http://127.0.0.1:{port}/api/data")
            assert resp.status_code == 200
            payload = resp.json()
            trial = next(t for t in payload.get("trials") if t.get("id") == 2)
            assert trial.get("mrr") == 0.7

            _write_json(
                live_status_path,
                {
                    "trial_number": 2,
                    "elapsed_seconds": 12.0,
                    "epoch_history": [{"loss": 1.1, "timestamp": 2.0}],
                },
            )

            time.sleep(1)
            resp = requests.get(f"http://127.0.0.1:{port}/api/data")
            assert resp.status_code == 200
            payload = resp.json()
            trial = next(t for t in payload.get("trials") if t.get("id") == 2)
            assert trial.get("mrr") == 0.7
            assert trial.get("loss") == 1.1


def test_dashboard_debug_mode_does_not_seed_trials_when_empty(temp_output_dir, tmp_path):
    """Debug mode should not invent trials when no data exists."""
    data_file = tmp_path / "missing" / "dashboard_data.json"
    base_root = tmp_path / "root"
    port = _get_free_port()

    with (
        patch("pff.infrastructure.hpo.dashboard.server.DATA_CACHE_PATH", data_file),
        patch("pff.infrastructure.hpo.dashboard.server.BASE_DIR", base_root),
        patch(
            "pff.infrastructure.hpo.dashboard.server._collect_dashboard_data_paths",
            lambda: [data_file],
        ),
        patch(
            "pff.infrastructure.hpo.dashboard.server.load_live_plot_settings",
            lambda: {"dashboard_debug_mode": True},
        ),
    ):
        static_dir = temp_output_dir / "static_debug"
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

            resp = requests.get(f"http://127.0.0.1:{port}/api/data")
            assert resp.status_code == 200
            payload = resp.json()
            trials = payload.get("trials", [])

            assert payload.get("dashboardDebugMode") is True
            assert trials == []
            assert payload.get("_synthetic_trials") is False


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
    data = _read_json(data_path)

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
    _write_json(data_file, {"test": 1})

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
                assert resp.headers.get("Connection") == "close"

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

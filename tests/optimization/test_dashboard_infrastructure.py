"""Infrastructure and Contract tests for HPO Dashboard.

Ensures:
1. Dashboard code is strictly separated from data (never in outputs/).
2. Static artifacts are present.
3. Server contract (endpoints exist).
"""

import requests
import threading
import time
from unittest.mock import patch

from pff import settings
from pff.infrastructure.hpo.dashboard.server import run_server


def test_dashboard_is_not_inside_outputs():
    """Ensure dashboard HTML never mistakenly ends up in outputs/."""
    outputs = settings.OUTPUTS_DIR
    forbidden = [
        outputs / "live_dashboard.html",
        outputs / "optimization" / "plots" / "live_dashboard.html",
    ]
    # We check they don't exist.
    # Note: If previous runs left them, this might fail, but it's a good guard.
    for p in forbidden:
        assert not p.exists(), f"Found forbidden dashboard artifact at {p}"


def test_dashboard_static_artifacts_exist():
    """Ensure minimal static assets exist in the correct location."""
    # This assumes the source-code location, not necessarily the installed package location
    # depending on how tests run. We look relative to settings.ROOT_DIR

    base = settings.ROOT_DIR / "pff/infrastructure/hpo/dashboard"
    static = base / "static"

    assert base.exists(), "Dashboard module missing"
    assert (base / "server.py").exists(), "Server module missing"
    assert (static / "index.html").exists(), "Index.html missing"

    # Check index content sanity
    html = (static / "index.html").read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "react" in html.lower()


def test_dashboard_server_contract(tmp_path):
    """Verify server endpoints contract (index and API)."""
    # Setup mock data
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    data_file = data_dir / "dashboard_data.json"
    data_file.write_text('{"studyName": "Contract Test", "trials": []}')

    # Use real static dir
    real_static = settings.ROOT_DIR / "pff/infrastructure/hpo/dashboard/static"

    port = 8803

    with (
        patch("pff.infrastructure.hpo.dashboard.server.DATA_CACHE_PATH", data_file),
        patch("pff.infrastructure.hpo.dashboard.server.STATIC_DIR", real_static),
    ):
        server_thread = threading.Thread(
            target=run_server, kwargs={"port": port, "bind": "127.0.0.1"}, daemon=True
        )
        server_thread.start()
        time.sleep(1)

        base_url = f"http://127.0.0.1:{port}"

        try:
            # 1. Root serves HTML
            r = requests.get(base_url + "/")
            assert r.status_code == 200
            assert "text/html" in r.headers["Content-Type"]
            assert "<html" in r.text

            # 2. API serves JSON
            r = requests.get(base_url + "/api/data")
            assert r.status_code == 200
            assert "application/json" in r.headers["Content-Type"]
            assert r.json()["studyName"] == "Contract Test"

            # 3. Status API
            r = requests.get(base_url + "/api/status")
            assert r.status_code == 200
            assert "application/json" in r.headers["Content-Type"]

        finally:
            pass

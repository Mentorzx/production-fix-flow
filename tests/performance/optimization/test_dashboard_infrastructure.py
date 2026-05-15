"""Infrastructure and Contract tests for HPO Dashboard.

Ensures:
1. Dashboard code is strictly separated from data (never in outputs/).
2. Static artifacts are present.
3. Server contract (endpoints exist).
"""

import socket
import threading
import time
from unittest.mock import patch

import requests  # type: ignore[import-untyped]

from pff import settings
from pff.infrastructure.hpo.dashboard.server import run_server


def _get_free_port() -> int:
    """Get a free TCP port from the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
    base = settings.PACKAGE_DIR / "infrastructure" / "hpo" / "dashboard"
    dist = base / "dist"
    static = base / "static"

    assert base.exists(), "Dashboard module missing"
    assert (base / "server.py").exists(), "Server module missing"
    assert (base / "build_dashboard.sh").exists(), "Dashboard build script missing"
    assert (base / "package.json").exists(), "Dashboard package manifest missing"
    assert (static / "index.html").exists(), "Index.html missing"

    if dist.exists():
        assert (dist / "dashboard.js").exists(), "Dashboard JS bundle missing"
        assert (dist / "dashboard.css").exists(), "Dashboard CSS bundle missing"
        assert (dist / "version.json").exists(), "Dashboard version metadata missing"

    # Check index content sanity
    html = (static / "index.html").read_text(encoding="utf-8")
    assert "<!doctype html>" in html.lower()
    assert "react" in html.lower()


def test_dashboard_server_contract(tmp_path):
    """Verify server endpoints contract (index and API)."""
    # Setup mock data
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    data_file = data_dir / "dashboard_data.json"
    data_file.write_text('{"studyName": "Contract Test", "trials": []}')

    temp_dist = tmp_path / "dist"
    temp_dist.mkdir()
    (temp_dist / "dashboard.js").write_text("console.log('contract');", encoding="utf-8")
    (temp_dist / "dashboard.css").write_text("body { color: #000; }", encoding="utf-8")

    # Use real static dir
    base = settings.PACKAGE_DIR / "infrastructure" / "hpo" / "dashboard"
    real_static = base / "static"

    port = _get_free_port()

    with (
        patch("pff.infrastructure.hpo.dashboard.server.DATA_CACHE_PATH", data_file),
        patch("pff.infrastructure.hpo.dashboard.server.STATIC_DIR", real_static),
        patch("pff.infrastructure.hpo.dashboard.server.DIST_DIR", temp_dist),
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

            # 4. Dist assets must resolve against the patched dist directory.
            r = requests.get(base_url + "/dist/dashboard.js")
            assert r.status_code == 200
            assert "console.log('contract');" in r.text

        finally:
            pass

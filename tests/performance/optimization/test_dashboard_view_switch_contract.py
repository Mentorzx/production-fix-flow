"""Static checks to prevent view-switch starvation under frequent SSE updates.

The dashboard receives frequent updates (SSE + clock ticks). View switching must be immediate
to avoid the UI keeping stale content (e.g., Forecast tab study layout persisting in Trial view).
"""

from __future__ import annotations

from pff.shared.core.config import settings


def test_store_view_mode_switch_is_not_wrapped_in_transition() -> None:
    path = (
        settings.PACKAGE_DIR
        / "infrastructure"
        / "hpo"
        / "dashboard"
        / "static"
        / "js"
        / "store"
        / "store.jsx"
    )
    assert path.exists(), "store.jsx missing"

    content = path.read_text(encoding="utf-8", errors="ignore")
    assert (
        "setViewMode: (mode) => startTransition" not in content
    ), "setViewMode must be immediate (not wrapped in startTransition) to prevent starvation."

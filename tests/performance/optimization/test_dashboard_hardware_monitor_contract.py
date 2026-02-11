"""Static checks for HardwareMonitorCard design contracts (docs are the objective).

Contracts:
1. Hardware history chart must not render a Recharts Legend (names live in the bars above).
2. Metric labels (CPU/GPU/VRAM/RAM) must use hintboxes.
"""

from __future__ import annotations

from pff.shared.core.config import settings


def test_hardware_monitor_card_has_no_legend_and_metric_hintboxes() -> None:
    path = (
        settings.PACKAGE_DIR
        / "infrastructure"
        / "hpo"
        / "dashboard"
        / "static"
        / "js"
        / "features"
        / "hpo"
        / "charts"
        / "HardwareMonitorCard.jsx"
    )
    assert path.exists(), "HardwareMonitorCard.jsx missing"

    content = path.read_text(encoding="utf-8", errors="ignore")

    assert "<Legend" not in content, "Hardware monitor chart must not render a Legend"

    # Hintboxes are driven by MetricRegistry + PortalTooltip.
    assert (
        "MetricRegistry" in content
    ), "Hardware monitor labels must use MetricRegistry hints"
    assert (
        "PortalTooltip" in content
    ), "Hardware monitor labels must show hintboxes via PortalTooltip"

    for key in ('key: "cpu"', 'key: "gpu"', 'key: "vram"', 'key: "ram"'):
        assert key in content, f"Expected metric key mapping for {key}"

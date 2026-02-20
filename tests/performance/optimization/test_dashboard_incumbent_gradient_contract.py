"""Static checks for incumbent chart area gradient visibility contract."""

from __future__ import annotations

from pff.shared.core.config import settings


def test_incumbent_chart_uses_explicit_gradient_area_contract() -> None:
    """Execute test incumbent chart uses explicit gradient area contract."""

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
        / "IncumbentTrajectoryCard.jsx"
    )
    assert path.exists(), "IncumbentTrajectoryCard.jsx missing"

    content = path.read_text(encoding="utf-8", errors="ignore")
    assert "grad-objective-" in content, "Incumbent chart must define objective gradient id"
    assert "getChartAreaGradientStops(" in content, "Incumbent chart must use gradient token helper"
    assert "fill={`url(#grad-objective-" in content, "Area must fill from objective gradient"
    assert "baseValue={yDomain[0]}" in content, "Area must use explicit numeric base value"

"""Static checks to prevent chart sizing regressions."""

from pathlib import Path


def test_dashboard_sources_do_not_use_responsive_container() -> None:
    """Ensure chart sources avoid ResponsiveContainer to prevent size warnings."""
    root = Path("pff/infrastructure/hpo/dashboard/static/js")
    assert root.exists(), "Dashboard source root missing"

    offenders = []
    for path in root.rglob("*.[jt]sx"):
        if path.name == "app.js":
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if "ResponsiveContainer" in content:
            offenders.append(str(path))

    assert not offenders, f"ResponsiveContainer found in: {offenders}"


def test_loss_projection_card_has_min_height_guard() -> None:
    """Ensure LossProjectionCard keeps a non-zero minHeight to avoid invisible charts."""
    path = Path("pff/infrastructure/hpo/dashboard/static/js/features/hpo/charts/LossProjectionCard.jsx")
    assert path.exists(), "LossProjectionCard.jsx missing"

    content = path.read_text(encoding="utf-8", errors="ignore")
    assert "minHeight={120}" in content, "LossProjectionCard should enforce a minimum height"
    assert "minHeight={0}" not in content, "LossProjectionCard should not set minHeight=0"

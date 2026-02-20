"""Static checks for deterministic KPI number animation contracts."""

from __future__ import annotations

from pff.shared.core.config import settings


def _read_dashboard_js(path_suffix: str) -> str:
    path = (
        settings.PACKAGE_DIR
        / "infrastructure"
        / "hpo"
        / "dashboard"
        / "static"
        / "js"
        / path_suffix
    )
    assert path.exists(), f"missing file: {path_suffix}"
    return path.read_text(encoding="utf-8", errors="ignore")


def test_animated_number_component_exposes_targeted_roll_contract() -> None:
    """Execute test animated number component exposes targeted roll contract."""

    content = _read_dashboard_js("ui/AnimatedNumberText.jsx")
    assert "data-jackpot-target" in content, (
        "Animated number component must expose explicit target marker"
    )
    assert "data-jackpot-skip" in content, (
        "Animated number component must skip global scanner overlap"
    )
    assert "prefers-reduced-motion" in content, (
        "Animated number must respect reduced motion preference"
    )


def test_kpi_row_uses_scope_and_tab_seed_for_stable_animation() -> None:
    """Execute test kpi row uses scope and tab seed for stable animation."""

    content = _read_dashboard_js("layout/KpiRow.jsx")
    assert "scope:${viewMode}|tab:${activeTab}" in content, (
        "Kpi animation seed must include scope and active tab."
    )
    assert "<AnimatedNumberText" in content, (
        "Kpi row duration nodes must use deterministic animated numbers"
    )


def test_trial_status_and_stat_badge_are_seed_aware() -> None:
    """Execute test trial status and stat badge are seed aware."""

    trial_status = _read_dashboard_js("features/hpo/charts/TrialStatusCard.jsx")
    stat_badge = _read_dashboard_js("ui/StatBadge.jsx")

    assert "animationSeed" in trial_status, "TrialStatusCard must accept animationSeed"
    assert "<AnimatedNumberText" in trial_status, (
        "TrialStatusCard must animate the trial headline digits"
    )

    assert "animationSeed" in stat_badge, "StatBadge must accept animationSeed"
    assert "<AnimatedNumberText" in stat_badge, (
        "StatBadge must animate numeric value deterministically"
    )

"""Contract tests for HPO Dashboard layout.

These checks are intentionally static (source-based) to keep them fast and
deterministic while protecting key UI invariants used in production.
"""

from pff.shared.core.config import settings

_DASHBOARD_JS_ROOT = settings.PACKAGE_DIR / "infrastructure" / "hpo" / "dashboard" / "static" / "js"
_OVERVIEW_TAB = _DASHBOARD_JS_ROOT / "layout" / "OverviewTab.jsx"
_DASHBOARD = _DASHBOARD_JS_ROOT / "layout" / "Dashboard.jsx"
_KPI_ROW = _DASHBOARD_JS_ROOT / "layout" / "KpiRow.jsx"
_FORECAST_TAB = _DASHBOARD_JS_ROOT / "layout" / "ForecastTab.jsx"
_REGRESSION_CHART = _DASHBOARD_JS_ROOT / "features" / "hpo" / "charts" / "RegressionChartCard.jsx"


def _read_overview_tab() -> str:
    assert _OVERVIEW_TAB.exists(), f"Missing dashboard source: {_OVERVIEW_TAB}"
    return _OVERVIEW_TAB.read_text(encoding="utf-8", errors="strict")


def _read_forecast_tab() -> str:
    assert _FORECAST_TAB.exists(), f"Missing dashboard source: {_FORECAST_TAB}"
    return _FORECAST_TAB.read_text(encoding="utf-8", errors="strict")


def _read_dashboard() -> str:
    assert _DASHBOARD.exists(), f"Missing dashboard source: {_DASHBOARD}"
    return _DASHBOARD.read_text(encoding="utf-8", errors="strict")


def _read_kpi_row() -> str:
    assert _KPI_ROW.exists(), f"Missing dashboard source: {_KPI_ROW}"
    return _KPI_ROW.read_text(encoding="utf-8", errors="strict")


def _read_regression_chart() -> str:
    assert _REGRESSION_CHART.exists(), f"Missing dashboard source: {_REGRESSION_CHART}"
    return _REGRESSION_CHART.read_text(encoding="utf-8", errors="strict")


def test_overview_tab_does_not_define_inline_trial_status_card() -> None:
    """TrialStatusCard must be a shared component to keep parity across views."""
    content = _read_overview_tab()
    assert "const TrialStatusCard" not in content, (
        "OverviewTab.jsx should import TrialStatusCard instead of defining it inline."
    )


def test_kpi_row_keeps_trial_card_first() -> None:
    """Trial card must be the first KPI card across views (docs are the objective)."""
    content = _read_kpi_row()

    trial_card_idx = content.find("<TrialStatusCard")
    assert trial_card_idx != -1, "Expected TrialStatusCard in KpiRow.jsx"

    best_global_idx = content.find('label="Melhor Global"')
    assert best_global_idx != -1, "Expected 'Melhor Global' StatBadge in KpiRow.jsx"

    assert trial_card_idx < best_global_idx, (
        "TrialStatusCard must come before 'Melhor Global' in the KPI row."
    )


def test_kpi_row_is_rendered_persistently_in_dashboard() -> None:
    """The KPI row must be rendered outside per-tab conditionals to persist across tabs."""
    content = _read_dashboard()

    kpi_idx = content.find("<KpiRow")
    assert kpi_idx != -1, "Expected KpiRow to be rendered in Dashboard.jsx"

    overview_idx = content.find('activeTab === "overview"')
    assert overview_idx != -1, "Expected tab panels in Dashboard.jsx"

    assert kpi_idx < overview_idx, (
        "KpiRow must be rendered before tab panels to persist across tabs"
    )


def test_overview_tab_trial_view_shows_full_metrics_log_in_monitoring() -> None:
    """Trial monitoring must show the full epoch metrics log (docs are the objective)."""
    content = _read_overview_tab()

    trial_idx = content.find("// View Mode: Trial")
    assert trial_idx != -1, "Expected trial view branch in OverviewTab.jsx"
    trial_view = content[trial_idx:]

    assert "<FullMetricsLogCard" in trial_view, (
        "Expected FullMetricsLogCard in trial monitoring view"
    )
    assert "<GeneralizationGapCard" not in trial_view, (
        "GeneralizationGapCard (optimization dynamics) should not be in monitoring for trial view."
    )


def test_overview_tab_monitoring_has_stable_section_heights_across_views() -> None:
    """Study and Trial monitoring layouts must keep stable main heights to avoid flicker.

    Note: bottom tables are allowed to grow dynamically (design target in docs).
    """
    content = _read_overview_tab()

    study_idx = content.find('if (viewMode === "study")')
    assert study_idx != -1, "Expected study view branch in OverviewTab.jsx"

    trial_idx = content.find("// View Mode: Trial")
    assert trial_idx != -1, "Expected trial view branch marker in OverviewTab.jsx"

    study_view = content[study_idx:trial_idx]
    trial_view = content[trial_idx:]

    assert "h-[480px]" in study_view, (
        "Expected fixed main section height (480px) in study monitoring view"
    )
    assert "h-[480px]" in trial_view, (
        "Expected fixed main section height (480px) in trial monitoring view"
    )
    assert "h-[360px]" not in content, (
        "Bottom tables must not be constrained to a fixed height in monitoring"
    )


def test_forecast_tab_includes_optimization_dynamics_for_trial_view() -> None:
    """Optimization dynamics chart must live under Forecast for trial view."""
    content = _read_forecast_tab()

    assert 'if (viewMode === "trial")' in content, (
        "Expected ForecastTab to branch on viewMode for trial view"
    )
    assert "<GeneralizationGapCard" in content, (
        "Expected GeneralizationGapCard in ForecastTab trial view"
    )


def test_forecast_tab_mounts_local_optima_card_only_in_study_view() -> None:
    """Local-optima diagnostics must live in study-mode Forecast only."""
    content = _read_forecast_tab()

    trial_idx = content.find('if (viewMode === "trial")')
    assert trial_idx != -1, "Expected ForecastTab to branch on viewMode for trial view"
    study_return_idx = content.find("\n  return (", trial_idx)
    assert study_return_idx != -1, "Expected ForecastTab to return a study-mode layout"
    trial_view = content[trial_idx:study_return_idx]
    study_view = content[study_return_idx:]

    assert "sectionKey=\"forecast-local-optima\"" in study_view, (
        "ForecastTab study view must define the local-optima diagnostics section."
    )
    assert "<LocalOptimaDiagnosticsCard" in study_view, (
        "ForecastTab study view must mount LocalOptimaDiagnosticsCard."
    )
    assert "<LocalOptimaDiagnosticsCard" not in trial_view, (
        "LocalOptimaDiagnosticsCard must not render in ForecastTab trial view."
    )


def test_forecast_tab_passes_total_trials_to_regression_chart() -> None:
    """Regression chart must use configured total trial horizon from the store."""
    content = _read_forecast_tab()
    assert "totalTrials={data.totalTrials || DEFAULT_TOTAL_TRIALS}" in content, (
        "ForecastTab must pass totalTrials to RegressionChartCard."
    )


def test_regression_chart_uses_fixed_axis_domains() -> None:
    """Regression chart axes must stay fixed to deterministic study bounds."""
    content = _read_regression_chart()
    assert "domain={[1, safeTotalTrials]}" in content, (
        "Expected fixed X-axis domain from trial 1 to total trial horizon."
    )
    assert "domain={[0, 1]}" in content, "Expected fixed Y-axis domain in score space [0,1]."

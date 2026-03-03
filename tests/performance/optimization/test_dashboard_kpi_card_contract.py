"""Static checks for KPI cards (top row) to match the dashboard docs target.

The dashboard docs are the objective. These contracts protect:
1. Trial card shows epoch progress (current/total).
2. KPI StatBadge supports delta %, direction cue, hover highlight, and hintbox tooltip.
3. KPI row keeps consistent card sizing across the 4 KPIs (avoid an oversized Trial card).
4. KPI animation seed wiring exists for deterministic macro/micro transitions.
"""

from __future__ import annotations

from pff.shared.core.config import settings


def test_trial_status_card_includes_epoch_progress_and_total() -> None:
    """Execute test trial status card includes epoch progress and total.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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
        / "TrialStatusCard.jsx"
    )
    assert path.exists(), "TrialStatusCard.jsx missing"

    content = path.read_text(encoding="utf-8", errors="ignore")
    assert "Época" in content, "TrialStatusCard must display the 'Época' label"
    assert "current_epoch" in content, "TrialStatusCard must reference current_epoch"
    assert "total_epochs" in content, "TrialStatusCard must reference total_epochs"


def test_trial_status_card_prioritizes_active_trial_ids() -> None:
    """Trial card must prioritize active trial ids to avoid live status oscillation."""
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
        / "TrialStatusCard.jsx"
    )
    assert path.exists(), "TrialStatusCard.jsx missing"

    content = path.read_text(encoding="utf-8", errors="ignore")
    assert 'trial.state === "RUNNING" || trial.state === "WAITING"' in content, (
        "TrialStatusCard must prioritize RUNNING/WAITING ids before volatile live_status id."
    )


def test_stat_badge_supports_delta_direction_hover_and_hintbox() -> None:
    """Execute test stat badge supports delta direction hover and hintbox.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    path = (
        settings.PACKAGE_DIR
        / "infrastructure"
        / "hpo"
        / "dashboard"
        / "static"
        / "js"
        / "ui"
        / "StatBadge.jsx"
    )
    assert path.exists(), "StatBadge.jsx missing"

    content = path.read_text(encoding="utf-8", errors="ignore")
    assert "export const StatBadge" in content, "StatBadge component missing"

    for required in ("deltaPct", "direction", "helpText", "valueNode", "animationSeed"):
        assert required in content, f"StatBadge must support '{required}'"

    assert "hover:scale" in content, "StatBadge must have hover scale highlight"
    assert "<PortalTooltip" in content, "StatBadge must show a hintbox tooltip via PortalTooltip"


def test_store_current_trial_selector_uses_active_trial_states() -> None:
    """Store selector must prefer active trials to keep current trial id stable."""
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
    assert "activeTrialIds" in content, "Store must compute active trial ids"
    assert 'state === "RUNNING" || state === "WAITING"' in content, (
        "Store currentTrialId must prioritize RUNNING/WAITING states."
    )


def test_kpi_row_uses_equal_width_cards_like_docs() -> None:
    """Execute test kpi row uses equal width cards like docs.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    path = (
        settings.PACKAGE_DIR
        / "infrastructure"
        / "hpo"
        / "dashboard"
        / "static"
        / "js"
        / "layout"
        / "KpiRow.jsx"
    )
    assert path.exists(), "KpiRow.jsx missing"

    content = path.read_text(encoding="utf-8", errors="ignore")
    assert "lg:grid-cols-4" in content, (
        "KpiRow must render the KPI cards with equal widths (docs target)"
    )
    assert "animationSeed" in content, "KpiRow must wire animationSeed into KPI cards"
    assert "data-jackpot-force" in content, (
        "KpiRow must expose deterministic KPI animation target zone"
    )


def test_best_trial_card_uses_current_trial_id_without_local_counter_state() -> None:
    """Best trial id must reflect store payload directly to avoid stale oscillation."""
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
        / "BestTrialCard.jsx"
    )
    assert path.exists(), "BestTrialCard.jsx missing"

    content = path.read_text(encoding="utf-8", errors="ignore")
    assert "Math.trunc(Number(trial.id))" in content, (
        "BestTrialCard must derive display id directly from trial.id."
    )
    assert "setInterval(" not in content, (
        "BestTrialCard should not animate id via interval-based local counter state."
    )


def test_overview_ranking_uses_completed_trials_with_numeric_score() -> None:
    """Ranking table must consume complete trials only to match best-trial semantics."""
    path = (
        settings.PACKAGE_DIR
        / "infrastructure"
        / "hpo"
        / "dashboard"
        / "static"
        / "js"
        / "layout"
        / "OverviewTab.jsx"
    )
    assert path.exists(), "OverviewTab.jsx missing"

    content = path.read_text(encoding="utf-8", errors="ignore")
    assert "const rankingTrials = useMemo(" in content, (
        "OverviewTab must derive explicit rankingTrials source."
    )
    assert 'state === "COMPLETE"' in content, (
        "Ranking source must filter only COMPLETE trials."
    )
    assert "Number.isFinite(Number(trial.value))" in content, (
        "Ranking source must require numeric score/value."
    )
    assert "const bestRankingTrial = useMemo(" in content, (
        "OverviewTab must derive best trial from the same rankingTrials source."
    )
    assert "<BestTrialCard trial={bestRankingTrial}" in content, (
        "BestTrialCard must consume ranking-derived best trial to keep consistency with table."
    )

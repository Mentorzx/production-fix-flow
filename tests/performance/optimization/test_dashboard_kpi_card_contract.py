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


def test_trial_status_card_prioritizes_active_progress_before_live_trial_id() -> None:
    """Trial card must stabilize on active/history progress before raw liveStatus ids."""
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
    active_idx = content.find("activeTrialIds.length > 0")
    completed_idx = content.find("completedTrialsAll > 0")
    live_idx = content.find("liveTrialId != null")
    assert active_idx != -1, "TrialStatusCard must still track RUNNING/WAITING trial ids."
    assert completed_idx != -1, "TrialStatusCard must derive completed trial progress."
    assert live_idx != -1, "TrialStatusCard must still derive a liveTrialId from liveStatus."
    assert active_idx < live_idx, (
        "TrialStatusCard must prioritize active RUNNING/WAITING ids before raw liveStatus ids."
    )
    assert completed_idx < live_idx, (
        "TrialStatusCard must stabilize on completed+1 progress before raw liveStatus ids."
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


def test_store_current_trial_selector_stabilizes_before_live_trial_id() -> None:
    """Store selector must prefer active/history progress before raw liveStatus ids."""
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
    active_idx = content.find("if (activeTrialIds.length > 0)")
    completed_idx = content.find("if (completedTrialsAll > 0) return nextTrialByCompletion;")
    live_idx = content.find("if (liveTrialId != null) return liveTrialId;")
    assert active_idx != -1, "Store must still track RUNNING/WAITING states."
    assert completed_idx != -1, "Store must expose completed+1 trial progress."
    assert live_idx != -1, "Store must still expose liveTrialId as a fallback current trial."
    assert active_idx < live_idx, (
        "Store currentTrialId must prioritize active RUNNING/WAITING rows before liveStatus."
    )
    assert completed_idx < live_idx, (
        "Store currentTrialId must stabilize on completed+1 progress before liveStatus."
    )


def test_store_sse_deduplicates_by_payload_signature_instead_of_timestamps_only() -> None:
    """Store must refresh when payload content changes without relying on coarse timestamps."""
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
    assert "lastAppliedPayloadSignatureRef" in content, (
        "Store must track the last applied SSE payload signature."
    )
    assert "pending.signature" in content, (
        "Store must compare pending SSE payload signatures before skipping updates."
    )
    assert "prev?.updatedAt === pending?.updatedAt" not in content, (
        "Store must not suppress updates using only updatedAt/liveStatus/trial-count heuristics."
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


def test_jackpot_animation_is_scoped_to_explicit_force_regions() -> None:
    """Jackpot scrambling must not touch overview cards or tables outside KPI zones."""
    path = (
        settings.PACKAGE_DIR
        / "infrastructure"
        / "hpo"
        / "dashboard"
        / "static"
        / "js"
        / "ui"
        / "useJackpotAnimation.js"
    )
    assert path.exists(), "useJackpotAnimation.js missing"

    content = path.read_text(encoding="utf-8", errors="ignore")
    assert "FORCE_SCOPE_SELECTOR" in content, (
        "Jackpot animation must define an explicit force scope selector."
    )
    assert "SCOPED_SELECTOR" in content, (
        "Jackpot animation must compose a scoped selector for explicit force regions."
    )
    assert "root.querySelectorAll(SCOPED_SELECTOR)" in content, (
        "Jackpot animation must only target nodes inside explicit force regions."
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

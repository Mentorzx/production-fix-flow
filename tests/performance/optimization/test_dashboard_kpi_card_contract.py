"""Static checks for KPI cards (top row) to match the dashboard docs target.

The dashboard docs are the objective. These contracts protect:
1. Trial card shows epoch progress (current/total).
2. KPI StatBadge supports delta %, direction cue, hover highlight, and hintbox tooltip.
3. KPI row keeps consistent card sizing across the 4 KPIs (avoid an oversized Trial card).
"""

from __future__ import annotations

from pathlib import Path


def test_trial_status_card_includes_epoch_progress_and_total() -> None:
    path = Path(
        "pff/infrastructure/hpo/dashboard/static/js/features/hpo/charts/TrialStatusCard.jsx"
    )
    assert path.exists(), "TrialStatusCard.jsx missing"

    content = path.read_text(encoding="utf-8", errors="ignore")
    assert "Época" in content, "TrialStatusCard must display the 'Época' label"
    assert "current_epoch" in content, "TrialStatusCard must reference current_epoch"
    assert "total_epochs" in content, "TrialStatusCard must reference total_epochs"


def test_stat_badge_supports_delta_direction_hover_and_hintbox() -> None:
    path = Path("pff/infrastructure/hpo/dashboard/static/js/ui/BaseComponents.jsx")
    assert path.exists(), "BaseComponents.jsx missing"

    content = path.read_text(encoding="utf-8", errors="ignore")
    assert "export const StatBadge" in content, "StatBadge component missing"

    for required in ("deltaPct", "direction", "helpText", "valueNode"):
        assert required in content, f"StatBadge must support '{required}'"

    assert "hover:scale" in content, "StatBadge must have hover scale highlight"
    assert "<PortalTooltip" in content, "StatBadge must show a hintbox tooltip via PortalTooltip"


def test_kpi_row_uses_equal_width_cards_like_docs() -> None:
    path = Path("pff/infrastructure/hpo/dashboard/static/js/layout/KpiRow.jsx")
    assert path.exists(), "KpiRow.jsx missing"

    content = path.read_text(encoding="utf-8", errors="ignore")
    assert "lg:grid-cols-4" in content, "KpiRow must render the KPI cards with equal widths (docs target)"

"""Static checks to enforce chart legend + hintbox contracts.

The dashboard design requires:
1. Legends are top-left to keep a stable visual anchor.
2. Legend labels use hintboxes (via renderLegendWithHints).
"""

from __future__ import annotations

from pathlib import Path


def test_dashboard_chart_legends_use_hints_and_top_left() -> None:
    root = Path("pff/infrastructure/hpo/dashboard/static/js/features/hpo/charts")
    assert root.exists(), "Charts source root missing"

    offenders: list[str] = []
    for path in sorted(root.glob("*.jsx")):
        content = path.read_text(encoding="utf-8", errors="ignore")
        if "<Legend" not in content:
            continue

        if "renderLegendWithHints" not in content:
            offenders.append(f"{path}: missing renderLegendWithHints")
        if 'verticalAlign="top"' not in content:
            offenders.append(f'{path}: legend verticalAlign must be "top"')
        if 'align="left"' not in content:
            offenders.append(f'{path}: legend align must be "left"')

    assert not offenders, "Legend contract violations:\\n- " + "\\n- ".join(offenders)


def test_trial_learning_metrics_card_includes_val_loss_series() -> None:
    path = Path(
        "pff/infrastructure/hpo/dashboard/static/js/features/hpo/charts/TrialLearningMetricsCard.jsx"
    )
    assert path.exists(), "TrialLearningMetricsCard.jsx missing"

    content = path.read_text(encoding="utf-8", errors="ignore")
    assert 'dataKey="val_loss"' in content, "TrialLearningMetricsCard must plot val_loss"
    assert 'name="VAL LOSS"' in content, "TrialLearningMetricsCard must label val_loss series"


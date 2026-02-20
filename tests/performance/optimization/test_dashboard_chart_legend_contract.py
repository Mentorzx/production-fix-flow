"""Static checks to enforce chart legend + hintbox contracts.

The dashboard design requires:
1. Legends are top-right to keep a stable visual anchor.
2. Legend labels use hintboxes (via renderWithHints).
"""

from __future__ import annotations

from pff.shared.core.config import settings

_CHARTS_ROOT = (
    settings.PACKAGE_DIR
    / "infrastructure"
    / "hpo"
    / "dashboard"
    / "static"
    / "js"
    / "features"
    / "hpo"
    / "charts"
)


def test_dashboard_chart_legends_use_hints_and_top_left() -> None:
    """Execute test dashboard chart legends use hints and top left.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    assert _CHARTS_ROOT.exists(), "Charts source root missing"

    offenders: list[str] = []
    for path in sorted(_CHARTS_ROOT.glob("*.jsx")):
        content = path.read_text(encoding="utf-8", errors="ignore")
        if "<Legend" not in content:
            continue

        if "renderWithHints" not in content:
            offenders.append(f"{path}: missing renderWithHints")
        if 'verticalAlign="top"' not in content:
            offenders.append(f'{path}: legend verticalAlign must be "top"')
        if 'align="right"' not in content:
            offenders.append(f'{path}: legend align must be "right"')

    assert not offenders, "Legend contract violations:\\n- " + "\\n- ".join(offenders)


def test_trial_learning_metrics_card_includes_val_loss_series() -> None:
    """Execute test trial learning metrics card includes val loss series.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    path = _CHARTS_ROOT / "TrialLearningMetricsCard.jsx"
    assert path.exists(), "TrialLearningMetricsCard.jsx missing"

    content = path.read_text(encoding="utf-8", errors="ignore")
    assert 'dataKey="val_loss"' in content, "TrialLearningMetricsCard must plot val_loss"
    assert 'name="VAL LOSS"' in content, "TrialLearningMetricsCard must label val_loss series"


def test_dashboard_chart_legends_are_interactive() -> None:
    """Execute test dashboard chart legends are interactive.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    assert _CHARTS_ROOT.exists(), "Charts source root missing"

    offenders: list[str] = []
    for path in sorted(_CHARTS_ROOT.glob("*.jsx")):
        content = path.read_text(encoding="utf-8", errors="ignore")
        if "<Legend" not in content:
            continue
        if "InteractiveLegend" not in content:
            offenders.append(f"{path}: missing InteractiveLegend content")
        if "useLegendVisibility" not in content:
            offenders.append(f"{path}: missing useLegendVisibility hook")
        if "hide={!isSeriesVisible(" not in content:
            offenders.append(f"{path}: missing hide toggle wiring")

    assert not offenders, "Interactive legend contract violations:\\n- " + "\\n- ".join(offenders)

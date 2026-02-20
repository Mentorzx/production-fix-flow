"""Static checks to prevent BestTrialCard badge alignment regressions."""

from __future__ import annotations

import re

from pff.shared.core.config import settings


def test_best_trial_badge_centering_is_not_overridden_by_breath_animation() -> None:
    """The breathing animation must not include translateX(-50%) to avoid off-centering on scale."""
    css_path = (
        settings.PACKAGE_DIR
        / "infrastructure"
        / "hpo"
        / "dashboard"
        / "static"
        / "css"
        / "micro-interactions.css"
    )
    assert css_path.exists(), "micro-interactions.css missing"
    content = css_path.read_text(encoding="utf-8", errors="ignore")

    m = re.search(r"@keyframes\s+pff-breath\s*\{(?P<body>[\s\S]*?)\n\}", content)
    assert m, "pff-breath keyframes missing"

    keyframes = m.group("body")
    assert "translateX(" not in keyframes, "pff-breath keyframes must not translate"


def test_best_trial_badge_translate_is_applied_outside_breath_element() -> None:
    """Ensure the centering translate is not on the same element that receives pff-breath."""
    jsx_path = (
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
    assert jsx_path.exists(), "BestTrialCard.jsx missing"
    content = jsx_path.read_text(encoding="utf-8", errors="ignore")

    # Centering wrapper must exist.
    assert "left-1/2 -translate-x-1/2" in content, "BestTrialCard badge wrapper must be centered"

    # The wrapper line should not also carry the animation class.
    wrapper_lines = [
        line for line in content.splitlines() if "left-1/2" in line and "-translate-x-1/2" in line
    ]
    assert wrapper_lines, "BestTrialCard centered wrapper line missing"
    assert all("pff-breath" not in line for line in wrapper_lines), (
        "pff-breath must be applied to an inner element (scale only), not the centered wrapper"
    )

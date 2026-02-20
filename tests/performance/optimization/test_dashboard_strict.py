"""Strict infrastructure tests for HPO Dashboard.

Enforces:
1. No CDN usage (Fail/Warn).
2. No Babel in browser (Fail/Warn).
3. Valid JSX syntax (Fail).
"""

import re

import pytest

from pff import settings

DASHBOARD_HTML = (
    settings.PACKAGE_DIR / "infrastructure" / "hpo" / "dashboard" / "static" / "index.html"
)


def test_no_tailwind_play_cdn():
    """Fail if Tailwind Play CDN is used."""
    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    assert "cdn.tailwindcss.com" not in html, (
        "Tailwind Play CDN is for dev only. Migrate to PostCSS build."
    )


def test_no_inbrowser_babel():
    """Fail if Babel Standalone is used."""
    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    forbidden = ["@babel/standalone", "babel-standalone", 'type="text/babel"']
    assert all(x not in html for x in forbidden), (
        "Babel Standalone is for prototyping. Precompile JSX."
    )


def test_no_adjacent_jsx_in_inline_svg():
    """Lint check: ensure inline SVG JSX has wrapper elements (Fragment) if multiple children."""
    html = DASHBOARD_HTML.read_text(encoding="utf-8")

    # Regex to find d={...} props that likely contain multiple root elements without fragment
    # Heuristic: d={<tag .../><tag .../>} without surrounding <>...</>
    # This is tricky with regex, but we can look for specific patterns

    # Pattern: d={< (something) /> < (something)
    # If we see two opening brackets '<' inside the curly braces without an initial '<>'

    matches = re.finditer(r"d=\{([^}]+)\}", html)

    for m in matches:
        content = m.group(1)
        if content.strip().startswith("<>"):
            continue

        tags = content.count("/>") + content.count("</")

        # If more than 1 tag and no fragment wrapper -> Failure
        if tags > 1:
            pytest.fail(f"Found adjacent JSX elements without wrapper in d prop: {content}")


def test_recharts_prop_types_dependency():
    """Ensure prop-types is included (Required for Recharts UMD)."""
    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    assert "prop-types" in html, "Recharts UMD requires prop-types to be loaded before it."

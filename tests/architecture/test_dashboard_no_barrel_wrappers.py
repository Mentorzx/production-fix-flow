"""Architecture guardrail: dashboard JS must avoid barrel wrapper re-exports."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_JS_ROOT = (
    REPO_ROOT / "src" / "pff" / "infrastructure" / "hpo" / "dashboard" / "static" / "js"
)


def test_dashboard_js_must_not_use_reexport_wrappers() -> None:
    """Forbid `export ... from` wrappers in dashboard JS modules."""
    violations: list[str] = []

    for path in DASHBOARD_JS_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".js", ".jsx", ".ts", ".tsx"}:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        content = path.read_text(encoding="utf-8")
        if "export {" in content and " from " in content:
            for line in content.splitlines():
                if "export {" in line and " from " in line:
                    violations.append(f"{rel}: {line.strip()}")

    assert not violations, (
        "Dashboard JS must import directly from concrete modules (no barrel wrappers):\n"
        + "\n".join(sorted(violations))
    )

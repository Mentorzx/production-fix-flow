"""Hardware scaling contract guardrails (AGENTS.md 8)."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path("src/pff")


def test_no_direct_os_cpu_count_usage() -> None:
    """Prevent direct os.cpu_count() usage in production code."""
    violations: list[tuple[Path, int, str]] = []

    for path in sorted(_ROOT.rglob("*.py")):
        content = path.read_text(encoding="utf-8")
        for idx, line in enumerate(content.splitlines(), start=1):
            if "# noqa" in line.lower():
                continue
            if "os.cpu_count(" in line:
                violations.append((path, idx, line.strip()))

    assert not violations, (
        "Direct os.cpu_count() usage found. Use HardwareManager/probe abstractions:\n"
        + "\n".join(f"{path}:{line_no} {line}" for path, line_no, line in violations)
    )

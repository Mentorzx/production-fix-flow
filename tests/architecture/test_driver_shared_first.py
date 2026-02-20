"""Driver layer shared-first guardrails (AGENTS.md 4.10)."""

from __future__ import annotations

import re
from pathlib import Path

_RAW_HTTP_PATTERNS = (
    re.compile(r"""\burllib\.request\b"""),
    re.compile(r"""\brequests\."""),
    re.compile(r"""\bhttpx\."""),
)
_RAW_THREADING_PATTERNS = (
    re.compile(r"""\bimport\s+threading\b"""),
    re.compile(r"""\bthreading\."""),
)

_DRIVERS_ROOT = Path("src/pff/drivers")


def _iter_driver_files() -> list[Path]:
    """Execute iter driver files.



    Returns:

        Return value produced by the callable.

    """

    if not _DRIVERS_ROOT.exists():
        return []
    return sorted(_DRIVERS_ROOT.rglob("*.py"))


def test_no_raw_http_clients_in_drivers() -> None:
    """Ensure drivers use shared HTTP client abstractions."""
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_driver_files():
        content = path.read_text(encoding="utf-8")
        for idx, line in enumerate(content.splitlines(), start=1):
            if "# noqa" in line.lower():
                continue
            if "pff.shared.clients.http_client" in line:
                continue
            if any(pattern.search(line) for pattern in _RAW_HTTP_PATTERNS):
                violations.append((path, idx, line.strip()))

    assert not violations, (
        "Raw HTTP client usage found in drivers. Route through pff.shared.clients:\n"
        + "\n".join(f"{path}:{line_no} {line}" for path, line_no, line in violations)
    )


def test_no_raw_threading_in_drivers() -> None:
    """Ensure driver-layer concurrency abstractions are used."""
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_driver_files():
        content = path.read_text(encoding="utf-8")
        for idx, line in enumerate(content.splitlines(), start=1):
            if "# noqa" in line.lower():
                continue
            if any(pattern.search(line) for pattern in _RAW_THREADING_PATTERNS):
                violations.append((path, idx, line.strip()))

    assert not violations, (
        "Raw threading usage found in drivers. Route through infrastructure/shared abstractions:\n"
        + "\n".join(f"{path}:{line_no} {line}" for path, line_no, line in violations)
    )

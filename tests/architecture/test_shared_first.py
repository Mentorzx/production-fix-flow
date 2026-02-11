"""Shared-first compliance test (AGENTS.md 4.10)."""

from __future__ import annotations

import re
from pathlib import Path

_RAW_OPEN_PATTERN = re.compile(r"""\bopen\s*\(""")
_RAW_REQUESTS_PATTERN = re.compile(r"""(?:^|\s)requests\.""")
_RAW_THREADING_PATTERN = re.compile(r"""(?:^|\s)threading\.""")
_RAW_MULTIPROCESSING_PATTERN = re.compile(r"""(?:^|\s)multiprocessing\.""")

_LAYER_DIRS = (
    Path("src/pff/domain"),
    Path("src/pff/application"),
)

_ALLOWLIST_FILES = {
    Path("src/pff/application/ports/persistence.py"),
}


def test_no_raw_open_in_domain_application() -> None:
    """Ensure no raw open() calls in domain/application layers.

    Per AGENTS.md 4.10: Filesystem I/O must route through FileManager.
    """
    violations: list[tuple[Path, int, str]] = []

    for layer_dir in _LAYER_DIRS:
        if not layer_dir.exists():
            continue
        for path in layer_dir.rglob("*.py"):
            rel_path = path
            if path.is_absolute():
                rel_path = path.relative_to(Path.cwd())

            if rel_path in _ALLOWLIST_FILES:
                continue

            content = path.read_text(encoding="utf-8")
            for idx, line in enumerate(content.splitlines(), start=1):
                if "FileManager" in line or "file_manager" in line:
                    continue
                if "# noqa" in line.lower():
                    continue
                if _RAW_OPEN_PATTERN.search(line):
                    if "open(" in line and ("urlopen" in line or "zipfile" in line):
                        continue
                    violations.append((rel_path, idx, line.strip()))

    assert not violations, (
        "Raw open() calls found in domain/application. "
        "Use FileManager instead:\n"
        + "\n".join(f"{path}:{line_no} {line}" for path, line_no, line in violations)
    )


def test_no_raw_requests_in_domain_application() -> None:
    """Ensure HTTP goes through pff.shared.clients.

    Per AGENTS.md 4.10: HTTP must route through shared http_client.
    """
    violations: list[tuple[Path, int, str]] = []

    for layer_dir in _LAYER_DIRS:
        if not layer_dir.exists():
            continue
        for path in layer_dir.rglob("*.py"):
            rel_path = path
            if path.is_absolute():
                rel_path = path.relative_to(Path.cwd())

            if rel_path in _ALLOWLIST_FILES:
                continue

            content = path.read_text(encoding="utf-8")
            for idx, line in enumerate(content.splitlines(), start=1):
                if "# noqa" in line.lower():
                    continue
                if _RAW_REQUESTS_PATTERN.search(line):
                    violations.append((rel_path, idx, line.strip()))

    assert not violations, (
        "Raw requests.* calls found in domain/application. "
        "Use pff.shared.clients.http_client instead:\n"
        + "\n".join(f"{path}:{line_no} {line}" for path, line_no, line in violations)
    )


def test_no_raw_threading_in_domain_application() -> None:
    """Ensure concurrency goes through pff.shared.acceleration.

    Per AGENTS.md 4.10: Concurrency must route through ConcurrencyManager.
    """
    violations: list[tuple[Path, int, str]] = []

    for layer_dir in _LAYER_DIRS:
        if not layer_dir.exists():
            continue
        for path in layer_dir.rglob("*.py"):
            rel_path = path
            if path.is_absolute():
                rel_path = path.relative_to(Path.cwd())

            if rel_path in _ALLOWLIST_FILES:
                continue

            content = path.read_text(encoding="utf-8")
            for idx, line in enumerate(content.splitlines(), start=1):
                if "# noqa" in line.lower():
                    continue
                if _RAW_THREADING_PATTERN.search(line):
                    violations.append((rel_path, idx, line.strip()))
                if _RAW_MULTIPROCESSING_PATTERN.search(line):
                    violations.append((rel_path, idx, line.strip()))

    assert not violations, (
        "Raw threading/multiprocessing calls found in domain/application. "
        "Use pff.shared.acceleration.ConcurrencyManager instead:\n"
        + "\n".join(f"{path}:{line_no} {line}" for path, line_no, line in violations)
    )

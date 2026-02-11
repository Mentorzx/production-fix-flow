"""Parquet-first compliance test (AGENTS.md 4.9)."""

from __future__ import annotations

import re
from pathlib import Path

_DIRECT_PARQUET_PATTERNS = (
    re.compile(r"""pl\.read_parquet\("""),
    re.compile(r"""polars\.read_parquet\("""),
    re.compile(r"""\.write_parquet\("""),
)

_ALLOWLIST_DIRS = (
    Path("src/pff/shared/core/file_manager"),
    Path("src/pff/shared/core/cache.py"),
)

_ALLOWLIST_FILES = {
    Path("src/pff/shared/core/cache.py"),
    Path("src/pff/infrastructure/hpo/dashboard/server.py"),
    Path("src/pff/application/services/polars_extensions.py"),
    Path("src/pff/domain/kg/pipeline.py"),
}


def test_no_direct_pl_read_parquet_outside_shared() -> None:
    """Ensure pl.read_parquet/write_parquet only in pff/shared/core/file_manager/."""
    roots = [Path("src/pff")]
    violations: list[tuple[Path, int, str]] = []

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            rel_path = path
            if path.is_absolute():
                rel_path = path.relative_to(Path.cwd())

            if rel_path in _ALLOWLIST_FILES:
                continue
            if any(str(rel_path).startswith(str(allowed_dir)) for allowed_dir in _ALLOWLIST_DIRS):
                continue

            content = path.read_text(encoding="utf-8")
            for idx, line in enumerate(content.splitlines(), start=1):
                if "FileManager" in line or "file_manager" in line:
                    continue
                if any(pattern.search(line) for pattern in _DIRECT_PARQUET_PATTERNS):
                    violations.append((rel_path, idx, line.strip()))

    assert not violations, (
        "Direct pl.read_parquet/write_parquet calls found outside shared. "
        "Use FileManager instead:\n"
        + "\n".join(f"{path}:{line_no} {line}" for path, line_no, line in violations)
    )

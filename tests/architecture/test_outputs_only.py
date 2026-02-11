"""Outputs-only compliance test (AGENTS.md)."""

from __future__ import annotations

import re
from pathlib import Path

_WRITE_PATTERNS = (
    re.compile(r"""\bopen\([^)]*['"]w"""),
    re.compile(r"""\.open\(['"]w"""),
    re.compile(r"""\.write_text\("""),
    re.compile(r"""\.write_bytes\("""),
)

_ALLOWLIST = {
    Path("src/pff/shared/core/file_manager.py"),
    Path("src/pff/shared/core/logging/reorderer.py"),
    Path("src/pff/infrastructure/cleanup/commands/filesystem.py"),
    Path("src/pff/infrastructure/cleanup/file_ops.py"),
    Path("src/pff/infrastructure/hpo/callbacks_internal/visualizers.py"),
    Path("scripts/convert_zip_to_parquet_silver.py"),
    Path("scripts/benchmark_dslfm_optimizations.py"),
    Path("scripts/update_goldens.py"),
    Path("scripts/update_golden_help.py"),
}

# Directories where file writes are allowed (I/O utilities)
_ALLOWLIST_DIRS = (
    Path("src/pff/shared/core/file_manager"),
    Path("src/pff/shared/core/logging"),
)


def test_no_direct_writes_outside_utils() -> None:
    """Ensure direct file writes are limited to utils/cleanup internals."""
    roots = [Path("src/pff"), Path("scripts")]
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
            if rel_path in _ALLOWLIST:
                continue
            # Check if path is under any allowlisted directory
            if any(allowed_dir in rel_path.parents for allowed_dir in _ALLOWLIST_DIRS):
                continue
            content = path.read_text(encoding="utf-8")
            for idx, line in enumerate(content.splitlines(), start=1):
                if "FileManager." in line or "file_manager" in line:
                    continue
                if any(pattern.search(line) for pattern in _WRITE_PATTERNS):
                    violations.append((rel_path, idx, line.strip()))

    assert not violations, "\n".join(
        f"{path}:{line_no} {line}" for path, line_no, line in violations
    )

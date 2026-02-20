"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/architecture/test_no_module_singletons.py

"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOTS = (
    REPO_ROOT / "src" / "pff" / "domain",
    REPO_ROOT / "src" / "pff" / "application",
)
FORBIDDEN_CTOR_NAMES = {"FileManager", "CacheManager"}


def _iter_python_files() -> list[Path]:
    """Execute iter python files.



    Returns:

        Return value produced by the callable.

    """

    files: list[Path] = []
    for root in TARGET_ROOTS:
        if root.exists():
            files.extend(p for p in root.rglob("*.py") if p.is_file())
    return files


def _call_name(node: ast.Call) -> str | None:
    """Execute call name.



    Args:

        node: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def test_no_module_level_file_or_cache_manager_singletons() -> None:
    """Prevent hidden global state via module-level manager singletons."""
    violations: list[str] = []
    for path in _iter_python_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            value = None
            if isinstance(node, ast.Assign):
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                value = node.value
            if not isinstance(value, ast.Call):
                continue
            ctor = _call_name(value)
            if ctor not in FORBIDDEN_CTOR_NAMES:
                continue
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name) and t.id]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            else:
                targets = []
            target_str = ", ".join(targets) if targets else "<unknown>"
            violations.append(f"{rel}:{node.lineno} -> {target_str} = {ctor}(...)")

    assert not violations, (
        "Module-level FileManager/CacheManager singletons are forbidden in "
        "domain/application (hidden global state):\n" + "\n".join(sorted(violations))
    )

"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/architecture/test_no_stdlib_json.py

"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOTS = (REPO_ROOT / "src" / "pff", REPO_ROOT / "scripts")


def _iter_python_files() -> list[Path]:
    """Execute iter python files.



    Returns:

        Return value produced by the callable.

    """

    files: list[Path] = []
    for root in ROOTS:
        if root.exists():
            for path in sorted(root.rglob("*.py")):
                if "node_modules" in path.parts:
                    continue
                files.append(path)
    return files


def test_no_stdlib_json_imports() -> None:
    """Block stdlib json usage to enforce orjson/msgspec for speed and consistency."""
    violations: list[str] = []
    for path in _iter_python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, SyntaxError):
            continue
        rel = path.relative_to(REPO_ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "json" or alias.name.startswith("json."):
                        violations.append(f"{rel}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
                if node.module == "json" or node.module.startswith("json."):
                    violations.append(f"{rel}:{node.lineno} from {node.module} import ...")
    assert not violations, (
        "Stdlib json is blocked. Use orjson or msgspec for deterministic/efficient JSON handling.\n"
        + "\n".join(violations)
    )

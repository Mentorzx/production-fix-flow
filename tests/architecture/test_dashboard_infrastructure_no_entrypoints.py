"""Architecture guardrail: dashboard infrastructure modules must not be entrypoints."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_INFRA_ROOT = REPO_ROOT / "src" / "pff" / "infrastructure" / "hpo" / "dashboard"


def _is_name_main_comparison(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare):
        return False
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
        return False
    if len(node.comparators) != 1:
        return False

    left = node.left
    right = node.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    )


def test_dashboard_infrastructure_modules_must_not_have_main_entrypoints() -> None:
    """Prevent CLI entrypoints from leaking into infrastructure dashboard modules."""
    violations: list[str] = []

    for path in DASHBOARD_INFRA_ROOT.rglob("*.py"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        has_top_level_main_fn = any(
            isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body
        )
        has_main_guard = any(
            isinstance(node, ast.If) and _is_name_main_comparison(node.test) for node in tree.body
        )

        if has_top_level_main_fn or has_main_guard:
            violations.append(rel)

    assert not violations, (
        "Dashboard infrastructure must expose functions only. "
        "Move entrypoints to drivers:\n" + "\n".join(sorted(violations))
    )


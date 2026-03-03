"""Architecture guardrail to enforce explicit LineService API injection in drivers."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "pff"


class _LineServiceCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls_without_api_client: list[tuple[int, int]] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        is_line_service_call = isinstance(func, ast.Name) and func.id == "LineService"
        if is_line_service_call:
            has_api_kw = any(kw.arg == "api_client" for kw in node.keywords if kw.arg)
            if not has_api_kw:
                self.calls_without_api_client.append((node.lineno, node.col_offset))
        self.generic_visit(node)


def test_drivers_must_inject_api_client_into_line_service() -> None:
    """All LineService constructor calls in src drivers must pass api_client explicitly."""
    violations: list[str] = []

    for path in (SRC_ROOT / "drivers").rglob("*.py"):
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        visitor = _LineServiceCallVisitor()
        visitor.visit(tree)

        if visitor.calls_without_api_client:
            rel = path.relative_to(REPO_ROOT).as_posix()
            for line, col in visitor.calls_without_api_client:
                violations.append(f"{rel}:{line}:{col}")

    assert not violations, (
        "LineService in drivers must receive api_client explicit injection:\n"
        + "\n".join(sorted(violations))
    )

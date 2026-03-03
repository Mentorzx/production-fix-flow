"""Architecture guardrail for direct manager instantiation in pure layers."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOTS = (
    REPO_ROOT / "src" / "pff" / "domain",
    REPO_ROOT / "src" / "pff" / "application",
)
FORBIDDEN_CTORS = {"FileManager", "CacheManager"}

# Baseline allowlist for legacy modules. New files must not be added here.
ALLOWED_FILES = {
    "src/pff/application/audit_use_case.py",
    "src/pff/application/learn_use_case.py",
    "src/pff/application/services/business_service/core.py",
    "src/pff/application/services/business_service/model_integration.py",
    "src/pff/application/services/business_service/rule_engine.py",
    "src/pff/application/services/business_service/shared/rule_builder.py",
    "src/pff/application/services/intelligent_preprocessor.py",
    "src/pff/application/services/line_service/base.py",
    "src/pff/application/services/line_service/config.py",
    "src/pff/application/services/polars_extensions.py",
    "src/pff/application/services/sequence_service.py",
    "src/pff/domain/audit/bench.py",
    "src/pff/domain/audit/evt.py",
    "src/pff/domain/audit/manifest.py",
    "src/pff/domain/audit/profile.py",
    "src/pff/domain/audit/report.py",
    "src/pff/domain/audit/schema.py",
    "src/pff/domain/kg/builder.py",
    "src/pff/domain/kg/config.py",
    "src/pff/domain/kg/data_optimizer.py",
    "src/pff/domain/kg/pipeline.py",
    "src/pff/domain/kg/preprocess.py",
    "src/pff/domain/kg/preprocessing/config.py",
    "src/pff/domain/kg/preprocessing/pipeline.py",
    "src/pff/domain/learning/dslfm/checkpoint_manager.py",
    "src/pff/domain/learning/dslfm/kgc_manager.py",
    "src/pff/domain/learning/dslfm/metrics_reporter.py",
    "src/pff/domain/learning/dslfm/neg_sampling.py",
    "src/pff/domain/learning/ml/base_trainer.py",
    "src/pff/domain/learning/ml/model_factory.py",
}


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


def _has_forbidden_ctor(tree: ast.AST) -> bool:
    """Execute has forbidden ctor.



    Args:

        tree: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_CTORS:
            return True
        if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_CTORS:
            return True
    return False


def _build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    """Create a child->parent map for AST traversal checks."""
    parent_by_node: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_by_node[child] = parent
    return parent_by_node


def _is_ctor_fallback_call(node: ast.Call, parent_by_node: dict[ast.AST, ast.AST]) -> bool:
    """Return True when a constructor call is used as fallback in `x or Ctor()`."""
    parent = parent_by_node.get(node)
    return isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.Or)


def test_no_new_filemanager_or_cachemanager_instantiation_files() -> None:
    """Prevent new direct manager instantiation in domain/application layers."""
    current_files: set[str] = set()
    for path in _iter_python_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        if _has_forbidden_ctor(tree):
            current_files.add(rel)

    new_violations = sorted(current_files - ALLOWED_FILES)
    assert not new_violations, (
        "New FileManager/CacheManager instantiation files detected in domain/application. "
        "Inject via ports/adapters instead:\n" + "\n".join(new_violations)
    )


def test_application_filemanager_instantiation_must_be_di_fallback_only() -> None:
    """In application layer, FileManager() calls must appear only as DI fallback."""
    violations: list[str] = []
    app_root = REPO_ROOT / "src" / "pff" / "application"
    for path in app_root.rglob("*.py"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        parent_by_node = _build_parent_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Name) and func.id == "FileManager"):
                continue
            if _is_ctor_fallback_call(node, parent_by_node):
                continue
            violations.append(rel)
            break

    assert not violations, (
        "FileManager() must be used only as DI fallback (`x or FileManager()`) in application:\n"
        + "\n".join(sorted(violations))
    )


def test_application_cachemanager_instantiation_must_be_di_fallback_only() -> None:
    """In application layer, CacheManager() calls must appear only as DI fallback."""
    violations: list[str] = []
    app_root = REPO_ROOT / "src" / "pff" / "application"
    for path in app_root.rglob("*.py"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        parent_by_node = _build_parent_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Name) and func.id == "CacheManager"):
                continue
            if _is_ctor_fallback_call(node, parent_by_node):
                continue
            violations.append(rel)
            break

    assert not violations, (
        "CacheManager() must be used only as DI fallback (`x or CacheManager()`) in application:\n"
        + "\n".join(sorted(violations))
    )

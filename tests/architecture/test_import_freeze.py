from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOTS = [REPO_ROOT / "pff", REPO_ROOT / "scripts"]
INTERNAL_PREFIXES = ("pff", "scripts")
LEGACY_PREFIXES = ("pff.validators", "pff.db")
DRIVER_PREFIXES = ("pff.drivers", "pff.__main__", "scripts", "pff")
FORBIDDEN_DOMAIN_IMPORTS = ("pff.infrastructure", "pff.drivers")
FORBIDDEN_APPLICATION_IMPORTS = ("pff.infrastructure", "pff.drivers")
FORBIDDEN_INFRA_IMPORTS = ("pff.drivers",)

_ALLOWED_LAYER_VIOLATIONS = {
    "pff/application/audit_use_case.py: pff.application.audit_use_case -> pff.infrastructure.persistence.file_storage",
    "pff/application/learn_use_case.py: pff.application.learn_use_case -> pff.infrastructure.persistence.db.repositories.kg_splits",
    "pff/application/services/business_service/core.py: pff.application.services.business_service.core -> pff.infrastructure.persistence.audit.storage",
    "pff/application/services/business_service/core.py: pff.application.services.business_service.core -> pff.infrastructure.persistence.db.repositories",
    "pff/domain/kg/kg/builder.py: pff.domain.kg.builder -> pff.infrastructure.persistence.db.repositories",
    "pff/domain/kg/kg/data_loader.py: pff.domain.kg.data_loader -> pff.infrastructure.persistence.db.repositories",
    "pff/domain/kg/kg/factory.py: pff.domain.kg.factory -> pff.infrastructure.persistence.db.repositories",
    "pff/domain/kg/kg/pipeline.py: pff.domain.kg.pipeline -> pff.infrastructure.persistence.db.repositories",
    "pff/domain/kg/kg/preprocess.py: pff.domain.kg.preprocess -> pff.infrastructure.persistence.db.repositories.kg_mappings",
    "pff/domain/kg/kg/preprocess.py: pff.domain.kg.preprocess -> pff.infrastructure.persistence.db.repositories",
}


def _module_name(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT)
    if rel.name == "__init__.py":
        rel = rel.parent
    return ".".join(rel.with_suffix("").parts)


def _resolve_relative_base(
    module: str | None,
    level: int,
    current: str,
    *,
    is_package: bool,
) -> str | None:
    parts = current.split(".")
    if not is_package and parts:
        parts = parts[:-1]
    if level <= 0:
        return None
    drop = level - 1
    if drop > len(parts):
        return None
    base = parts[: len(parts) - drop]
    if module:
        base.extend(module.split("."))
    if not base:
        return None
    return ".".join(base)


def _iter_python_files() -> Iterable[Path]:
    for package_root in PACKAGE_ROOTS:
        if not package_root.exists():
            continue
        yield from package_root.rglob("*.py")


def _iter_imports(path: Path, module: str) -> Iterator[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return

    is_package = path.name == "__init__.py"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module is None and node.level == 0:
                continue
            if node.level:
                base = _resolve_relative_base(
                    node.module,
                    node.level,
                    module,
                    is_package=is_package,
                )
            else:
                base = node.module
            if not base:
                continue
            if node.module is None:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    yield f"{base}.{alias.name}"
            else:
                yield base


def _iter_internal_imports() -> Iterator[tuple[Path, str, str]]:
    for path in _iter_python_files():
        source_module = _module_name(path)
        for imported in _iter_imports(path, source_module):
            if imported.startswith(INTERNAL_PREFIXES):
                yield path, source_module, imported


def test_no_legacy_namespaces() -> None:
    """Fail if any import still uses legacy namespaces after the cutover."""
    violations = []
    for path, source_module, imported in _iter_internal_imports():
        if imported.startswith(LEGACY_PREFIXES):
            rel_path = path.relative_to(REPO_ROOT)
            violations.append(f"{rel_path}: {source_module} -> {imported}")

    assert not violations, "Legacy namespaces are still imported:\n" + "\n".join(sorted(violations))


def test_drivers_only_imported_by_drivers() -> None:
    """Ensure drivers are only imported by other drivers/entrypoints."""
    violations = []
    for path, source_module, imported in _iter_internal_imports():
        if imported.startswith("pff.drivers") and not source_module.startswith(DRIVER_PREFIXES):
            rel_path = path.relative_to(REPO_ROOT)
            violations.append(f"{rel_path}: {source_module} -> {imported}")

    assert not violations, "Non-driver modules are importing drivers:\n" + "\n".join(
        sorted(violations)
    )


def test_layer_dependencies_freeze() -> None:
    """Freeze existing layer violations; forbid new ones."""
    violations = []
    for path, source_module, imported in _iter_internal_imports():
        if source_module.startswith("pff.domain") and imported.startswith(FORBIDDEN_DOMAIN_IMPORTS):
            rel_path = path.relative_to(REPO_ROOT)
            violations.append(f"{rel_path}: {source_module} -> {imported}")
        if source_module.startswith("pff.application") and imported.startswith(
            FORBIDDEN_APPLICATION_IMPORTS
        ):
            rel_path = path.relative_to(REPO_ROOT)
            violations.append(f"{rel_path}: {source_module} -> {imported}")
        if source_module.startswith("pff.infrastructure") and imported.startswith(
            FORBIDDEN_INFRA_IMPORTS
        ):
            rel_path = path.relative_to(REPO_ROOT)
            violations.append(f"{rel_path}: {source_module} -> {imported}")

    unexpected = sorted(set(violations) - _ALLOWED_LAYER_VIOLATIONS)
    assert not unexpected, (
        "New layer violations detected (update baseline only for intentional refactors):\n"
        + "\n".join(unexpected)
    )

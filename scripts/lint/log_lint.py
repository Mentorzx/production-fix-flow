#!/usr/bin/env python3
"""Log-lint: validates logging compliance per AGENTS.md §4.5.

Rules enforced:
  1. logger.info / logger.success: content MUST be PT-BR.
  2. logger.warning / logger.error / logger.debug: content MUST be EN.
  3. print() is forbidden in src/pff/ (except drivers with click.echo).
  4. Prefer f-strings over % or .format() in log calls.

Usage:
    poetry run python scripts/lint/log_lint.py --check src/pff/
    poetry run python scripts/lint/log_lint.py --fix src/pff/
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PT_BR_INDICATORS = {
    "Iniciando",
    "Concluido",
    "concluido",
    "Processando",
    "Carregando",
    "Finalizado",
    "Resultado",
    "Treinamento",
    "treinamento",
    "Epoca",
    "epoca",
    "Modelo",
    "modelo",
    "Configuracao",
    "configuracao",
    "Executando",
    "executando",
    "Salvando",
    "salvando",
    "Gerando",
    "gerando",
    "Calculando",
    "calculando",
    "Otimizacao",
    "otimizacao",
    "Validacao",
    "validacao",
    "Preparando",
    "preparando",
    "Aplicando",
    "aplicando",
    "Atualizando",
    "atualizando",
    "Removendo",
    "removendo",
    "Limpando",
    "limpando",
    "Criando",
    "criando",
    "Verificando",
    "verificando",
    "Importando",
    "importando",
    "Exportando",
    "exportando",
    "Conectando",
    "conectando",
    "Desconectando",
    "Encerrando",
    "Disponivel",
    "disponivel",
    "Indisponivel",
    "Tempo",
    "tempo",
    "Tamanho",
    "tamanho",
    "Total",
    "Parcial",
    "Completo",
    "Erro",
    "Sucesso",
    "Falha",
    "Invalidando",
    "invalidando",
    "etapa",
    "cacheados",
    "cacheado",
    "total",
}

EN_INDICATORS = {
    "Starting",
    "Completed",
    "Processing",
    "Loading",
    "Finished",
    "Result",
    "Training",
    "training",
    "Epoch",
    "epoch",
    "Model",
    "model",
    "Configuration",
    "configuration",
    "Executing",
    "executing",
    "Saving",
    "saving",
    "Generating",
    "generating",
    "Calculating",
    "calculating",
    "Optimization",
    "Validation",
    "validation",
    "Preparing",
    "preparing",
    "Applying",
    "applying",
    "Updating",
    "updating",
    "Removing",
    "removing",
    "Cleaning",
    "cleaning",
    "Creating",
    "creating",
    "Verifying",
    "verifying",
    "Importing",
    "importing",
    "Exporting",
    "exporting",
    "Connecting",
    "connecting",
    "Disconnecting",
    "Shutting",
    "Available",
    "available",
    "Unavailable",
    "Failed",
    "Error",
    "Warning",
    "Unable",
    "Cannot",
    "Could not",
    "Invalid",
    "Missing",
    "Unexpected",
    "Timeout",
    "Retrying",
    "Falling back",
    "falling back",
    "Skipping",
    "skipping",
    "Ignoring",
    "ignoring",
    "Total",
    "total",
}

LOG_CALL_RE = re.compile(
    r"""logger\.(info|success|warning|error|debug)\s*\(\s*(?:f?["'])(.*?)["']""",
    re.DOTALL,
)

PRINT_RE = re.compile(r"""(?<!\w)print\s*\(""")

FORMAT_STYLE_RE = re.compile(
    r"""logger\.\w+\s*\([^)]*(?:%[sd]|\.format\s*\()""",
    re.DOTALL,
)


@dataclass
class Violation:
    """A single log-lint violation."""

    file: str
    line: int
    rule: str
    message: str
    fixable: bool = False


_FSTRING_EXPR_RE = re.compile(r"\{[^}]+\}")


def _detect_language(text: str) -> str | None:
    """Heuristic language detection. Returns 'pt-br', 'en', or None.

    Strips f-string interpolation expressions (e.g. ``{model_name}``) before
    counting keyword indicators so that variable names don't pollute the signal.
    """
    cleaned = _FSTRING_EXPR_RE.sub("", text)
    pt_count = sum(1 for w in PT_BR_INDICATORS if w in cleaned)
    en_count = sum(1 for w in EN_INDICATORS if w in cleaned)
    if pt_count > en_count and pt_count >= 1:
        return "pt-br"
    if en_count > pt_count and en_count >= 1:
        return "en"
    return None


def check_file(filepath: Path, base: Path) -> list[Violation]:
    """Check a single Python file for log-lint violations."""
    violations: list[Violation] = []
    rel = filepath.relative_to(base)
    rel_str = str(rel)

    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return violations

    lines = content.splitlines()
    is_driver = "drivers/" in rel_str
    in_main_guard = False

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()

        if stripped.startswith("if __name__") and "__main__" in stripped:
            in_main_guard = True
        elif (
            in_main_guard and stripped and not stripped.startswith("#") and not line.startswith(" ")
        ):
            in_main_guard = False

        if PRINT_RE.search(line) and not in_main_guard and not is_driver:
            if (
                not stripped.startswith("#")
                and "noqa" not in line
                and ">>>" not in stripped
                and '"""' not in stripped
                and "'''" not in stripped
            ):
                violations.append(
                    Violation(
                        file=rel_str,
                        line=lineno,
                        rule="no-print",
                        message="print() forbidden in src/pff/ (use logger instead)",
                        fixable=True,
                    )
                )

        for match in LOG_CALL_RE.finditer(line):
            level = match.group(1)
            msg_text = match.group(2)

            if level in ("info", "success"):
                lang = _detect_language(msg_text)
                if lang == "en":
                    violations.append(
                        Violation(
                            file=rel_str,
                            line=lineno,
                            rule="lang-ptbr",
                            message=f"logger.{level}() must be PT-BR, detected EN",
                        )
                    )
            elif level in ("warning", "error", "debug"):
                lang = _detect_language(msg_text)
                if lang == "pt-br":
                    violations.append(
                        Violation(
                            file=rel_str,
                            line=lineno,
                            rule="lang-en",
                            message=f"logger.{level}() must be EN, detected PT-BR",
                        )
                    )

        if FORMAT_STYLE_RE.search(line):
            if not stripped.startswith("#"):
                violations.append(
                    Violation(
                        file=rel_str,
                        line=lineno,
                        rule="use-fstring",
                        message="Prefer f-strings over %/format() in log calls",
                    )
                )

    return violations


def fix_prints(filepath: Path) -> int:
    """Replace print() with logger.info() in a file. Returns count of fixes."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0

    fixed = 0
    lines = content.splitlines()
    new_lines = []
    has_logger_import = "from pff.shared" in content and "logger" in content

    for line in lines:
        if PRINT_RE.search(line) and not line.strip().startswith("#"):
            if "if __name__" not in line and "drivers/" not in str(filepath):
                new_line = PRINT_RE.sub("logger.info(", line, count=1)
                new_lines.append(new_line)
                fixed += 1
                continue
        new_lines.append(line)

    if fixed > 0:
        if not has_logger_import:
            new_lines.insert(0, "from pff.shared import logger")
        filepath.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return fixed


def main() -> int:
    """Execute main.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    parser = argparse.ArgumentParser(description="PFF log-lint (AGENTS.md §4.5 compliance)")
    parser.add_argument("path", nargs="?", default="src/pff/", help="Path to check")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", default=True, help="Check only (default)")
    mode.add_argument("--fix", action="store_true", help="Auto-fix safe violations")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.is_absolute():
        target = REPO_ROOT / target

    if not target.exists():
        print(f"Path not found: {target}")
        return 2

    py_files = sorted(target.rglob("*.py")) if target.is_dir() else [target]
    all_violations: list[Violation] = []
    total_fixed = 0

    for py_file in py_files:
        violations = check_file(py_file, REPO_ROOT)
        all_violations.extend(violations)

        if args.fix:
            fixable = [v for v in violations if v.fixable and v.rule == "no-print"]
            if fixable:
                total_fixed += fix_prints(py_file)

    if all_violations:
        print(f"\nLog-lint violations ({len(all_violations)}):")
        print(f"{'File':<55} {'Line':<6} {'Rule':<14} {'Message'}")
        print("-" * 110)
        for v in all_violations[:100]:
            print(f"{v.file:<55} {v.line:<6} {v.rule:<14} {v.message}")
        if len(all_violations) > 100:
            print(f"  ... and {len(all_violations) - 100} more")

    if args.fix and total_fixed:
        print(f"\nAuto-fixed {total_fixed} print() -> logger.info() replacement(s)")

    unfixed = [v for v in all_violations if not v.fixable or not args.fix]
    if unfixed:
        print(f"\n{len(unfixed)} violation(s) require manual fix")
        return 1

    if not all_violations:
        print("Log-lint: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Logging language contract tests (AGENTS.md)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LogMatch:
    path: Path
    line_no: int
    level: str
    message: str


_LOG_PATTERN = re.compile(
    r"""logger\.(info|success|warning|error)\(\s*f?(['"])(.*?)\2""",
    re.IGNORECASE,
)

_PORTUGUESE_TOKENS = (
    "nao",
    "não",
    "nenhum",
    "dados",
    "arquivo",
    "carregando",
    "iniciando",
    "concluido",
    "concluida",
    "sucesso",
    "falha",
    "salvo",
    "salvar",
    "removido",
    "removida",
    "entidade",
    "relacao",
    "triplas",
    "preprocess",
    "treino",
    "validacao",
    "teste",
    "pipeline",
    "otimizacao",
)

_ENGLISH_TOKENS = (
    "failed",
    "error",
    "invalid",
    "missing",
    "not found",
    "unknown",
    "deprecated",
    "unable",
    "exception",
    "traceback",
)


def _strip_fstring_expressions(message: str) -> str:
    return re.sub(r"\{[^}]*\}", "", message)


def _contains_token(message: str, token: str) -> bool:
    if " " in token:
        return token in message
    return re.search(rf"\\b{re.escape(token)}\\b", message) is not None


def _iter_log_messages(root: Path) -> list[LogMatch]:
    matches: list[LogMatch] = []
    for path in root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        content = path.read_text(encoding="utf-8")
        for idx, line in enumerate(content.splitlines(), start=1):
            for match in _LOG_PATTERN.finditer(line):
                level, _, message = match.groups()
                matches.append(LogMatch(path=path, line_no=idx, level=level, message=message))
    return matches


def test_info_success_are_portuguese() -> None:
    """Ensure info/success logs do not contain obvious English error tokens."""
    root_dirs = [Path("src/pff"), Path("scripts")]
    violations: list[LogMatch] = []
    for root in root_dirs:
        if not root.exists():
            continue
        for match in _iter_log_messages(root):
            if match.level.lower() not in ("info", "success"):
                continue
            message = _strip_fstring_expressions(match.message.lower())
            if any(_contains_token(message, token) for token in _ENGLISH_TOKENS):
                violations.append(match)

    assert not violations, "\n".join(
        f"{v.path}:{v.line_no} logger.{v.level} -> {v.message!r}" for v in violations
    )


def test_warning_error_are_english() -> None:
    """Ensure warning/error logs do not contain common Portuguese tokens."""
    root_dirs = [Path("src/pff"), Path("scripts")]
    violations: list[LogMatch] = []
    for root in root_dirs:
        if not root.exists():
            continue
        for match in _iter_log_messages(root):
            if match.level.lower() not in ("warning", "error"):
                continue
            message = _strip_fstring_expressions(match.message.lower())
            if any(_contains_token(message, token) for token in _PORTUGUESE_TOKENS):
                violations.append(match)

    assert not violations, "\n".join(
        f"{v.path}:{v.line_no} logger.{v.level} -> {v.message!r}" for v in violations
    )

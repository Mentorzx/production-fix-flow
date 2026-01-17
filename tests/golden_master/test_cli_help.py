from __future__ import annotations

import contextlib
import io
import unicodedata
from pathlib import Path

from pff.drivers.cli.main import CLIRunner

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _strip_log_lines(text: str) -> str:
    markers = (" INFO", " WARNING", " ERROR", " SUCCESS", " DEBUG")
    filtered: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("[") and any(marker in stripped for marker in markers):
            continue
        filtered.append(line)
    return "\n".join(filtered)


def _normalize_help(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = _strip_log_lines(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.strip() + "\n"


def _capture_help(argv: list[str]) -> str:
    runner = CLIRunner()
    parser = runner.parser
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            parser.parse_args(argv)
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    return _normalize_help(buffer.getvalue())


def test_cli_help_golden_master() -> None:
    expected = (FIXTURE_DIR / "cli_help.txt").read_text(encoding="utf-8")
    assert _capture_help(["--help"]) == expected


def test_cli_learn_help_golden_master() -> None:
    expected = (FIXTURE_DIR / "cli_learn_help.txt").read_text(encoding="utf-8")
    assert _capture_help(["learn", "--help"]) == expected


def test_cli_logs_help_golden_master() -> None:
    expected = (FIXTURE_DIR / "cli_logs_help.txt").read_text(encoding="utf-8")
    assert _capture_help(["logs", "--help"]) == expected

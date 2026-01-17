from __future__ import annotations

import subprocess
import sys
import unicodedata
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _run_hpo_help() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pff.drivers.cli.main", "hpo", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"pff hpo --help failed: code={result.returncode} stderr={result.stderr}"
        )
    return _normalize_help(result.stdout)


def test_hpo_help_golden_master() -> None:
    expected = (FIXTURE_DIR / "hpo_help.txt").read_text(encoding="utf-8")
    assert _run_hpo_help() == expected

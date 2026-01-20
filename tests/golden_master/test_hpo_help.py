from __future__ import annotations

import subprocess
import sys
import unicodedata
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _strip_log_lines(text: str) -> str:
    lines = text.splitlines()
    # Find the start of the actual help output
    try:
        start_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("usage:"))
        return "\n".join(lines[start_idx:])
    except StopIteration:
        return text


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
    # The expected content for hpo_help.txt has been updated.
    # The new content for the relevant section is:
    #   -h, --help            show this help message and exit
    #   --model {dslfm-kgc}   Modelo KGE (DSLFM-KGC com BERT + VAE + IBP + PC)
    #   --trials TRIALS       Numero de trials
    expected = (FIXTURE_DIR / "hpo_help.txt").read_text(encoding="utf-8")
    assert _run_hpo_help() == expected

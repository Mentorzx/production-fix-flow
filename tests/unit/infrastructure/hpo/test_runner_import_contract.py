"""Import-time contract tests for HPO runner module."""

from __future__ import annotations

import orjson
import subprocess
import sys


def test_runner_import_defers_trial_modules() -> None:
    """Importing runner must not eagerly import heavy trial modules."""
    script = """
import orjson
import sys

import pff.infrastructure.hpo.runner  # noqa: F401

targets = [
    "pff.infrastructure.hpo.trials.objective",
    "pff.infrastructure.hpo.trials.study",
    "pff.infrastructure.hpo.trials.pipeline",
]
print(orjson.dumps({name: (name in sys.modules) for name in targets}).decode())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = orjson.loads(result.stdout.strip())
    assert loaded["pff.infrastructure.hpo.trials.objective"] is False
    assert loaded["pff.infrastructure.hpo.trials.study"] is False
    assert loaded["pff.infrastructure.hpo.trials.pipeline"] is False

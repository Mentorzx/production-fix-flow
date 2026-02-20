"""Import-time contract tests for domain HPO package."""

from __future__ import annotations

import orjson
import subprocess
import sys


def test_hpo_package_import_is_lazy() -> None:
    """Importing pff.domain.hpo must not eagerly load heavy search modules."""
    script = """
import orjson
import sys

import pff.domain.hpo  # noqa: F401

targets = [
    "pff.domain.hpo.search_space",
    "pff.domain.learning.ml",
    "pff.domain.learning.dslfm",
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
    assert loaded["pff.domain.hpo.search_space"] is False
    assert loaded["pff.domain.learning.ml"] is False
    assert loaded["pff.domain.learning.dslfm"] is False


def test_hpo_model_symbol_still_available_after_lazy_import() -> None:
    """Public model symbols remain available through lazy exports."""
    script = """
import orjson
from pff.domain import hpo

print(orjson.dumps({
    "default_model": hpo.KGE_MODEL_DSLFM,
    "resolved": hpo.resolve_kge_model("dslfm_kgc"),
}).decode())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = orjson.loads(result.stdout.strip())
    assert payload["default_model"] == "dslfm"
    assert payload["resolved"] == "dslfm"

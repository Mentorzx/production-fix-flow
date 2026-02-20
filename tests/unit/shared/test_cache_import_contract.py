"""Import-time contract for cache module."""

from __future__ import annotations

import orjson
import subprocess
import sys


def test_cache_import_does_not_apply_settings() -> None:
    """Importing cache package must not apply config side effects."""
    script = """
import orjson
from pff.shared.core.cache import constants as cache_constants

print(orjson.dumps({
    "applied": cache_constants._CACHE_SETTINGS_APPLIED,
}).decode())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = orjson.loads(result.stdout.strip())
    assert payload["applied"] is False

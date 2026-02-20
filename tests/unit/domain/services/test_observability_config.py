"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/services/test_observability_config.py

"""

import os

from pff.infrastructure.observability import ObservabilityManager


def test_observability_env_overrides_config(monkeypatch):
    """Execute test observability env overrides config.



    Args:

        monkeypatch: Input value used by this callable.

    """

    monkeypatch.setenv("RAY_METRICS_EXPORT_INTERVAL_MS", "9999")
    ObservabilityManager()

    assert os.environ["RAY_METRICS_EXPORT_INTERVAL_MS"] == "9999"

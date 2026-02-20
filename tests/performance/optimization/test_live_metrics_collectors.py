"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/performance/optimization/test_live_metrics_collectors.py

"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pff.infrastructure.hpo.callbacks_internal.collectors import flatten_trial_metrics


class _DummyTrial:
    def __init__(self) -> None:
        """Execute init."""

        self.value = 0.42
        self.user_attrs: dict[str, Any] = {}
        now = datetime.now(timezone.utc)
        self.datetime_start = now
        self.datetime_complete = now + timedelta(seconds=12)


def test_flatten_trial_metrics_fallback_duration() -> None:
    """Execute test flatten trial metrics fallback duration.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    trial = _DummyTrial()
    metrics = flatten_trial_metrics(trial)
    assert metrics["score"] == 0.42
    assert metrics["duration"] == 12.0

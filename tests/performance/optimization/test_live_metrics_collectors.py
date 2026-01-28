from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pff.infrastructure.hpo.callbacks_internal.collectors import flatten_trial_metrics


class _DummyTrial:
    def __init__(self) -> None:
        self.value = 0.42
        self.user_attrs: dict[str, Any] = {}
        now = datetime.now(timezone.utc)
        self.datetime_start = now
        self.datetime_complete = now + timedelta(seconds=12)


def test_flatten_trial_metrics_fallback_duration() -> None:
    trial = _DummyTrial()
    metrics = flatten_trial_metrics(trial)
    assert metrics["score"] == 0.42
    assert metrics["duration"] == 12.0

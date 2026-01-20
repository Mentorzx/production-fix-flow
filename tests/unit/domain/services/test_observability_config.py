import os

from pff.infrastructure.observability import ObservabilityManager


def test_observability_env_overrides_config(monkeypatch):
    monkeypatch.setenv("RAY_METRICS_EXPORT_INTERVAL_MS", "9999")
    ObservabilityManager()

    assert os.environ["RAY_METRICS_EXPORT_INTERVAL_MS"] == "9999"

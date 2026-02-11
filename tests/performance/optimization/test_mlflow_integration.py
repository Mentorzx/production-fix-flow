from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from pff.infrastructure.hpo.config_loader import clear_config_cache
from pff.infrastructure.hpo.tracker import MLflowTracker, _load_mlflow_config


def _install_mlflow_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a lightweight mlflow mock into sys.modules."""
    mlflow_mock = MagicMock()
    mlflow_mock.get_experiment_by_name.return_value = None
    mlflow_mock.create_experiment.return_value = "exp-1"
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_mock)
    return mlflow_mock


def test_mlflow_uses_config_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracker should honor tracking URI and experiment from config."""
    mlflow_mock = _install_mlflow_mock(monkeypatch)
    monkeypatch.setattr(
        "pff.infrastructure.hpo.tracker._load_mlflow_config",
        lambda: {
            "enabled": True,
            "tracking_uri": "/custom/path",
            "experiment_name": "custom_exp",
        },
    )

    tracker = MLflowTracker()

    assert tracker.tracking_uri == "/custom/path"
    assert tracker.experiment_name == "custom_exp"
    mlflow_mock.set_tracking_uri.assert_called_once_with("/custom/path")


def test_mlflow_disabled_skips_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    """If config disables MLflow, tracker should not initialize mlflow."""
    mlflow_mock = _install_mlflow_mock(monkeypatch)
    monkeypatch.setattr(
        "pff.infrastructure.hpo.tracker._load_mlflow_config",
        lambda: {
            "enabled": False,
            "tracking_uri": "/custom/path",
            "experiment_name": "custom_exp",
        },
    )

    tracker = MLflowTracker()

    assert tracker.enabled is False
    assert tracker.mlflow is None
    assert tracker.get_tracking_uri() is None
    mlflow_mock.set_tracking_uri.assert_not_called()


def test_mlflow_defaults_when_config_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults should be used when config section is missing."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("PFF_MLFLOW_ENABLED", raising=False)
    monkeypatch.setattr(
        "pff.shared.core.file_manager.FileManager.read", lambda self, path: {}
    )
    clear_config_cache()

    config = _load_mlflow_config()

    from pff import settings

    default_uri = str(settings.OUTPUTS_DIR / "optimization" / "mlruns")

    assert config["tracking_uri"] == default_uri
    assert config["experiment_name"] == "pff_hpo"
    assert config["enabled"] is True


def test_mlflow_env_tracking_uri_strips_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracking URI should not keep surrounding quotes from env values."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", '"outputs/optimization/mlruns"')
    monkeypatch.delenv("PFF_MLFLOW_ENABLED", raising=False)
    monkeypatch.setattr(
        "pff.shared.core.file_manager.FileManager.read", lambda self, path: {}
    )

    config = _load_mlflow_config()

    assert config["tracking_uri"] == "outputs/optimization/mlruns"

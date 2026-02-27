"""


Notes:

    File: tests/unit/infrastructure/hpo/test_dashboard_search_space_advice.py

"""

from __future__ import annotations

from typing import Any

from pff.infrastructure.hpo.dashboard import server as dashboard_server


class _DummyAdvisor:
    def advise(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "recommendations": [
                {
                    "param_name": "learning_rate",
                    "action": "keep",
                    "confidence": "low",
                    "current_space": {"low": 0.001, "high": 0.01},
                    "proposed_space": {"low": 0.001, "high": 0.01},
                    "attempts_summary": {"count": 1},
                }
            ],
            "metadata": {"advisor_version": dashboard_server.ADVISOR_VERSION},
        }


class _RecordingAdvisor:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    def advise(self, *_args: Any, **kwargs: Any) -> dict[str, Any]:
        self.last_kwargs = dict(kwargs)
        return {
            "recommendations": [],
            "metadata": {"advisor_version": dashboard_server.ADVISOR_VERSION},
        }


def test_load_consolidated_data_fallback_when_profile_fails(monkeypatch) -> None:
    """Execute consolidated data fallback when dataset profiling fails."""

    def _raise_profile_error(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("profile failed")

    raw_data = {
        "studyName": "test",
        "direction": "maximize",
        "trials": [
            {
                "id": 1,
                "state": "COMPLETE",
                "value": 0.42,
                "params": {"learning_rate": 0.005},
            }
        ],
        "searchSpace": {"learning_rate": {"low": 0.001, "high": 0.01}},
        "importances": {},
    }

    monkeypatch.setattr(
        dashboard_server, "_load_raw_dashboard_data", lambda *_args, **_kwargs: dict(raw_data)
    )
    monkeypatch.setattr(dashboard_server, "_load_live_status", lambda: None)
    monkeypatch.setattr(
        dashboard_server, "_collect_terminal_logs", lambda live_status, _raw_data: live_status
    )
    monkeypatch.setattr(
        dashboard_server, "_inject_telemetry", lambda _handler, live_status: live_status or {}
    )
    monkeypatch.setattr(dashboard_server, "_consolidate_live_trial", lambda *_args: None)
    monkeypatch.setattr(dashboard_server, "_compute_best_value", lambda *_args: None)
    monkeypatch.setattr(dashboard_server, "_apply_lookback_memory", lambda *_args: None)
    monkeypatch.setattr(dashboard_server, "_get_search_space_advisor", lambda: _DummyAdvisor())
    monkeypatch.setattr(
        dashboard_server, "compute_dataset_profile_fingerprint", _raise_profile_error
    )

    handler = dashboard_server.PeakStateDashboardHandler.__new__(
        dashboard_server.PeakStateDashboardHandler
    )
    handler.hardware_manager = None

    payload = handler._load_consolidated_data()

    assert dashboard_server._has_usable_search_space_advice(payload.get("searchSpaceAdvice"))
    assert isinstance(payload.get("totalTrials"), int)
    assert payload.get("totalTrials", 0) > 0
    assert payload.get("total_trials_target") == payload.get("totalTrials")
    assert payload.get("completed_trials_all") == 1
    assert payload.get("completed_trials_non_warmstart") == 1
    assert payload.get("warmstart_trials") == 0


def test_apply_study_defaults_infers_search_space_from_trials() -> None:
    """Infer search-space metadata from trial params when serialized space is missing."""
    raw_data = {
        "trials": [
            {
                "id": 1,
                "state": "COMPLETE",
                "value": 0.33,
                "params": {
                    "learning_rate": 1e-4,
                    "batch_size": 256,
                    "t_norm": "product",
                    "self_adversarial": True,
                },
            },
            {
                "id": 2,
                "state": "COMPLETE",
                "value": 0.37,
                "params": {
                    "learning_rate": 2e-3,
                    "batch_size": 512,
                    "t_norm": "godel",
                    "self_adversarial": False,
                },
            },
        ],
        "searchSpace": {},
    }

    dashboard_server._apply_study_defaults(raw_data)

    search_space = raw_data.get("searchSpace")
    assert isinstance(search_space, dict)
    assert search_space["learning_rate"]["type"] == "float"
    assert search_space["learning_rate"]["low"] == 1e-4
    assert search_space["learning_rate"]["high"] == 2e-3
    assert search_space["batch_size"]["type"] == "categorical"
    assert search_space["batch_size"]["choices"] == [256, 512]
    assert search_space["t_norm"]["type"] == "categorical"
    assert search_space["self_adversarial"]["choices"] == [False, True]


def test_sanitize_live_status_fold_resets_out_of_range_value() -> None:
    """Out-of-range fold ids should be clamped to fold 0 for dashboard safety."""
    raw_data = {"totalFolds": 3}
    live_status = {"cv_fold_id": 7}

    dashboard_server._sanitize_live_status_fold(raw_data, live_status)

    assert live_status["cv_fold_id"] == 0


def test_serve_search_space_advice_refresh_bypasses_cached_payload(monkeypatch) -> None:
    """Refresh flag should force advisor recomputation even with usable cached advice."""
    cached_advice = {
        "recommendations": [{"param_name": "x", "action": "keep"}],
        "metadata": {"advisor_version": dashboard_server.ADVISOR_VERSION},
    }
    raw_data = {
        "searchSpaceAdvice": cached_advice,
        "searchSpace": {"x": {"low": 0.0, "high": 1.0}},
        "trials": [],
        "importances": {},
        "direction": "maximize",
        "studyName": "refresh_test",
    }
    advisor = _RecordingAdvisor()

    monkeypatch.setattr(
        dashboard_server.PeakStateDashboardHandler,
        "_load_consolidated_data",
        lambda self: dict(raw_data),
    )
    monkeypatch.setattr(dashboard_server, "_get_search_space_advisor", lambda: advisor)
    monkeypatch.setattr(
        dashboard_server,
        "compute_dataset_profile_fingerprint",
        lambda: ("fp-test", {"n_entities": 1, "n_relations": 1, "n_triples": 1, "density": 1.0}),
    )

    handler = dashboard_server.PeakStateDashboardHandler.__new__(
        dashboard_server.PeakStateDashboardHandler
    )
    handler.path = "/api/hpo/search-space-advice?refresh=1"

    payload = handler._serve_search_space_advice()
    assert isinstance(payload, dict)
    assert advisor.last_kwargs is not None
    assert advisor.last_kwargs.get("force_recompute") is True


def test_serve_search_space_advice_uses_cached_payload_without_refresh(monkeypatch) -> None:
    """Without refresh flag, usable cached payload should be returned directly."""
    cached_advice = {
        "recommendations": [{"param_name": "x", "action": "keep"}],
        "metadata": {"advisor_version": dashboard_server.ADVISOR_VERSION},
    }
    raw_data = {
        "searchSpaceAdvice": cached_advice,
        "searchSpace": {"x": {"low": 0.0, "high": 1.0}},
        "trials": [],
        "importances": {},
        "direction": "maximize",
        "studyName": "cached_test",
    }
    advisor = _RecordingAdvisor()

    monkeypatch.setattr(
        dashboard_server.PeakStateDashboardHandler,
        "_load_consolidated_data",
        lambda self: dict(raw_data),
    )
    monkeypatch.setattr(dashboard_server, "_get_search_space_advisor", lambda: advisor)

    handler = dashboard_server.PeakStateDashboardHandler.__new__(
        dashboard_server.PeakStateDashboardHandler
    )
    handler.path = "/api/hpo/search-space-advice"

    payload = handler._serve_search_space_advice()
    assert payload == cached_advice
    assert advisor.last_kwargs is None

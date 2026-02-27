"""Regression tests for trial count preparation."""

from __future__ import annotations

from dataclasses import dataclass

from optuna.trial import TrialState

from pff.infrastructure.hpo.trials.study import (
    _build_study_result_payload,
    _prepare_trial_counts,
)


@dataclass
class _DummyTrial:
    state: object
    user_attrs: dict
    params: dict | None = None
    value: float | None = None
    values: list[float] | None = None
    number: int = 0
    _trial_id: int | None = None
    _storage: object | None = None


@dataclass
class _DummyStudy:
    trials: list[_DummyTrial]
    study_name: str = "study_a"


@dataclass
class _DummyStorage:
    attrs_by_id: dict[int, dict]

    def get_trial_system_attrs(self, trial_id: int) -> dict:
        return self.attrs_by_id.get(trial_id, {})


def test_prepare_trial_counts_respects_requested_n_trials() -> None:
    """Requested n_trials must cap the run even if expected_trials is larger."""
    study = _DummyStudy(trials=[])
    completed, total_target, remaining, callback = _prepare_trial_counts(
        study=study,
        n_trials=1,
        expected_trials=50,
        resume_mode=False,
    )
    assert completed == 0
    assert total_target == 1
    assert remaining == 1
    assert callback.max_trials == 1


def test_prepare_trial_counts_excludes_storage_warmstart_trials() -> None:
    """Warmstart detection must use study storage attributes when present."""
    storage = _DummyStorage(attrs_by_id={101: {"warmstart_seed": True}, 102: {}})
    study = _DummyStudy(
        trials=[
            _DummyTrial(
                state=TrialState.COMPLETE,
                user_attrs={},
                number=0,
                _trial_id=101,
            ),
            _DummyTrial(
                state=TrialState.COMPLETE,
                user_attrs={},
                number=1,
                _trial_id=102,
            ),
        ]
    )
    study._storage = storage  # type: ignore[attr-defined]

    completed, total_target, remaining, callback = _prepare_trial_counts(
        study=study,
        n_trials=5,
        expected_trials=5,
        resume_mode=True,
    )

    assert completed == 1
    assert total_target == 5
    assert remaining == 4
    assert callback.max_trials == 5


def test_build_study_result_payload_uses_normalized_trial_counters() -> None:
    """Result payload must expose explicit count fields with non-warmstart semantics."""
    storage = _DummyStorage(attrs_by_id={201: {"warmstart_seed": True}, 202: {}, 203: {}})
    trials = [
        _DummyTrial(
            state=TrialState.COMPLETE,
            user_attrs={},
            params={"x": 1},
            value=0.1,
            values=None,
            number=0,
            _trial_id=201,
        ),
        _DummyTrial(
            state=TrialState.COMPLETE,
            user_attrs={},
            params={"x": 2},
            value=0.2,
            values=None,
            number=1,
            _trial_id=202,
        ),
        _DummyTrial(
            state=TrialState.RUNNING,
            user_attrs={"warmstart": False},
            params={"x": 3},
            value=None,
            values=None,
            number=2,
            _trial_id=203,
        ),
    ]
    study = _DummyStudy(trials=trials)
    study._storage = storage  # type: ignore[attr-defined]

    payload = _build_study_result_payload(
        study=study,
        best_params={"x": 2},
        best_value=0.2,
        optimization_time=12.3,
        interrupted=False,
        multi_enabled=False,
        pareto_front=[],
        live_plot_callback=None,
        total_target_trials=50,
    )

    assert payload["n_trials"] == 1
    assert payload["total_trials_target"] == 50
    assert payload["completed_trials_non_warmstart"] == 1
    assert payload["completed_trials_all"] == 2
    assert payload["warmstart_trials"] == 1

"""Regression tests for trial count preparation."""

from __future__ import annotations

from dataclasses import dataclass

from pff.infrastructure.hpo.trials.study import _prepare_trial_counts


@dataclass
class _DummyTrial:
    state: object
    user_attrs: dict
    system_attrs: dict


@dataclass
class _DummyStudy:
    trials: list[_DummyTrial]


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

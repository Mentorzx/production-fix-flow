from __future__ import annotations

from typing import Any

from pff.infrastructure.hpo.trials.study import _run_optuna_optimization


def test_run_optuna_optimization_uses_configured_parallel_params() -> None:
    """Optuna execution should honor configured n_jobs and gc_after_trial."""

    called: dict[str, Any] = {}

    class _DummyStudy:
        def optimize(self, objective, **kwargs):  # noqa: ANN001
            called["objective"] = objective
            called["kwargs"] = kwargs

    def _objective(_trial) -> float:
        return 0.0

    interrupted = _run_optuna_optimization(
        study=_DummyStudy(),
        objective=_objective,
        remaining_trials=3,
        n_jobs=4,
        callbacks=[None, object()],
        gc_after_trial=False,
    )

    assert interrupted is False
    assert called["kwargs"]["n_trials"] == 3
    assert called["kwargs"]["n_jobs"] == 4
    assert called["kwargs"]["gc_after_trial"] is False
    assert len(called["kwargs"]["callbacks"]) == 1


def test_run_optuna_optimization_skips_when_no_remaining_trials() -> None:
    """No optimize call should occur when remaining_trials <= 0."""

    class _DummyStudy:
        def optimize(self, *_args, **_kwargs):  # noqa: ANN001
            raise AssertionError("optimize should not be called")

    def _objective(_trial) -> float:
        return 0.0

    interrupted = _run_optuna_optimization(
        study=_DummyStudy(),
        objective=_objective,
        remaining_trials=0,
        n_jobs=2,
        callbacks=[],
        gc_after_trial=True,
    )

    assert interrupted is False

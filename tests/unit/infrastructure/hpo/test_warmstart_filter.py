"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/infrastructure/hpo/test_warmstart_filter.py

"""

from __future__ import annotations

from pathlib import Path

import pytest
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import TrialState

from pff.infrastructure.hpo.runner import HPOMemoryConfig, PersistentBestTrialMemory


class _DummyTrial:
    def __init__(self, params: dict) -> None:
        """Execute init.



        Args:

            params: Input value used by this callable.

        """

        self.params = params


class _DummyStudy:
    def __init__(self) -> None:
        """Execute init."""

        self.trials: list[_DummyTrial] = []
        self.enqueued: list[tuple[dict, dict]] = []
        self.added: list = []

    def enqueue_trial(self, params: dict, user_attrs: dict | None = None) -> None:
        """Execute enqueue trial.



        Args:

            params: Input value used by this callable.

            user_attrs: Optional input value.

        """

        self.enqueued.append((params, user_attrs or {}))

    def add_trial(self, trial) -> None:
        """Execute add trial.



        Args:

            trial: Input value used by this callable.

        """

        self.added.append(trial)


def test_warmstart_filters_out_of_range_params(tmp_path: Path) -> None:
    """Execute test warmstart filters out of range params.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    config = HPOMemoryConfig(enabled=True, warmstart_trials=1, storage_subdir="warmstart")
    memory = PersistentBestTrialMemory(output_dir=tmp_path, config=config)

    distributions = memory._serialize_distributions(
        {
            "min_delta": FloatDistribution(low=1e-5, high=5e-4),
            "validate_every": IntDistribution(low=4, high=6),
            "learning_rate": FloatDistribution(low=1e-5, high=1e-3),
            "t_norm": CategoricalDistribution(choices=("product", "lukasiewicz", "godel")),
        }
    )
    memory.entries = [
        {
            "params": {
                "min_delta": 1e-3,
                "validate_every": 3,
                "learning_rate": 5e-4,
                "t_norm": "product",
            },
            "value": 0.42,
            "distributions": distributions,
        }
    ]

    study = _DummyStudy()
    injected = memory.warmstart_study(study)

    assert injected == 1
    assert len(study.added) == 1
    assert study.added[0].params == {"learning_rate": 5e-4, "t_norm": "product"}
    assert study.added[0].user_attrs.get("warmstart") is True


def test_warmstart_handles_internal_contains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute test warmstart handles internal contains.



    Args:

        tmp_path: Input value used by this callable.

        monkeypatch: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    class _DummyDist:
        def to_internal_repr(self, value: object) -> int:
            """Execute to internal repr.



            Args:

                value: Input value used by this callable.



            Returns:

                Return value produced by the callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            if value == "product":
                return 0
            return 10

        def _contains(self, internal_value: int) -> bool:
            return internal_value == 0

    config = HPOMemoryConfig(enabled=True, warmstart_trials=1, storage_subdir="warmstart")
    memory = PersistentBestTrialMemory(output_dir=tmp_path, config=config)
    memory.entries = [
        {
            "params": {"t_norm": "product"},
            "value": 0.42,
            "distributions": {"t_norm": {"type": "categorical", "choices": ["product"]}},
        }
    ]

    monkeypatch.setattr(
        PersistentBestTrialMemory,
        "_deserialize_distributions",
        staticmethod(lambda _: {"t_norm": _DummyDist()}),
    )

    study = _DummyStudy()
    injected = memory.warmstart_study(study)

    assert injected == 1
    assert len(study.added) == 1
    assert study.added[0].params == {"t_norm": "product"}


def test_warmstart_prefers_current_distributions(tmp_path: Path) -> None:
    """Execute test warmstart prefers current distributions.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    config = HPOMemoryConfig(enabled=True, warmstart_trials=1, storage_subdir="warmstart")
    memory = PersistentBestTrialMemory(output_dir=tmp_path, config=config)
    memory.set_current_distributions(
        {
            "min_delta": FloatDistribution(low=1e-5, high=5e-4),
            "validate_every": IntDistribution(low=4, high=6),
            "t_norm": CategoricalDistribution(choices=("product", "lukasiewicz")),
        }
    )
    memory.entries = [
        {
            "params": {
                "min_delta": 1e-3,
                "validate_every": 5,
                "t_norm": "product",
            },
            "value": 0.33,
            "distributions": memory._serialize_distributions(
                {
                    "min_delta": FloatDistribution(low=1e-5, high=1e-2),
                    "validate_every": IntDistribution(low=1, high=10),
                }
            ),
        }
    ]

    study = _DummyStudy()
    injected = memory.warmstart_study(study)

    assert injected == 1
    assert len(study.added) == 1
    assert study.added[0].params == {"validate_every": 5, "t_norm": "product"}


def test_warmstart_keeps_value_when_distribution_has_no_contains(tmp_path: Path) -> None:
    """Execute test warmstart keeps value when distribution has no contains.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    class _NoContainsDist:
        pass

    config = HPOMemoryConfig(enabled=True, warmstart_trials=1, storage_subdir="warmstart")
    memory = PersistentBestTrialMemory(output_dir=tmp_path, config=config)
    memory.entries = [
        {
            "params": {"custom_param": "x"},
            "value": 0.25,
            "distributions": {},
        }
    ]
    memory.set_current_distributions({"custom_param": _NoContainsDist()})

    study = _DummyStudy()
    injected = memory.warmstart_study(study)

    assert injected == 0
    assert len(study.added) == 0
    assert len(study.enqueued) == 1
    assert study.enqueued[0][0] == {"custom_param": "x"}


def test_record_trial_keeps_only_top_k_values(tmp_path: Path) -> None:
    """Keep only top-k best trials in persistent memory."""

    class _DummyCompletedTrial:
        def __init__(self, number: int, value: float) -> None:
            self.number = number
            self.value = value
            self.params = {"lr": value}
            self.distributions = {}
            self.state = TrialState.COMPLETE

    class _DummyNamedStudy:
        study_name = "unit-study"

    config = HPOMemoryConfig(
        enabled=True, top_k_trials=3, warmstart_trials=2, storage_subdir="topk"
    )
    memory = PersistentBestTrialMemory(output_dir=tmp_path, config=config)
    study = _DummyNamedStudy()

    for idx, value in enumerate([0.1, 0.8, 0.4, 0.9, 0.2]):
        memory.record_trial(study, _DummyCompletedTrial(idx, value))

    kept_values = [entry["value"] for entry in memory.entries]
    assert sorted(kept_values, reverse=True) == [0.9, 0.8, 0.4]

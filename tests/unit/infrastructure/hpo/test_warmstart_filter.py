from __future__ import annotations

from pathlib import Path

import pytest
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from pff.infrastructure.hpo.runner import HPOMemoryConfig, PersistentBestTrialMemory


class _DummyTrial:
    def __init__(self, params: dict) -> None:
        self.params = params


class _DummyStudy:
    def __init__(self) -> None:
        self.trials: list[_DummyTrial] = []
        self.enqueued: list[tuple[dict, dict]] = []
        self.added: list = []

    def enqueue_trial(self, params: dict, user_attrs: dict | None = None) -> None:
        self.enqueued.append((params, user_attrs or {}))

    def add_trial(self, trial) -> None:
        self.added.append(trial)


def test_warmstart_filters_out_of_range_params(tmp_path: Path) -> None:
    config = HPOMemoryConfig(
        enabled=True, warmstart_trials=1, storage_subdir="warmstart"
    )
    memory = PersistentBestTrialMemory(output_dir=tmp_path, config=config)

    distributions = memory._serialize_distributions(
        {
            "min_delta": FloatDistribution(low=1e-5, high=5e-4),
            "validate_every": IntDistribution(low=4, high=6),
            "learning_rate": FloatDistribution(low=1e-5, high=1e-3),
            "t_norm": CategoricalDistribution(
                choices=("product", "lukasiewicz", "godel")
            ),
        }
    )
    memory.entries = [
        {
            "params": {
                "min_delta": 1e-3,  # out of range -> drop
                "validate_every": 3,  # out of range -> drop
                "learning_rate": 5e-4,  # ok
                "t_norm": "product",  # ok categorical
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
    class _DummyDist:
        def to_internal_repr(self, value: object) -> int:
            if value == "product":
                return 0
            return 10

        def _contains(self, internal_value: object) -> bool:
            return int(internal_value) == 0

    config = HPOMemoryConfig(
        enabled=True, warmstart_trials=1, storage_subdir="warmstart"
    )
    memory = PersistentBestTrialMemory(output_dir=tmp_path, config=config)
    memory.entries = [
        {
            "params": {"t_norm": "product"},
            "value": 0.42,
            "distributions": {
                "t_norm": {"type": "categorical", "choices": ["product"]}
            },
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
    config = HPOMemoryConfig(
        enabled=True, warmstart_trials=1, storage_subdir="warmstart"
    )
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
                "min_delta": 1e-3,  # out of current range -> drop
                "validate_every": 5,  # ok
                "t_norm": "product",  # ok
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

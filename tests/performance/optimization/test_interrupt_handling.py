"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/performance/optimization/test_interrupt_handling.py

"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import polars as pl
import pytest

from pff.infrastructure.hpo.config_loader import load_storage_settings
from pff.infrastructure.hpo.distributed import DistributedOptimizer
from pff.infrastructure.hpo.storage import create_optuna_storage
from pff.infrastructure.hpo.strategies.base import OptimizationConfig
from pff.infrastructure.hpo.strategies.optuna_impl import OptunaStrategy
from pff.infrastructure.hpo.trials.pipeline import (
    TrialEvaluationConfig,
    TrialEvaluationPipeline,
)
from pff.infrastructure.hpo.trials.study import create_study_and_run
from pff.shared.ops.global_interrupt_manager import get_interrupt_manager


def test_pipeline_stops_when_interrupt_flag_set() -> None:
    """Execute test pipeline stops when interrupt flag set.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    manager = get_interrupt_manager()
    manager.reset()
    manager.force_stop("test-interrupt")

    config = TrialEvaluationConfig(
        params={},
        train_df=pl.DataFrame({"h": [1], "r": [1], "t": [1]}),
        valid_df=pl.DataFrame({"h": [1], "r": [1], "t": [1]}),
        target_entity_ratio=0.1,
        trial_number=0,
        trial_output_root=Path("outputs/test_interrupt"),
        artifact_manager=MagicMock(store=MagicMock(), study_name="test"),
    )

    with pytest.raises(KeyboardInterrupt):
        TrialEvaluationPipeline(**config.__dict__).run()

    manager.reset()


def test_optuna_strategy_returns_result_on_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute test optuna strategy returns result on interrupt.



    Args:

        monkeypatch: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Raises:

        Exception: Propagates domain-specific failures with context.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    manager = get_interrupt_manager()
    manager.reset()

    class DummyTrial:
        """Represent DummyTrial."""

        def __init__(self) -> None:
            """Execute init."""

            self.params = {"x": 1}
            self.value = 0.5
            self.number = 0
            self.state = "COMPLETE"
            self.intermediate_values = {}
            self.user_attrs = {}

    class DummyStudy:
        """Represent DummyStudy."""

        def __init__(self) -> None:
            """Execute init."""

            self.trials = [DummyTrial()]
            self.best_trial = self.trials[0]

        def optimize(self, *args, **kwargs) -> None:  # noqa: ARG002
            """Execute optimize.



            Args:

                *args: Additional positional arguments.

                **kwargs: Additional keyword arguments.

            """

            raise KeyboardInterrupt

    config = OptimizationConfig(n_trials=2)
    strategy = OptunaStrategy(config)

    def _create_dummy_study() -> DummyStudy:
        strategy.study = DummyStudy()
        return strategy.study

    monkeypatch.setattr(
        strategy,
        "create_study",
        _create_dummy_study,
    )

    result = strategy.run_optimization(lambda trial: 1.0, {})

    assert result.best_value == pytest.approx(0.5)
    assert result.n_trials == 1

    manager.reset()


def test_optuna_study_interrupt_returns_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute test optuna study interrupt returns partial.



    Args:

        tmp_path: Input value used by this callable.

        monkeypatch: Input value used by this callable.



    Raises:

        Exception: Propagates domain-specific failures with context.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    import optuna

    from pff.shared.core.file_manager import FileManager

    manager = get_interrupt_manager()
    manager.reset()

    def fake_optimize(self, *args, **kwargs):  # noqa: ARG002
        """Execute fake optimize.



        Args:

            *args: Additional positional arguments.

            **kwargs: Additional keyword arguments.

        """

        raise KeyboardInterrupt

    monkeypatch.setattr(optuna.study.Study, "optimize", fake_optimize, raising=False)
    monkeypatch.setattr(
        "pff.infrastructure.hpo.trials.study.load_optuna_settings",
        lambda fm: {
            "tpe": {
                "n_startup_trials": 1,
                "multivariate": False,
                "group": False,
                "constant_liar": False,
            },
            "hyperband": {"min_resource": 1, "max_resource": 2, "reduction_factor": 2},
        },
    )
    monkeypatch.setattr(
        "pff.infrastructure.hpo.trials.study.load_live_plot_settings",
        lambda fm: {"enabled": False},
    )
    monkeypatch.setattr(
        "pff.infrastructure.hpo.runner.BestModelSaverCallback",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("pff.settings.OUTPUTS_DIR", tmp_path, raising=False)

    fm = FileManager()
    study_suffix = hashlib.sha1(str(tmp_path).encode("utf-8")).hexdigest()[:8]
    study_name = f"int_test_{study_suffix}"
    storage_backend = str(load_storage_settings(fm).get("backend", "postgres")).lower()
    if storage_backend in {"postgres", "postgresql", "rdb", "rdbstorage"}:
        storage, storage_url = create_optuna_storage(
            storage_path=tmp_path / "opt.db", file_manager=fm
        )
        study_not_found = getattr(optuna.exceptions, "StudyNotFound", KeyError)
        try:
            optuna.delete_study(
                study_name=study_name,
                storage=storage if storage is not None else storage_url,
            )
        except (KeyError, study_not_found):
            pass

    result = create_study_and_run(
        study_name=study_name,
        storage_path=tmp_path / "opt.db",
        checkpoint_path=tmp_path / "ckpt.json",
        checkpoint_key="test_key",
        checkpoint_store=MagicMock(),
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "out",
        n_trials=1,
        expected_trials=1,
        resume_mode=False,
        checkpoint_data=None,
        hpo_memory_config={},
        trial_memory=SimpleNamespace(),
        warmstart_callback=None,
        objective_fn=lambda trial: 0.1,
        artifact_manager=MagicMock(store=MagicMock(), study_name="test"),
        enable_mlflow=False,
        file_manager=fm,
    )

    assert result["interrupted"] is True
    assert result["best_params"] == {}
    assert result["best_value"] is None
    manager.reset()


def test_distributed_optimizer_respects_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute test distributed optimizer respects interrupt.



    Args:

        monkeypatch: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    manager = get_interrupt_manager()
    manager.reset()
    manager.force_stop("test-interrupt")

    dist = DistributedOptimizer()

    result = dist.run_distributed(lambda trial: 1.0, {"x": [0.0, 1.0]}, n_trials=2, num_workers=1)

    assert result["interrupted"] is True
    manager.reset()

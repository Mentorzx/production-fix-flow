from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from pff.utils.ops.global_interrupt_manager import get_interrupt_manager
from scripts.optimization.trials.pipeline import TrialEvaluationPipeline, TrialEvaluationConfig, TrialArtifactManager
from scripts.optimization.trials.study import create_study_and_run
from scripts.optimization.advanced import DistributedOptimizer
from scripts.optimization.strategies.base import OptimizationConfig
from scripts.optimization.strategies.optuna_impl import OptunaStrategy
from scripts.optimization.strategies.hyperopt_impl import HyperoptStrategy


def test_pipeline_stops_when_interrupt_flag_set() -> None:
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
        artifact_manager=TrialArtifactManager(),
    )

    with pytest.raises(KeyboardInterrupt):
        TrialEvaluationPipeline(**config.__dict__).run()

    manager.reset()


def test_optuna_strategy_returns_result_on_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = get_interrupt_manager()
    manager.reset()

    class DummyTrial:
        def __init__(self) -> None:
            self.params = {"x": 1}
            self.value = 0.5
            self.number = 0
            self.state = "COMPLETE"
            self.intermediate_values = {}
            self.user_attrs = {}

    class DummyStudy:
        def __init__(self) -> None:
            self.trials = [DummyTrial()]
            self.best_trial = self.trials[0]

        def optimize(self, *args, **kwargs) -> None:  # noqa: ARG002
            raise KeyboardInterrupt

    config = OptimizationConfig(n_trials=2)
    strategy = OptunaStrategy(config)
    monkeypatch.setattr(strategy, "create_study", lambda: setattr(strategy, "study", DummyStudy()) or strategy.study)

    result = strategy.run_optimization(lambda trial: 1.0, {})

    assert result.best_value == pytest.approx(0.5)
    assert result.n_trials == 1

    manager.reset()


def test_hyperopt_strategy_returns_result_on_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("hyperopt", reason="Hyperopt não instalado")
    manager = get_interrupt_manager()
    manager.reset()

    config = OptimizationConfig(n_trials=2)
    strategy = HyperoptStrategy(config)

    dummy_trials = SimpleNamespace(
        trials=[
            {
                "result": {"loss": 1.0, "status": strategy.STATUS_OK},
                "misc": {"vals": {"x": [0.1]}},
            }
        ]
    )

    monkeypatch.setattr(strategy, "create_study", lambda: setattr(strategy, "trials", dummy_trials) or strategy.trials)
    monkeypatch.setattr(strategy, "fmin", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt))

    result = strategy.run_optimization(lambda trial: 1.0, {"x": {"type": "float", "low": 0.0, "high": 1.0}})

    assert result.best_value == pytest.approx(-1.0)
    assert result.n_trials == 1

    manager.reset()


def test_optuna_study_interrupt_returns_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pff.utils.core.file_manager import FileManager
    import optuna

    manager = get_interrupt_manager()
    manager.reset()

    def fake_optimize(self, *args, **kwargs):  # noqa: ARG002
        raise KeyboardInterrupt

    monkeypatch.setattr(optuna.study.Study, "optimize", fake_optimize, raising=False)
    monkeypatch.setattr(
        "scripts.optimization.trials.study.load_optuna_settings",
        lambda fm: {"tpe": {"n_startup_trials": 1, "multivariate": False, "group": False, "constant_liar": False}, "hyperband": {"min_resource": 1, "max_resource": 2, "reduction_factor": 2}},
    )
    monkeypatch.setattr("scripts.optimization.trials.study.load_live_plot_settings", lambda fm: {"enabled": False})
    monkeypatch.setattr("scripts.optimization.core.BestModelSaverCallback", lambda *args, **kwargs: None)
    monkeypatch.setattr("pff.settings.OUTPUTS_DIR", tmp_path, raising=False)

    result = create_study_and_run(
        study_name="int_test",
        storage_path=tmp_path / "opt.db",
        checkpoint_path=tmp_path / "ckpt.json",
        output_dir=tmp_path / "out",
        n_trials=1,
        expected_trials=1,
        resume_mode=False,
        checkpoint_data=None,
        hpo_memory_config={},
        trial_memory=SimpleNamespace(),
        warmstart_callback=None,
        objective_fn=lambda trial: 0.1,
        artifact_manager=TrialArtifactManager(),
        enable_mlflow=False,
        file_manager=FileManager(),
    )

    assert result["interrupted"] is True
    assert result["best_params"] == {}
    assert result["best_value"] is None
    manager.reset()


def test_distributed_optimizer_respects_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = get_interrupt_manager()
    manager.reset()
    manager.force_stop("test-interrupt")

    dist = DistributedOptimizer()

    result = dist.run_distributed(lambda trial: 1.0, {"x": [0.0, 1.0]}, n_trials=2, num_workers=1)

    assert result["interrupted"] is True
    manager.reset()

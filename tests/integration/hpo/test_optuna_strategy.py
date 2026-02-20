"""Tests for pff/infrastructure/hpo/strategies/optuna_impl.py.

Tests OptunaStrategy and AutoOptunaStrategy without running full
optimization, using mocks where appropriate.
"""

from __future__ import annotations

import pytest

pytest.importorskip("optuna")

from pff.infrastructure.hpo.strategies.base import (  # noqa: E402
    OptimizationConfig,
    TrialResult,
)
from pff.infrastructure.hpo.strategies.optuna_impl import (  # noqa: E402
    AutoOptunaStrategy,
    OptunaStrategy,
    _load_sampler_config,
)

# ─────────────────────────── OptimizationConfig Tests ───────────────────────


class TestOptimizationConfigForOptuna:
    """Tests for OptimizationConfig used with OptunaStrategy."""

    def test_config_defaults(self) -> None:
        """Config should have sensible defaults."""
        config = OptimizationConfig(n_trials=10)
        assert config.n_trials == 10
        assert config.direction in ("minimize", "maximize")

    def test_config_with_study_name(self) -> None:
        """Config should accept study_name."""
        config = OptimizationConfig(n_trials=10, study_name="my_study")
        assert config.study_name == "my_study"

    def test_config_with_pruning(self) -> None:
        """Config should accept enable_pruning."""
        config = OptimizationConfig(n_trials=10, enable_pruning=True)
        assert config.enable_pruning is True


# ─────────────────────────── OptunaStrategy Init Tests ───────────────────────


class TestOptunaStrategyInit:
    """Tests for OptunaStrategy initialization."""

    def test_strategy_creation(self) -> None:
        """OptunaStrategy should be creatable."""
        config = OptimizationConfig(n_trials=5)
        strategy = OptunaStrategy(config)
        assert strategy._framework_name == "optuna"
        assert strategy.optuna is not None

    def test_strategy_with_study_name(self) -> None:
        """Strategy should use provided study name."""
        config = OptimizationConfig(n_trials=5, study_name="test_study")
        strategy = OptunaStrategy(config)
        assert strategy._study_name == "test_study"

    def test_strategy_generates_study_name(self) -> None:
        """Strategy should generate study name if not provided."""
        config = OptimizationConfig(n_trials=5)
        strategy = OptunaStrategy(config)
        assert "optuna_study_" in strategy._study_name


# ─────────────────────────── OptunaStrategy Study Creation Tests ─────────────


class TestOptunaStrategyStudyCreation:
    """Tests for study creation."""

    def test_create_study(self) -> None:
        """create_study should create an Optuna study."""
        config = OptimizationConfig(n_trials=5, direction="minimize")
        strategy = OptunaStrategy(config)
        study = strategy.create_study()
        assert study is not None
        assert strategy.study is not None

    def test_create_study_maximize(self) -> None:
        """create_study should respect maximize direction."""
        config = OptimizationConfig(n_trials=5, direction="maximize")
        strategy = OptunaStrategy(config)
        study = strategy.create_study()
        assert study.direction.name == "MAXIMIZE"

    def test_create_study_minimize(self) -> None:
        """create_study should respect minimize direction."""
        config = OptimizationConfig(n_trials=5, direction="minimize")
        strategy = OptunaStrategy(config)
        study = strategy.create_study()
        assert study.direction.name == "MINIMIZE"

    def test_create_study_with_pruning(self) -> None:
        """create_study with pruning should set pruner."""
        config = OptimizationConfig(n_trials=5, enable_pruning=True)
        strategy = OptunaStrategy(config)
        study = strategy.create_study()
        assert study.pruner is not None


# ─────────────────────────── OptunaStrategy suggest_params Tests ─────────────


class TestOptunaStrategySuggestParams:
    """Tests for suggest_params method."""

    def test_suggest_params_float_range(self) -> None:
        """suggest_params should handle float range."""
        config = OptimizationConfig(n_trials=5)
        strategy = OptunaStrategy(config)
        strategy.create_study()

        search_space = {"learning_rate": (0.001, 0.1)}

        trial = strategy.study.ask()
        params = strategy.suggest_params(trial, search_space)

        assert "learning_rate" in params
        assert 0.001 <= params["learning_rate"] <= 0.1

    def test_suggest_params_categorical(self) -> None:
        """suggest_params should handle categorical values."""
        config = OptimizationConfig(n_trials=5)
        strategy = OptunaStrategy(config)
        strategy.create_study()

        search_space = {"optimizer": ["adam", "sgd", "rmsprop"]}

        trial = strategy.study.ask()
        params = strategy.suggest_params(trial, search_space)

        assert "optimizer" in params
        assert params["optimizer"] in ["adam", "sgd", "rmsprop"]

    def test_suggest_params_dict_int(self) -> None:
        """suggest_params should handle dict with int type."""
        config = OptimizationConfig(n_trials=5)
        strategy = OptunaStrategy(config)
        strategy.create_study()

        search_space = {"hidden_size": {"type": "int", "low": 32, "high": 256, "step": 32}}

        trial = strategy.study.ask()
        params = strategy.suggest_params(trial, search_space)

        assert "hidden_size" in params
        assert 32 <= params["hidden_size"] <= 256
        assert params["hidden_size"] % 32 == 0

    def test_suggest_params_dict_float_log(self) -> None:
        """suggest_params should handle log-scale float."""
        config = OptimizationConfig(n_trials=5)
        strategy = OptunaStrategy(config)
        strategy.create_study()

        search_space = {"learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True}}

        trial = strategy.study.ask()
        params = strategy.suggest_params(trial, search_space)

        assert "learning_rate" in params
        assert 1e-5 <= params["learning_rate"] <= 1e-1

    def test_suggest_params_dict_categorical(self) -> None:
        """suggest_params should handle dict with categorical type."""
        config = OptimizationConfig(n_trials=5)
        strategy = OptunaStrategy(config)
        strategy.create_study()

        search_space = {
            "activation": {
                "type": "categorical",
                "choices": ["relu", "tanh", "sigmoid"],
            }
        }

        trial = strategy.study.ask()
        params = strategy.suggest_params(trial, search_space)

        assert "activation" in params
        assert params["activation"] in ["relu", "tanh", "sigmoid"]


# ─────────────────────────── OptunaStrategy Trial Methods Tests ──────────────


class TestOptunaStrategyTrialMethods:
    """Tests for trial-related methods."""

    def test_get_all_trials_empty(self) -> None:
        """get_all_trials should return empty list before optimization."""
        config = OptimizationConfig(n_trials=5)
        strategy = OptunaStrategy(config)
        strategy.create_study()

        trials = strategy.get_all_trials()
        assert trials == []

    def test_get_best_trial_raises_before_optimization(self) -> None:
        """get_best_trial should raise before any trials."""
        config = OptimizationConfig(n_trials=5)
        strategy = OptunaStrategy(config)
        strategy.create_study()

        with pytest.raises(Exception):
            strategy.get_best_trial()

    def test_get_optimization_history_empty(self) -> None:
        """get_optimization_history should return empty list before optimization."""
        config = OptimizationConfig(n_trials=5)
        strategy = OptunaStrategy(config)
        strategy.create_study()

        history = strategy.get_optimization_history()
        assert history == []

    def test_get_param_importances_empty(self) -> None:
        """get_param_importances should return empty dict before optimization."""
        config = OptimizationConfig(n_trials=5)
        strategy = OptunaStrategy(config)
        strategy.create_study()

        importances = strategy.get_param_importances()
        assert importances == {}


# ─────────────────────────── OptunaStrategy Pruner Tests ─────────────────────


class TestOptunaStrategyPruner:
    """Tests for pruner creation."""

    def test_create_hyperband_pruner(self) -> None:
        """Default pruner should be HyperbandPruner."""
        config = OptimizationConfig(n_trials=5, enable_pruning=True)
        strategy = OptunaStrategy(config)
        pruner = strategy._create_pruner()
        assert pruner is not None
        assert "Hyperband" in pruner.__class__.__name__

    def test_create_median_pruner(self) -> None:
        """Should create MedianPruner when specified."""
        config = OptimizationConfig(n_trials=5, enable_pruning=True)
        config.pruner_type = "median"  # type: ignore
        strategy = OptunaStrategy(config)
        pruner = strategy._create_pruner()
        assert "Median" in pruner.__class__.__name__


# ─────────────────────────── AutoOptunaStrategy Tests ─────────────────────────


class TestAutoOptunaStrategy:
    """Tests for AutoOptunaStrategy."""

    def test_auto_strategy_creation(self) -> None:
        """AutoOptunaStrategy should be creatable."""
        config = OptimizationConfig(n_trials=5)
        strategy = AutoOptunaStrategy(config)
        assert strategy._framework_name == "optuna-auto"

    def test_auto_strategy_multi_objective(self) -> None:
        """AutoOptunaStrategy should support multi-objective flag."""
        config = OptimizationConfig(n_trials=5)
        strategy = AutoOptunaStrategy(config, is_multi_objective=True)
        assert strategy.is_multi_objective is True

    def test_auto_select_sampler_single_objective(self) -> None:
        """Auto sampler for single objective should be TPE or CMA-ES."""
        config = OptimizationConfig(n_trials=100)
        strategy = AutoOptunaStrategy(config, is_multi_objective=False)
        strategy.create_study()
        sampler = strategy._auto_select_sampler()
        # Should be TPE for large n_trials
        sampler_name = sampler.__class__.__name__
        assert "TPE" in sampler_name or "CmaEs" in sampler_name or "Auto" in sampler_name

    def test_auto_select_sampler_multi_objective(self) -> None:
        """Auto sampler for multi-objective should be NSGA-II."""
        config = OptimizationConfig(n_trials=50)
        strategy = AutoOptunaStrategy(config, is_multi_objective=True)
        strategy.create_study()
        sampler = strategy._auto_select_sampler()
        assert "NSGA" in sampler.__class__.__name__

    def test_auto_select_pruner_short_timeout(self) -> None:
        """Auto pruner for short timeout should be MedianPruner."""
        config = OptimizationConfig(n_trials=50, timeout_seconds=300)
        strategy = AutoOptunaStrategy(config)
        strategy.create_study()
        pruner = strategy._auto_select_pruner()
        assert "Median" in pruner.__class__.__name__

    def test_auto_select_pruner_long_timeout(self) -> None:
        """Auto pruner for long timeout should be HyperbandPruner."""
        config = OptimizationConfig(n_trials=50, timeout_seconds=1800)
        strategy = AutoOptunaStrategy(config)
        strategy.create_study()
        pruner = strategy._auto_select_pruner()
        assert "Hyperband" in pruner.__class__.__name__


# ─────────────────────────── Mini Optimization Tests ─────────────────────────


class TestOptunaStrategyMiniOptimization:
    """Tests running minimal optimizations."""

    def test_run_single_trial(self) -> None:
        """Should complete a single trial optimization."""
        config = OptimizationConfig(n_trials=1, direction="minimize")
        strategy = OptunaStrategy(config)

        def objective(trial) -> float:
            """Execute objective.



            Args:

                trial: Input value used by this callable.



            Returns:

                Return value produced by the callable.

            """

            x = trial.suggest_float("x", -10, 10)
            return x**2

        result = strategy.run_optimization(objective, {})

        assert result.n_trials == 1
        assert result.best_params is not None
        assert "x" in result.best_params

    def test_run_few_trials_with_search_space(self) -> None:
        """Should complete optimization with search space."""
        config = OptimizationConfig(n_trials=3, direction="minimize")
        strategy = OptunaStrategy(config)

        search_space = {"x": (-5.0, 5.0)}

        def objective(trial) -> float:
            """Execute objective.



            Args:

                trial: Input value used by this callable.



            Returns:

                Return value produced by the callable.

            """

            params = strategy.suggest_params(trial, search_space)
            return params["x"] ** 2

        result = strategy.run_optimization(objective, search_space)

        assert result.n_trials == 3
        assert result.best_value >= 0

    def test_optimization_result_fields(self) -> None:
        """Optimization result should have all expected fields."""
        config = OptimizationConfig(n_trials=2, direction="maximize")
        strategy = OptunaStrategy(config)

        def objective(trial) -> float:
            """Execute objective.



            Args:

                trial: Input value used by this callable.



            Returns:

                Return value produced by the callable.

            """

            x = trial.suggest_float("x", 0, 1)
            return x

        result = strategy.run_optimization(objective, {})

        assert hasattr(result, "best_params")
        assert hasattr(result, "best_value")
        assert hasattr(result, "best_trial_number")
        assert hasattr(result, "n_trials")
        assert hasattr(result, "trials")
        assert hasattr(result, "study_name")
        assert hasattr(result, "optimization_time")
        assert hasattr(result, "framework")

    def test_trials_list_populated(self) -> None:
        """Trials list should be populated after optimization."""
        config = OptimizationConfig(n_trials=3)
        strategy = OptunaStrategy(config)

        def objective(trial) -> float:
            """Execute objective.



            Args:

                trial: Input value used by this callable.



            Returns:

                Return value produced by the callable.

            """

            return trial.suggest_float("x", 0, 1)

        result = strategy.run_optimization(objective, {})

        assert len(result.trials) == 3
        for trial in result.trials:
            assert isinstance(trial, TrialResult)
            assert trial.trial_number >= 0


# ─────────────────────────── Config Loading Tests ─────────────────────────────


class TestSamplerConfigLoading:
    """Tests for sampler config loading."""

    def test_load_sampler_config_returns_dict(self) -> None:
        """_load_sampler_config should return dict."""
        result = _load_sampler_config()
        assert isinstance(result, dict)

    def test_load_sampler_config_cached(self) -> None:
        """_load_sampler_config should return consistent results."""
        result1 = _load_sampler_config()
        result2 = _load_sampler_config()
        assert result1 == result2

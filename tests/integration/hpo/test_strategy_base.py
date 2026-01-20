"""Tests for HPO strategy base classes and configuration."""

from pff.infrastructure.hpo.strategies.base import (
    OptimizationConfig,
    OptimizationResult,
    TrialResult,
)


class TestOptimizationConfig:
    """Tests for OptimizationConfig dataclass."""

    def test_optimization_config_defaults(self):
        """Verify default values are set correctly."""
        config = OptimizationConfig()
        assert config.n_trials == 100
        assert config.random_state == 42
        assert config.enable_pruning is True
        assert config.show_progress_bar is True
        assert config.direction == "maximize"
        assert config.pruner_type == "hyperband"

    def test_optimization_config_custom_values(self):
        """Verify custom values are accepted."""
        config = OptimizationConfig(
            n_trials=50,
            timeout_seconds=3600,
            random_state=123,
            enable_pruning=False,
            direction="minimize",
        )
        assert config.n_trials == 50
        assert config.timeout_seconds == 3600
        assert config.random_state == 123
        assert config.enable_pruning is False
        assert config.direction == "minimize"

    def test_optimization_config_storage_url(self):
        """Verify storage URL can be set."""
        config = OptimizationConfig(
            storage_url="sqlite:///optuna.db",
            study_name="test_study",
        )
        assert config.storage_url == "sqlite:///optuna.db"
        assert config.study_name == "test_study"

    def test_optimization_config_wilcoxon_params(self):
        """Verify Wilcoxon pruner parameters."""
        config = OptimizationConfig(
            pruner_type="wilcoxon",
            wilcoxon_p_threshold=0.05,
            wilcoxon_n_startup_steps=5,
        )
        assert config.pruner_type == "wilcoxon"
        assert config.wilcoxon_p_threshold == 0.05
        assert config.wilcoxon_n_startup_steps == 5

    def test_optimization_config_n_jobs(self):
        """Verify n_jobs parameter."""
        config = OptimizationConfig(n_jobs=4)
        assert config.n_jobs == 4


class TestTrialResult:
    """Tests for TrialResult dataclass."""

    def test_trial_result_minimal(self):
        """Verify minimal TrialResult creation."""
        result = TrialResult(
            params={"lr": 0.01},
            value=0.95,
            trial_number=0,
            state="COMPLETE",
        )
        assert result.params == {"lr": 0.01}
        assert result.value == 0.95
        assert result.trial_number == 0
        assert result.state == "COMPLETE"

    def test_trial_result_with_optionals(self):
        """Verify optional fields in TrialResult."""
        result = TrialResult(
            params={"lr": 0.01, "batch_size": 32},
            value=0.92,
            trial_number=5,
            state="COMPLETE",
            intermediate_values={1: 0.5, 2: 0.7, 3: 0.85},
            user_attrs={"model_path": "/path/to/model"},
        )
        assert result.intermediate_values == {1: 0.5, 2: 0.7, 3: 0.85}
        assert result.user_attrs == {"model_path": "/path/to/model"}

    def test_trial_result_pruned_state(self):
        """Verify pruned trial state."""
        result = TrialResult(
            params={"lr": 0.1},
            value=None,
            trial_number=3,
            state="PRUNED",
        )
        assert result.value is None
        assert result.state == "PRUNED"

    def test_trial_result_failed_state(self):
        """Verify failed trial state."""
        result = TrialResult(
            params={"lr": 0.001},
            value=None,
            trial_number=7,
            state="FAIL",
        )
        assert result.state == "FAIL"


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_optimization_result_creation(self):
        """Verify OptimizationResult creation."""
        trials = [
            TrialResult(params={"lr": 0.01}, value=0.9, trial_number=0, state="COMPLETE"),
            TrialResult(params={"lr": 0.02}, value=0.95, trial_number=1, state="COMPLETE"),
        ]
        result = OptimizationResult(
            best_params={"lr": 0.02},
            best_value=0.95,
            best_trial_number=1,
            n_trials=2,
            trials=trials,
            study_name="test_study",
            optimization_time=120.5,
            framework="optuna",
        )
        assert result.best_params == {"lr": 0.02}
        assert result.best_value == 0.95
        assert result.best_trial_number == 1
        assert result.n_trials == 2
        assert len(result.trials) == 2
        assert result.study_name == "test_study"
        assert result.optimization_time == 120.5
        assert result.framework == "optuna"

    def test_optimization_result_empty_trials(self):
        """Verify OptimizationResult with empty trials."""
        result = OptimizationResult(
            best_params={},
            best_value=0.0,
            best_trial_number=0,
            n_trials=0,
            trials=[],
            study_name="empty_study",
            optimization_time=0.0,
            framework="optuna",
        )
        assert result.n_trials == 0
        assert len(result.trials) == 0

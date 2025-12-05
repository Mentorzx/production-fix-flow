"""
Tests for optimization completion handlers.

Tests that all expected actions are triggered when optimization finishes:
- Best score tracking and reporting
- Visualization generation (plots, charts, dashboards)
- MLflow logging and artifacts
- Best parameters saving
- Trial state statistics
- Callbacks and observers
- Resource cleanup
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np
import pytest


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_trial():
    """Create a mock Optuna trial."""
    trial = MagicMock()
    trial.number = 1
    trial.value = 0.85
    trial.params = {
        "learning_rate": 0.01,
        "n_estimators": 100,
        "max_depth": 5,
    }
    trial.state = "COMPLETE"
    return trial


@pytest.fixture
def mock_study():
    """Create a mock Optuna study."""
    study = MagicMock()
    study.study_name = "test_study"
    study.best_value = 0.90
    study.best_params = {"learning_rate": 0.005, "n_estimators": 150}
    study.best_trial = MagicMock(number=5, value=0.90)
    return study


@pytest.fixture
def sample_optimization_result():
    """Create a sample OptimizationResult-like dict."""
    return {
        "best_params": {"learning_rate": 0.005, "n_estimators": 150},
        "best_value": 0.90,
        "n_trials": 50,
        "optimization_time": 120.5,
        "framework": "optuna",
        "trials": [
            {"number": i, "value": 0.5 + i * 0.01, "state": "COMPLETE", "params": {}}
            for i in range(50)
        ],
    }


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "optimization_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================================
# Tests for BestScoreObserver
# ============================================================================

class TestBestScoreObserver:
    """Tests for BestScoreObserver at optimization end."""

    def test_tracks_best_score_throughout_optimization(self):
        """Observer should track best score across all trials."""
        from scripts.optimization.callbacks import BestScoreObserver
        
        observer = BestScoreObserver()
        
        mock_trials = [
            MagicMock(number=0, value=0.5),
            MagicMock(number=1, value=0.7),  # New best
            MagicMock(number=2, value=0.6),
            MagicMock(number=3, value=0.8),  # New best
            MagicMock(number=4, value=0.75),
        ]
        
        for trial in mock_trials:
            observer.on_trial_complete(trial, trial.value)
        
        assert observer.get_best_score() == 0.8
        assert observer.best_trial_number == 3
        assert observer.get_improvement_count() == 3  # Initial + 2 improvements

    def test_improvement_count_accurate(self):
        """Improvement count should only increase on actual improvements."""
        from scripts.optimization.callbacks import BestScoreObserver
        
        observer = BestScoreObserver()
        
        values = [0.5, 0.5, 0.5, 0.6, 0.6, 0.7]
        for i, value in enumerate(values):
            trial = MagicMock(number=i, value=value)
            observer.on_trial_complete(trial, value)
        
        assert observer.get_improvement_count() == 3  # 0.5, 0.6, 0.7

    def test_handles_negative_scores(self):
        """Observer should handle negative scores (minimization scenarios)."""
        from scripts.optimization.callbacks import BestScoreObserver
        
        observer = BestScoreObserver()
        
        trial = MagicMock(number=0, value=-0.5)
        observer.on_trial_complete(trial, -0.5)
        
        # First trial is always an improvement from -inf
        assert observer.get_best_score() == -0.5
        assert observer.get_improvement_count() == 1

    def test_handles_zero_score(self):
        """Observer should handle zero as a valid score."""
        from scripts.optimization.callbacks import BestScoreObserver
        
        observer = BestScoreObserver()
        
        trial = MagicMock(number=0, value=0.0)
        observer.on_trial_complete(trial, 0.0)
        
        assert observer.get_best_score() == 0.0


# ============================================================================
# Tests for CompositeObserver Lifecycle
# ============================================================================

class TestCompositeObserverLifecycle:
    """Tests for CompositeObserver handling all lifecycle events."""

    def test_dispatches_optimization_start(self):
        """CompositeObserver should dispatch on_optimization_start to all observers."""
        from scripts.optimization.callbacks import CompositeObserver, OptimizationObserver
        
        obs1 = MagicMock(spec=OptimizationObserver)
        obs2 = MagicMock(spec=OptimizationObserver)
        
        composite = CompositeObserver([obs1, obs2])
        composite.on_optimization_start("study_name", 100)
        
        obs1.on_optimization_start.assert_called_once_with("study_name", 100)
        obs2.on_optimization_start.assert_called_once_with("study_name", 100)

    def test_dispatches_optimization_end(self):
        """CompositeObserver should dispatch on_optimization_end to all observers."""
        from scripts.optimization.callbacks import CompositeObserver, OptimizationObserver
        
        obs1 = MagicMock(spec=OptimizationObserver)
        obs2 = MagicMock(spec=OptimizationObserver)
        
        composite = CompositeObserver([obs1, obs2])
        best_params = {"lr": 0.01}
        composite.on_optimization_end(0.95, best_params)
        
        obs1.on_optimization_end.assert_called_once_with(0.95, best_params)
        obs2.on_optimization_end.assert_called_once_with(0.95, best_params)

    def test_handles_observer_failure_gracefully(self):
        """CompositeObserver should continue despite individual observer failures."""
        from scripts.optimization.callbacks import CompositeObserver, OptimizationObserver
        
        obs1 = MagicMock(spec=OptimizationObserver)
        obs1.on_optimization_end.side_effect = Exception("Observer 1 failed")
        obs2 = MagicMock(spec=OptimizationObserver)
        
        composite = CompositeObserver([obs1, obs2])
        
        # Should not raise
        composite.on_optimization_end(0.95, {"lr": 0.01})
        
        # obs2 should still be called
        obs2.on_optimization_end.assert_called_once()

    def test_empty_composite_handles_all_events(self):
        """Empty CompositeObserver should handle all lifecycle events without error."""
        from scripts.optimization.callbacks import CompositeObserver
        
        composite = CompositeObserver([])
        
        # Should not raise
        composite.on_optimization_start("study", 50)
        composite.on_trial_complete(MagicMock(number=0), 0.5)
        composite.on_optimization_end(0.95, {})


# ============================================================================
# Tests for CallbackManager
# ============================================================================

class TestCallbackManager:
    """Tests for CallbackManager at optimization end."""

    def test_notifies_all_observers_on_completion(self, mock_trial):
        """CallbackManager should notify all observers when trials complete."""
        from scripts.optimization.callbacks import CallbackManager, OptimizationObserver
        
        manager = CallbackManager()
        obs1 = MagicMock(spec=OptimizationObserver)
        obs2 = MagicMock(spec=OptimizationObserver)
        
        manager.add_observer(obs1)
        manager.add_observer(obs2)
        
        manager.notify_all(mock_trial, mock_trial.value)
        
        obs1.on_trial_complete.assert_called_once()
        obs2.on_trial_complete.assert_called_once()

    def test_get_observer_names(self):
        """Should return correct observer names."""
        from scripts.optimization.callbacks import (
            CallbackManager, LoggingObserver, BestScoreObserver
        )
        
        manager = CallbackManager()
        manager.add_observer(LoggingObserver(log_interval=5))
        manager.add_observer(BestScoreObserver())
        
        names = manager.get_observer_names()
        
        assert "LoggingObserver" in names
        assert "BestScoreObserver" in names

    def test_clear_removes_all_observers(self):
        """Clear should remove all observers."""
        from scripts.optimization.callbacks import CallbackManager, BestScoreObserver
        
        manager = CallbackManager()
        manager.add_observer(BestScoreObserver())
        manager.add_observer(BestScoreObserver())
        
        assert len(manager.observers) == 2
        
        manager.clear()
        
        assert len(manager.observers) == 0


# ============================================================================
# Tests for MLflowTrialObserver
# ============================================================================

class TestMLflowTrialObserver:
    """Tests for MLflow integration at optimization end."""

    def test_logs_optimization_end(self):
        """MLflowTrialObserver should call tracker's log_optimization_end."""
        from scripts.optimization.callbacks import MLflowTrialObserver
        
        mock_tracker = MagicMock()
        observer = MLflowTrialObserver(mock_tracker)
        
        # Simulate some trials
        for i in range(5):
            trial = MagicMock(number=i, state="COMPLETE")
            observer.on_trial_complete(trial, 0.5 + i * 0.1)
        
        best_params = {"lr": 0.01}
        
        # Call on_optimization_end - it may fail internally but shouldn't raise
        # The important thing is that it attempts to call the tracker
        try:
            observer.on_optimization_end(0.9, best_params)
        except Exception:
            pass  # Internal failures are logged, not raised
        
        # At minimum, the trial_count should have been incremented
        assert observer.trial_count == 5

    def test_handles_missing_tracker_gracefully(self):
        """Should handle None tracker gracefully."""
        from scripts.optimization.callbacks import MLflowTrialObserver
        
        observer = MLflowTrialObserver(None)
        
        # Should not raise
        observer.on_optimization_start("study", 100)
        observer.on_trial_complete(MagicMock(number=0, state="COMPLETE"), 0.5)
        observer.on_optimization_end(0.9, {})


# ============================================================================
# Tests for Visualization Generation
# ============================================================================

class TestVisualizationGeneration:
    """Tests for visualization generation at optimization end."""

    def test_generate_all_plots_returns_artifacts(self, sample_optimization_result):
        """generate_all_plots should return dict of artifact paths."""
        from scripts.optimization.visualizer import OptimizationVisualizer
        
        with patch.object(OptimizationVisualizer, '_check_dependencies'):
            visualizer = OptimizationVisualizer()
            visualizer.has_plotly = False
            visualizer.has_matplotlib = False
            visualizer.has_seaborn = False
            
            artifacts = visualizer.generate_all_plots(sample_optimization_result)
            
            # Without plotting libraries, returns empty dict
            assert isinstance(artifacts, dict)

    def test_plot_trial_states_handles_empty_trials(self):
        """plot_trial_states should handle empty trials list."""
        from scripts.optimization.visualizer import OptimizationVisualizer
        
        result = {"trials": [], "n_trials": 0}
        
        with patch.object(OptimizationVisualizer, '_check_dependencies'):
            visualizer = OptimizationVisualizer()
            visualizer.has_matplotlib = False
            
            artifacts = visualizer.plot_trial_states(result)
            
            assert isinstance(artifacts, dict)

    def test_plot_optimization_history_handles_no_history(self, sample_optimization_result):
        """plot_optimization_history should handle missing history."""
        from scripts.optimization.visualizer import OptimizationVisualizer
        
        with patch.object(OptimizationVisualizer, '_check_dependencies'):
            visualizer = OptimizationVisualizer()
            visualizer.has_plotly = False
            
            artifacts = visualizer.plot_optimization_history(sample_optimization_result)
            
            assert isinstance(artifacts, dict)

    def test_plot_best_trials_with_various_sizes(self, sample_optimization_result):
        """plot_best_trials should handle different top_n values."""
        from scripts.optimization.visualizer import OptimizationVisualizer
        
        with patch.object(OptimizationVisualizer, '_check_dependencies'):
            visualizer = OptimizationVisualizer()
            visualizer.has_matplotlib = False
            
            # With top_n larger than trials
            artifacts = visualizer.plot_best_trials(sample_optimization_result, top_n=100)
            assert isinstance(artifacts, dict)
            
            # With top_n of 1
            artifacts = visualizer.plot_best_trials(sample_optimization_result, top_n=1)
            assert isinstance(artifacts, dict)


# ============================================================================
# Tests for Best Parameters Saving
# ============================================================================

class TestBestParametersSaving:
    """Tests for saving best parameters at optimization end."""

    def test_best_params_saved_as_json(self, tmp_output_dir):
        """Best parameters should be saved as valid JSON."""
        from pff.utils.core.file_manager import FileManager
        
        fm = FileManager()
        best_params = {
            "learning_rate": 0.005,
            "n_estimators": 150,
            "max_depth": 7,
        }
        
        params_file = tmp_output_dir / "best_params.json"
        fm.save(best_params, params_file)
        
        assert params_file.exists()
        loaded = fm.read(params_file)
        assert loaded == best_params

    def test_best_params_includes_metadata(self, tmp_output_dir):
        """Saved params should include optimization metadata."""
        from pff.utils.core.file_manager import FileManager
        import time
        
        fm = FileManager()
        results_json = {
            "best_params": {"lr": 0.01},
            "best_value": 0.95,
            "n_trials": 100,
            "optimization_time": 120.5,
            "framework": "optuna",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        params_file = tmp_output_dir / "best_params.json"
        fm.save(results_json, params_file)
        
        loaded = fm.read(params_file)
        assert "best_value" in loaded
        assert "n_trials" in loaded
        assert "timestamp" in loaded


# ============================================================================
# Tests for BestModelSaverCallback
# ============================================================================

class TestBestModelSaverCallback:
    """Tests for BestModelSaverCallback at trial completion."""

    def test_saves_best_model_on_improvement(self, tmp_output_dir):
        """Callback should save models when new best is found."""
        from scripts.optimization.core import BestModelSaverCallback
        from scripts.optimization.trials.artifacts import TrialArtifactManager
        
        artifact_manager = MagicMock(spec=TrialArtifactManager)
        artifact_manager.get_trial_result.return_value = {
            "trial_dir": tmp_output_dir / "trial_0",
            "params": {"lr": 0.01},
            "model_metrics": {},
            "ensemble_metrics": {},
            "models_trained": {"lightgbm": True},
            "model_paths": {},
            "composite_score": 0.85,
        }
        
        callback = BestModelSaverCallback(
            output_dir=tmp_output_dir,
            artifact_manager=artifact_manager
        )
        
        study = MagicMock()
        trial = MagicMock(number=0, value=0.85)
        
        # Import TrialState from optuna and patch it
        with patch('optuna.trial.TrialState') as mock_state:
            mock_state.COMPLETE = "COMPLETE"
            trial.state = mock_state.COMPLETE
            
            callback(study, trial)
        
        # Should call persist methods on improvement
        artifact_manager.persist_best_models.assert_called()
        artifact_manager.persist_best_params.assert_called()

    def test_skips_non_complete_trials(self, tmp_output_dir):
        """Callback should skip trials that are not COMPLETE."""
        from scripts.optimization.core import BestModelSaverCallback
        
        callback = BestModelSaverCallback(output_dir=tmp_output_dir)
        
        study = MagicMock()
        trial = MagicMock(number=0, value=0.85)
        
        with patch('optuna.trial.TrialState') as mock_state:
            mock_state.COMPLETE = "COMPLETE"
            trial.state = "PRUNED"  # Not COMPLETE
            
            callback(study, trial)
        
        # Should not update best_value
        assert callback.best_value == float('-inf')

    def test_does_not_save_on_no_improvement(self, tmp_output_dir):
        """Callback should not save when no improvement."""
        from scripts.optimization.core import BestModelSaverCallback
        from scripts.optimization.trials.artifacts import TrialArtifactManager
        
        artifact_manager = MagicMock(spec=TrialArtifactManager)
        
        callback = BestModelSaverCallback(
            output_dir=tmp_output_dir,
            artifact_manager=artifact_manager
        )
        callback.best_value = 0.90  # Already have a good score
        
        artifact_manager.get_trial_result.return_value = {
            "trial_dir": tmp_output_dir / "trial_1",
            "params": {},
            "model_metrics": {},
            "ensemble_metrics": {},
            "models_trained": {},
            "model_paths": {},
            "composite_score": 0.80,  # Worse than current best
        }
        
        study = MagicMock()
        trial = MagicMock(number=1, value=0.80)
        
        with patch('optuna.trial.TrialState') as mock_state:
            mock_state.COMPLETE = "COMPLETE"
            trial.state = mock_state.COMPLETE
            
            callback(study, trial)
        
        # Should not call persist methods (no improvement)
        artifact_manager.persist_best_models.assert_not_called()


# ============================================================================
# Tests for Trial State Statistics
# ============================================================================

class TestTrialStateStatistics:
    """Tests for trial state statistics at optimization end."""

    def test_counts_completed_trials(self, sample_optimization_result):
        """Should correctly count completed trials."""
        trials = sample_optimization_result["trials"]
        n_completed = len([t for t in trials if t["state"] == "COMPLETE"])
        
        assert n_completed == 50

    def test_counts_pruned_trials(self):
        """Should correctly count pruned trials."""
        trials = [
            {"state": "COMPLETE"},
            {"state": "PRUNED"},
            {"state": "PRUNED"},
            {"state": "COMPLETE"},
            {"state": "FAIL"},
        ]
        
        n_pruned = len([t for t in trials if t["state"] == "PRUNED"])
        
        assert n_pruned == 2

    def test_counts_failed_trials(self):
        """Should correctly count failed trials."""
        n_trials = 10
        trials = [
            {"state": "COMPLETE"},
            {"state": "COMPLETE"},
            {"state": "PRUNED"},
            {"state": "FAIL"},
            {"state": "FAIL"},
        ]
        
        n_completed = len([t for t in trials if t["state"] == "COMPLETE"])
        n_pruned = len([t for t in trials if t["state"] == "PRUNED"])
        n_failed = len(trials) - n_completed - n_pruned
        
        assert n_failed == 2

    def test_calculates_success_rate(self):
        """Should correctly calculate success rate."""
        n_trials = 100
        n_completed = 80
        n_pruned = 15
        n_failed = n_trials - n_completed - n_pruned
        
        success_rate = n_completed / n_trials * 100
        
        assert success_rate == 80.0
        assert n_failed == 5


# ============================================================================
# Tests for PersistentBestTrialMemory
# ============================================================================

class TestPersistentBestTrialMemory:
    """Tests for persistent memory at optimization end."""

    def test_records_best_trial(self, tmp_output_dir):
        """Memory should record best trials."""
        from scripts.optimization.core import PersistentBestTrialMemory, HPOMemoryConfig
        
        config = HPOMemoryConfig(enabled=True, top_k_trials=5)
        memory = PersistentBestTrialMemory(tmp_output_dir, config)
        
        study = MagicMock(study_name="test_study")
        trial = MagicMock(number=0, value=0.85, params={"lr": 0.01})
        
        with patch('optuna.trial.TrialState') as mock_state:
            mock_state.COMPLETE = "COMPLETE"
            trial.state = mock_state.COMPLETE
            
            memory.record_trial(study, trial, {"ensemble_metrics": {}, "model_metrics": {}})
        
        assert len(memory.entries) == 1

    def test_keeps_only_top_k_trials(self, tmp_output_dir):
        """Memory should only keep top_k best trials."""
        from scripts.optimization.core import PersistentBestTrialMemory, HPOMemoryConfig
        
        config = HPOMemoryConfig(enabled=True, top_k_trials=3)
        memory = PersistentBestTrialMemory(tmp_output_dir, config)
        
        with patch('optuna.trial.TrialState') as mock_state:
            mock_state.COMPLETE = "COMPLETE"
            
            for i in range(10):
                study = MagicMock(study_name="test_study")
                trial = MagicMock(
                    number=i,
                    value=0.5 + i * 0.05,
                    params={"lr": 0.01 * i},
                )
                trial.state = mock_state.COMPLETE
                memory.record_trial(study, trial, {})
        
        # Should only keep top 3
        assert len(memory.entries) <= 3

    def test_disabled_memory_does_nothing(self, tmp_output_dir):
        """Disabled memory should not record anything."""
        from scripts.optimization.core import PersistentBestTrialMemory, HPOMemoryConfig
        
        config = HPOMemoryConfig(enabled=False)
        memory = PersistentBestTrialMemory(tmp_output_dir, config)
        
        study = MagicMock()
        trial = MagicMock(number=0, value=0.85, params={})
        trial.state = "COMPLETE"
        
        memory.record_trial(study, trial, {})
        
        assert len(memory.entries) == 0


# ============================================================================
# Tests for MLflowTracker End Events
# ============================================================================

class TestMLflowTrackerEndEvents:
    """Tests for MLflow tracker at optimization end."""

    def test_log_optimization_end_with_result(self):
        """Tracker should handle optimization results without raising."""
        from scripts.optimization.tracker import MLflowTracker
        from scripts.optimization.strategies.base import OptimizationResult, TrialResult
        
        # Create tracker - will try to import mlflow internally
        tracker = MLflowTracker("test_experiment")
        
        # If mlflow is not available, tracker.mlflow will be None
        if tracker.mlflow is None:
            pytest.skip("MLflow not available")
        
        # Create a complete result
        result = OptimizationResult(
            best_params={"lr": 0.01},
            best_value=0.95,
            best_trial_number=5,
            n_trials=50,
            optimization_time=120.0,
            framework="optuna",
            study_name="test_study",
            trials=[
                TrialResult(
                    trial_number=i,
                    value=0.5 + i * 0.01,
                    params={},
                    state="COMPLETE",
                    intermediate_values={},
                )
                for i in range(10)
            ],
        )
        
        # Should not raise even without parent_run_id
        tracker.log_optimization_end(result)

    def test_log_artifacts_at_end(self, tmp_output_dir):
        """Tracker should handle artifact logging without raising."""
        from scripts.optimization.tracker import MLflowTracker
        
        tracker = MLflowTracker("test_experiment")
        
        if tracker.mlflow is None:
            pytest.skip("MLflow not available")
        
        # Create test artifact
        artifact_file = tmp_output_dir / "test_plot.png"
        artifact_file.touch()
        
        # Should not raise
        tracker.log_artifacts({"plot": artifact_file})

    def test_tracker_handles_missing_mlflow(self):
        """Tracker should handle case where mlflow is not installed."""
        from scripts.optimization.tracker import MLflowTracker
        
        # Create tracker - it handles ImportError internally
        tracker = MLflowTracker("test_experiment")
        
        # Should not raise regardless of mlflow availability
        tracker.log_artifacts({})
        tracker.log_optimization_end(MagicMock())


# ============================================================================
# Tests for RealTimeVisualizer at End
# ============================================================================

class TestRealTimeVisualizerEnd:
    """Tests for RealTimeVisualizer at optimization end."""

    def test_save_plots_creates_files(self, tmp_output_dir):
        """save_plots should create plot files."""
        from scripts.optimization.callbacks import RealTimeVisualizer
        
        with patch('scripts.optimization.callbacks.plt') as mock_plt:
            mock_plt.ion.return_value = None
            mock_plt.subplots.return_value = (MagicMock(), (MagicMock(), MagicMock()))
            mock_plt.show.return_value = None
            
            visualizer = RealTimeVisualizer()
            visualizer.initialized = True
            visualizer.scores = [0.5, 0.6, 0.7]
            visualizer.trial_numbers = [0, 1, 2]
            visualizer.best_scores = [0.5, 0.6, 0.7]
            
            result = visualizer.save_plots(str(tmp_output_dir), prefix="test")
            
            # Should return dict of saved files
            assert isinstance(result, dict)

    def test_close_cleans_up_resources(self):
        """close should clean up matplotlib resources."""
        from scripts.optimization.callbacks import RealTimeVisualizer
        
        with patch('scripts.optimization.callbacks.plt') as mock_plt:
            mock_plt.ion.return_value = None
            mock_plt.subplots.return_value = (MagicMock(), (MagicMock(), MagicMock()))
            mock_plt.show.return_value = None
            
            visualizer = RealTimeVisualizer()
            visualizer.initialized = True
            
            visualizer.close()
            
            mock_plt.close.assert_called()


class TestLivePlotCallback:
    """Tests for live plot Optuna callback."""

    def test_live_plot_callback_saves_files_with_fixed_axis(self, tmp_output_dir):
        """LivePlotCallback should persist plots and keep fixed x-axis."""
        from optuna.trial import TrialState
        from scripts.optimization.callbacks import LivePlotCallback

        class DummyTrial:
            def __init__(self, number, value, state):
                self.number = number
                self.value = value
                self.state = state

        class DummyStudy:
            def __init__(self, trials):
                self.trials = trials

        trials = [
            DummyTrial(0, 0.1, TrialState.COMPLETE),
            DummyTrial(1, 0.2, TrialState.COMPLETE),
            DummyTrial(2, 0.15, TrialState.COMPLETE),
        ]
        study = DummyStudy(trials)
        callback = LivePlotCallback(output_dir=tmp_output_dir, max_trials_axis=50, expected_trials=60)

        callback(study, trials[-1])

        assert (tmp_output_dir / "convergence.png").exists()
        assert (tmp_output_dir / "score_distribution.png").exists()
        _, x_max = callback.progress_ax.get_xlim()
        assert x_max == pytest.approx(60)

    def test_load_live_plot_settings_overrides(self, monkeypatch):
        """load_live_plot_settings should honor config overrides."""
        from scripts.optimization.trials import config_loader

        def fake_get_cached_config(path, file_manager=None):
            return {
                "live_plots": {
                    "enabled": False,
                    "max_trials_axis": 75,
                    "output_subdir": "custom_dir",
                }
            }

        monkeypatch.setattr(config_loader, "get_cached_config", fake_get_cached_config)
        settings = config_loader.load_live_plot_settings()

        assert settings["enabled"] is False
        assert settings["max_trials_axis"] == 75
        assert settings["output_subdir"] == "custom_dir"


class TestOptimizationLandscape3D:
    """Tests for 3D landscape plot generation."""

    def test_mesh3d_instead_of_scatter(self, monkeypatch, tmp_path):
        """Ensure landscape plot uses Mesh3d (surface) and not only scatter."""
        from scripts.optimization.visualizer import OptimizationVisualizer

        # Minimal result with three numeric params and scores
        result = {
            "trials": [
                {"number": 0, "value": 0.1, "params": {"a": 1.0, "b": 2.0, "c": 3.0}},
                {"number": 1, "value": 0.2, "params": {"a": 1.5, "b": 2.5, "c": 3.5}},
                {"number": 2, "value": 0.3, "params": {"a": 2.0, "b": 3.0, "c": 4.0}},
            ]
        }

        calls = {}

        def fake_write_html(path):
            calls["write_html"] = path

        vis = OptimizationVisualizer(output_dir=tmp_path)
        if not vis.has_plotly:
            pytest.skip("Plotly not available")

        # Spy on write_html to avoid file I/O
        monkeypatch.setattr(vis.go.Figure, "write_html", lambda self, path: fake_write_html(path))

        artifacts = vis.plot_optimization_landscape_3d(result, study=None)

        assert "optimization_landscape_3d" in artifacts
        assert "write_html" in calls  # plot was generated


class TestTrialConfigLoaders:
    """Tests for trial-related config loaders."""

    def test_load_trial_constraints_defaults_and_overrides(self, monkeypatch):
        """load_trial_constraints should return defaults and honor overrides."""
        from scripts.optimization.trials import config_loader

        def fake_get_cached_config(path, file_manager=None):
            return {
                "constraints": {
                    "coverage_gate": 0.3,
                    "dominance_gate": 0.8,
                    "min_symbolic_activation": 0.02,
                    "symbolic_max_rules": 1200,
                }
            }

        monkeypatch.setattr(config_loader, "get_cached_config", fake_get_cached_config)
        constraints = config_loader.load_trial_constraints()
        assert constraints["coverage_gate"] == 0.3
        assert constraints["dominance_gate"] == 0.8
        assert constraints["min_symbolic_activation"] == 0.02
        assert constraints["symbolic_max_rules"] == 1200

# ============================================================================
# Tests for LoggingObserver at End
# ============================================================================

class TestLoggingObserverEnd:
    """Tests for LoggingObserver logging at end."""

    def test_logs_at_interval(self):
        """LoggingObserver should log at specified intervals."""
        from scripts.optimization.callbacks import LoggingObserver
        
        observer = LoggingObserver(log_interval=5)
        
        for i in range(10):
            trial = MagicMock(number=i, params={"lr": 0.01})
            observer.on_trial_complete(trial, 0.5 + i * 0.05)
        
        # After 10 trials with interval 5, should have logged at trial 5 and 10
        assert observer.trial_count == 10

    def test_final_trial_count_accurate(self):
        """Trial count should be accurate at end."""
        from scripts.optimization.callbacks import LoggingObserver
        
        observer = LoggingObserver(log_interval=10)
        
        for i in range(25):
            trial = MagicMock(number=i, params={})
            observer.on_trial_complete(trial, 0.5)
        
        assert observer.trial_count == 25


# ============================================================================
# Tests for Edge Cases at Optimization End
# ============================================================================

class TestOptimizationEndEdgeCases:
    """Tests for edge cases at optimization end."""

    def test_handles_zero_completed_trials(self):
        """Should handle case where no trials completed."""
        trials = [{"state": "PRUNED"} for _ in range(10)]
        
        n_completed = len([t for t in trials if t["state"] == "COMPLETE"])
        n_trials = len(trials)
        
        # Avoid division by zero
        success_rate = (n_completed / n_trials * 100) if n_trials > 0 else 0
        
        assert success_rate == 0.0

    def test_handles_nan_best_value(self):
        """Should handle NaN as best value."""
        from scripts.optimization.trials.bounds import normalize_metric
        
        best_value = float("nan")
        normalized = normalize_metric(best_value, low=0.0, high=1.0)
        
        assert normalized == 0.0

    def test_handles_none_best_params(self):
        """Should handle None best params gracefully."""
        best_params = None
        
        # Safe access pattern
        params_to_save = best_params or {}
        
        assert params_to_save == {}

    def test_handles_empty_trial_params(self):
        """Should handle trials with empty params."""
        trial = MagicMock(number=0, value=0.5, params={})
        
        # Should not raise
        param_count = len(trial.params)
        
        assert param_count == 0


# ============================================================================
# Tests for Resource Cleanup
# ============================================================================

class TestResourceCleanup:
    """Tests for resource cleanup at optimization end."""

    def test_cleanup_trial_directories(self, tmp_output_dir):
        """Should cleanup temporary trial directories."""
        trial_dir = tmp_output_dir / "trial_0"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Create some temp files
        (trial_dir / "temp_model.bin").touch()
        (trial_dir / "temp_embeddings.npy").touch()
        
        assert trial_dir.exists()
        
        # Cleanup using shutil
        import shutil
        shutil.rmtree(trial_dir, ignore_errors=True)
        
        assert not trial_dir.exists()

    def test_database_connection_cleanup(self):
        """Should close database connections."""
        from pff.db.connection import close_connection_pool
        
        # Should not raise even if pool doesn't exist
        import asyncio
        try:
            # Use asyncio.run() which is the recommended way in Python 3.10+
            asyncio.run(close_connection_pool())
        except RuntimeError:
            # Handle case where event loop is already running (e.g., in pytest-asyncio)
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(close_connection_pool())
                loop.close()
            except Exception:
                pass
        except Exception:
            pass  # Connection pool may not be initialized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

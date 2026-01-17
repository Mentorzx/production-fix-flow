"""Tests for BestModelSaverCallback and trial state handling.

Fast unit tests targeting:
- Callback invocation with different trial states
- Best value tracking
- Trial result retrieval fallback
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from unittest.mock import MagicMock

import pytest


class MockTrialState(Enum):
    """Mock Optuna TrialState."""

    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAIL = "FAIL"
    WAITING = "WAITING"


class MockBestModelSaverCallback:
    """Minimal mock of BestModelSaverCallback for fast testing."""

    def __init__(self) -> None:
        self.best_value = float("-inf")
        self.best_trial_number = -1
        self.trial_results: dict[int, dict[str, Any]] = {}
        self.saved_trials: list[int] = []
        self.cleanup_called: list[int] = []

    def record_result(self, trial_number: int, result: dict[str, Any]) -> None:
        """Mock artifact manager record."""
        self.trial_results[trial_number] = result

    def get_trial_result(self, trial: Any) -> dict[str, Any] | None:
        """Mock artifact manager get."""
        return self.trial_results.get(getattr(trial, "number", -1))

    def __call__(self, study: Any, trial: Any) -> None:
        """Callback invocation."""
        trial_state = getattr(trial, "state", None)

        # Skip non-complete trials
        if trial_state is not None and trial_state != MockTrialState.COMPLETE:
            return

        trial_result = self.get_trial_result(trial)
        if trial_result is None:
            return

        trial_value = getattr(trial, "value", float("-inf"))
        if trial_value is None:
            trial_value = float("-inf")

        is_best = trial_value > self.best_value

        if is_best:
            self.best_value = trial_value
            self.best_trial_number = getattr(trial, "number", -1)
            self.saved_trials.append(self.best_trial_number)

        # Cleanup
        trial_dir = trial_result.get("trial_dir")
        if trial_dir:
            self.cleanup_called.append(getattr(trial, "number", -1))


class TestBestModelSaverCallbackStates:
    """Test callback behavior with different trial states."""

    @pytest.fixture
    def callback(self) -> MockBestModelSaverCallback:
        return MockBestModelSaverCallback()

    def test_complete_trial_processed(self, callback: MockBestModelSaverCallback):
        """Complete trial should be processed."""
        callback.record_result(0, {"score": 0.8, "trial_dir": "/tmp/trial_0"})

        trial = MagicMock()
        trial.number = 0
        trial.value = 0.8
        trial.state = MockTrialState.COMPLETE

        callback(None, trial)

        assert callback.best_value == 0.8
        assert callback.best_trial_number == 0

    def test_pruned_trial_skipped(self, callback: MockBestModelSaverCallback):
        """Pruned trial should be skipped."""
        callback.record_result(0, {"score": 0.8})

        trial = MagicMock()
        trial.number = 0
        trial.value = 0.8
        trial.state = MockTrialState.PRUNED

        callback(None, trial)

        assert callback.best_value == float("-inf")
        assert callback.best_trial_number == -1

    def test_failed_trial_skipped(self, callback: MockBestModelSaverCallback):
        """Failed trial should be skipped."""
        callback.record_result(0, {"score": 0.8})

        trial = MagicMock()
        trial.number = 0
        trial.value = None
        trial.state = MockTrialState.FAIL

        callback(None, trial)

        assert callback.best_value == float("-inf")

    def test_running_trial_skipped(self, callback: MockBestModelSaverCallback):
        """Running trial should be skipped."""
        trial = MagicMock()
        trial.number = 0
        trial.state = MockTrialState.RUNNING

        callback(None, trial)

        assert callback.best_trial_number == -1


class TestBestValueTracking:
    """Test best value tracking logic."""

    @pytest.fixture
    def callback(self) -> MockBestModelSaverCallback:
        return MockBestModelSaverCallback()

    def test_first_trial_is_best(self, callback: MockBestModelSaverCallback):
        """First complete trial should be best."""
        callback.record_result(0, {"score": 0.5, "trial_dir": "/tmp/0"})

        trial = MagicMock()
        trial.number = 0
        trial.value = 0.5
        trial.state = MockTrialState.COMPLETE

        callback(None, trial)

        assert callback.best_value == 0.5
        assert 0 in callback.saved_trials

    def test_better_trial_updates_best(self, callback: MockBestModelSaverCallback):
        """Better trial should update best."""
        # First trial
        callback.record_result(0, {"score": 0.5, "trial_dir": "/tmp/0"})
        trial0 = MagicMock(number=0, value=0.5, state=MockTrialState.COMPLETE)
        callback(None, trial0)

        # Better trial
        callback.record_result(1, {"score": 0.7, "trial_dir": "/tmp/1"})
        trial1 = MagicMock(number=1, value=0.7, state=MockTrialState.COMPLETE)
        callback(None, trial1)

        assert callback.best_value == 0.7
        assert callback.best_trial_number == 1
        assert 0 in callback.saved_trials
        assert 1 in callback.saved_trials

    def test_worse_trial_does_not_update_best(
        self, callback: MockBestModelSaverCallback
    ):
        """Worse trial should not update best."""
        # First trial
        callback.record_result(0, {"score": 0.7, "trial_dir": "/tmp/0"})
        trial0 = MagicMock(number=0, value=0.7, state=MockTrialState.COMPLETE)
        callback(None, trial0)

        # Worse trial
        callback.record_result(1, {"score": 0.5, "trial_dir": "/tmp/1"})
        trial1 = MagicMock(number=1, value=0.5, state=MockTrialState.COMPLETE)
        callback(None, trial1)

        assert callback.best_value == 0.7
        assert callback.best_trial_number == 0
        assert 1 not in callback.saved_trials

    def test_equal_value_does_not_update(self, callback: MockBestModelSaverCallback):
        """Equal value should not update best (strict >)."""
        callback.record_result(0, {"score": 0.7, "trial_dir": "/tmp/0"})
        trial0 = MagicMock(number=0, value=0.7, state=MockTrialState.COMPLETE)
        callback(None, trial0)

        callback.record_result(1, {"score": 0.7, "trial_dir": "/tmp/1"})
        trial1 = MagicMock(number=1, value=0.7, state=MockTrialState.COMPLETE)
        callback(None, trial1)

        assert callback.best_trial_number == 0  # First one kept


class TestTrialResultRetrieval:
    """Test trial result retrieval edge cases."""

    @pytest.fixture
    def callback(self) -> MockBestModelSaverCallback:
        return MockBestModelSaverCallback()

    def test_missing_result_skipped(self, callback: MockBestModelSaverCallback):
        """Trial without recorded result should be skipped."""
        trial = MagicMock()
        trial.number = 999  # Not recorded
        trial.value = 0.9
        trial.state = MockTrialState.COMPLETE

        callback(None, trial)

        assert callback.best_value == float("-inf")

    def test_none_value_treated_as_neg_inf(self, callback: MockBestModelSaverCallback):
        """None value should be treated as -inf."""
        callback.record_result(0, {"score": 0.5, "trial_dir": "/tmp/0"})

        trial = MagicMock()
        trial.number = 0
        trial.value = None
        trial.state = MockTrialState.COMPLETE

        callback(None, trial)

        # None becomes -inf, which is not > -inf, so no update
        assert callback.best_value == float("-inf")


class TestCleanupBehavior:
    """Test trial directory cleanup."""

    @pytest.fixture
    def callback(self) -> MockBestModelSaverCallback:
        return MockBestModelSaverCallback()

    def test_cleanup_called_when_trial_dir_present(
        self, callback: MockBestModelSaverCallback
    ):
        """Cleanup should be called when trial_dir is present."""
        callback.record_result(0, {"score": 0.5, "trial_dir": "/tmp/trial_0"})

        trial = MagicMock(number=0, value=0.5, state=MockTrialState.COMPLETE)
        callback(None, trial)

        assert 0 in callback.cleanup_called

    def test_cleanup_not_called_when_no_trial_dir(
        self, callback: MockBestModelSaverCallback
    ):
        """Cleanup should not be called when trial_dir is missing."""
        callback.record_result(0, {"score": 0.5})  # No trial_dir

        trial = MagicMock(number=0, value=0.5, state=MockTrialState.COMPLETE)
        callback(None, trial)

        assert 0 not in callback.cleanup_called

    def test_cleanup_called_for_non_best_trials(
        self, callback: MockBestModelSaverCallback
    ):
        """Cleanup should be called even for non-best trials."""
        # Best trial
        callback.record_result(0, {"score": 0.8, "trial_dir": "/tmp/0"})
        trial0 = MagicMock(number=0, value=0.8, state=MockTrialState.COMPLETE)
        callback(None, trial0)

        # Worse trial
        callback.record_result(1, {"score": 0.5, "trial_dir": "/tmp/1"})
        trial1 = MagicMock(number=1, value=0.5, state=MockTrialState.COMPLETE)
        callback(None, trial1)

        # Both should be cleaned up
        assert 0 in callback.cleanup_called
        assert 1 in callback.cleanup_called

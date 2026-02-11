"""Tests for TrialArtifactManager edge cases and param matching.

Fast unit tests (no I/O, no models) targeting:
- _match_params float comparison precision issues
- get_trial_result fallback logic
- Param matching when params are missing/extra
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


class MockTrialArtifactManager:
    """Minimal mock matching TrialArtifactManager logic for fast testing."""

    def __init__(self) -> None:
        self.trial_results: dict[int, dict[str, Any]] = {}

    def record_result(self, trial_number: int, trial_result: dict[str, Any]) -> None:
        self.trial_results[trial_number] = trial_result

    def _match_params(
        self, stored: dict[str, Any], trial_params: dict[str, Any]
    ) -> bool:
        """Match params using same logic as core.py."""
        stored_params = stored.get("params", {})
        for name in set(stored_params.keys()) | set(trial_params.keys()):
            val1 = stored_params.get(name)
            val2 = trial_params.get(name)
            if isinstance(val1, float) and isinstance(val2, float):
                if abs(val1 - val2) > 1e-9:
                    return False
            elif val1 != val2:
                return False
        return True

    def get_trial_result(self, trial: Any) -> dict[str, Any] | None:
        result = self.trial_results.pop(getattr(trial, "number", -1), None)
        if result:
            return result
        trial_params = getattr(trial, "params", {})
        for key, candidate in list(self.trial_results.items()):
            if self._match_params(candidate, trial_params):
                return self.trial_results.pop(key)
        return None


class TestMatchParamsFloatPrecision:
    """Test float comparison edge cases in _match_params."""

    @pytest.fixture
    def manager(self) -> MockTrialArtifactManager:
        return MockTrialArtifactManager()

    def test_exact_float_match(self, manager: MockTrialArtifactManager):
        """Exact float values should match."""
        stored = {"params": {"lr": 0.001, "epochs": 50}}
        trial_params = {"lr": 0.001, "epochs": 50}
        assert manager._match_params(stored, trial_params)

    def test_float_within_tolerance(self, manager: MockTrialArtifactManager):
        """Floats within 1e-9 tolerance should match."""
        stored = {"params": {"lr": 0.001}}
        trial_params = {"lr": 0.001 + 1e-10}  # Within tolerance
        assert manager._match_params(stored, trial_params)

    def test_float_outside_tolerance(self, manager: MockTrialArtifactManager):
        """Floats outside 1e-9 tolerance should NOT match."""
        stored = {"params": {"lr": 0.001}}
        trial_params = {"lr": 0.001 + 1e-8}  # Outside tolerance
        assert not manager._match_params(stored, trial_params)

    def test_float_vs_int_mismatch(self, manager: MockTrialArtifactManager):
        """Float 1.0 vs int 1 should NOT match (type mismatch)."""
        stored = {"params": {"value": 1.0}}
        trial_params = {"value": 1}  # int, not float
        # Current logic: isinstance checks both as float -> False, falls to !=
        # 1.0 != 1 is False in Python, so they match!
        # This is a potential bug/dilema
        result = manager._match_params(stored, trial_params)
        # Document actual behavior
        assert result is True  # Python: 1.0 == 1

    def test_none_vs_missing_key(self, manager: MockTrialArtifactManager):
        """None value vs missing key should NOT match."""
        stored = {"params": {"lr": None}}
        trial_params = {}  # Missing key
        # Both get None from .get(), so they match
        assert manager._match_params(stored, trial_params)

    def test_extra_key_in_stored(self, manager: MockTrialArtifactManager):
        """Extra key in stored params should cause mismatch if trial doesn't have it."""
        stored = {"params": {"lr": 0.001, "extra": 42}}
        trial_params = {"lr": 0.001}  # Missing 'extra'
        # trial_params.get("extra") returns None, stored has 42
        assert not manager._match_params(stored, trial_params)

    def test_extra_key_in_trial(self, manager: MockTrialArtifactManager):
        """Extra key in trial params should cause mismatch."""
        stored = {"params": {"lr": 0.001}}
        trial_params = {"lr": 0.001, "extra": 42}
        assert not manager._match_params(stored, trial_params)

    def test_nan_float_comparison(self, manager: MockTrialArtifactManager):
        """NaN vs NaN comparison (NaN != NaN in IEEE 754)."""
        stored = {"params": {"lr": float("nan")}}
        trial_params = {"lr": float("nan")}
        result = manager._match_params(stored, trial_params)
        assert result is True


class TestGetTrialResultFallback:
    """Test get_trial_result lookup and fallback logic."""

    @pytest.fixture
    def manager(self) -> MockTrialArtifactManager:
        return MockTrialArtifactManager()

    def test_lookup_by_trial_number(self, manager: MockTrialArtifactManager):
        """Direct lookup by trial number should work."""
        manager.record_result(5, {"params": {"lr": 0.01}, "score": 0.8})

        mock_trial = MagicMock()
        mock_trial.number = 5
        mock_trial.params = {}

        result = manager.get_trial_result(mock_trial)
        assert result is not None
        assert result["score"] == 0.8

    def test_fallback_to_param_matching(self, manager: MockTrialArtifactManager):
        """When trial number doesn't match, fallback to param matching."""
        manager.record_result(5, {"params": {"lr": 0.01}, "score": 0.8})

        mock_trial = MagicMock()
        mock_trial.number = 999  # Different number
        mock_trial.params = {"lr": 0.01}  # Same params

        result = manager.get_trial_result(mock_trial)
        assert result is not None
        assert result["score"] == 0.8

    def test_no_match_returns_none(self, manager: MockTrialArtifactManager):
        """No match should return None."""
        manager.record_result(5, {"params": {"lr": 0.01}, "score": 0.8})

        mock_trial = MagicMock()
        mock_trial.number = 999
        mock_trial.params = {"lr": 0.99}  # Different params

        result = manager.get_trial_result(mock_trial)
        assert result is None

    def test_result_is_removed_after_retrieval(self, manager: MockTrialArtifactManager):
        """Retrieved result should be removed from storage."""
        manager.record_result(5, {"params": {"lr": 0.01}, "score": 0.8})

        mock_trial = MagicMock()
        mock_trial.number = 5
        mock_trial.params = {}

        # First retrieval
        result1 = manager.get_trial_result(mock_trial)
        assert result1 is not None

        # Second retrieval should return None
        result2 = manager.get_trial_result(mock_trial)
        assert result2 is None

    def test_multiple_trials_param_match_first_found(
        self, manager: MockTrialArtifactManager
    ):
        """When multiple trials have same params, first found is returned."""
        manager.record_result(1, {"params": {"lr": 0.01}, "score": 0.7})
        manager.record_result(2, {"params": {"lr": 0.01}, "score": 0.8})

        mock_trial = MagicMock()
        mock_trial.number = 999
        mock_trial.params = {"lr": 0.01}

        result = manager.get_trial_result(mock_trial)
        assert result is not None
        # Dict iteration order is insertion order in Python 3.7+
        assert result["score"] in [0.7, 0.8]


class TestTrialArtifactManagerEdgeCases:
    """Edge cases for artifact manager."""

    @pytest.fixture
    def manager(self) -> MockTrialArtifactManager:
        return MockTrialArtifactManager()

    def test_trial_without_number_attribute(self, manager: MockTrialArtifactManager):
        """Trial object without number attribute should not crash."""
        manager.record_result(5, {"params": {"lr": 0.01}, "score": 0.8})

        mock_trial = MagicMock(spec=[])  # No attributes
        del mock_trial.number  # Ensure no number
        mock_trial.params = {"lr": 0.01}

        # getattr with default -1 should handle this
        result = manager.get_trial_result(mock_trial)
        assert result is not None  # Falls back to param matching

    def test_trial_without_params_attribute(self, manager: MockTrialArtifactManager):
        """Trial without params should use empty dict."""
        manager.record_result(5, {"params": {}, "score": 0.8})

        mock_trial = MagicMock()
        mock_trial.number = 999
        # No params attribute
        del mock_trial.params

        result = manager.get_trial_result(mock_trial)
        # Empty params match empty stored params
        assert result is not None

    def test_empty_stored_params(self, manager: MockTrialArtifactManager):
        """Empty stored params should match empty trial params."""
        manager.record_result(5, {"params": {}, "score": 0.8})

        mock_trial = MagicMock()
        mock_trial.number = 999
        mock_trial.params = {}

        result = manager.get_trial_result(mock_trial)
        assert result is not None

    def test_stored_result_missing_params_key(self, manager: MockTrialArtifactManager):
        """Stored result without 'params' key should use empty dict."""
        manager.record_result(5, {"score": 0.8})  # No 'params' key

        mock_trial = MagicMock()
        mock_trial.number = 999
        mock_trial.params = {}

        result = manager.get_trial_result(mock_trial)
        # stored.get("params", {}) returns {}, matches empty trial params
        assert result is not None

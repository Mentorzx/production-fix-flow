"""Tests for HPO warmstart and memory persistence logic.

Fast unit tests (no I/O, no Optuna study) targeting:
- PersistentBestTrialMemory behavior
- Warmstart trial injection
- Score delta filtering
- Top-K selection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class HPOMemoryConfig:
    """Mock config matching scripts/optimization/core.py."""
    enabled: bool = True
    top_k_trials: int = 5
    warmstart_trials: int = 3
    storage_subdir: str = "memory"
    min_score_delta: float = 0.01


class MockPersistentBestTrialMemory:
    """Minimal mock of PersistentBestTrialMemory for fast testing."""

    def __init__(self, config: HPOMemoryConfig) -> None:
        self.config = config
        self.entries: list[dict[str, Any]] = []

    def _should_record(self, score: float) -> bool:
        """Check if score qualifies for recording based on min_score_delta."""
        if not self.entries:
            return True
        best_score = max(e["score"] for e in self.entries)
        return score >= best_score - self.config.min_score_delta

    def record_trial(self, trial_number: int, params: dict, score: float) -> bool:
        """Record a trial if it meets criteria."""
        if not self.config.enabled:
            return False
        if not self._should_record(score):
            return False
        
        self.entries.append({
            "trial_number": trial_number,
            "params": params,
            "score": score,
        })
        
        # Keep only top_k
        if len(self.entries) > self.config.top_k_trials:
            self.entries.sort(key=lambda x: x["score"], reverse=True)
            self.entries = self.entries[:self.config.top_k_trials]
        
        return True

    def get_warmstart_params(self) -> list[dict]:
        """Get params for warmstarting a new study."""
        sorted_entries = sorted(self.entries, key=lambda x: x["score"], reverse=True)
        return [e["params"] for e in sorted_entries[:self.config.warmstart_trials]]


class TestHPOMemoryRecording:
    """Test trial recording logic."""

    @pytest.fixture
    def memory(self) -> MockPersistentBestTrialMemory:
        config = HPOMemoryConfig(enabled=True, top_k_trials=3, min_score_delta=0.05)
        return MockPersistentBestTrialMemory(config)

    def test_first_trial_always_recorded(self, memory: MockPersistentBestTrialMemory):
        """First trial should always be recorded regardless of score."""
        result = memory.record_trial(0, {"lr": 0.01}, score=0.1)
        assert result is True
        assert len(memory.entries) == 1

    def test_better_score_recorded(self, memory: MockPersistentBestTrialMemory):
        """Trial with better score should be recorded."""
        memory.record_trial(0, {"lr": 0.01}, score=0.5)
        result = memory.record_trial(1, {"lr": 0.02}, score=0.6)
        assert result is True
        assert len(memory.entries) == 2

    def test_worse_score_within_delta_recorded(self, memory: MockPersistentBestTrialMemory):
        """Trial within min_score_delta should be recorded."""
        memory.record_trial(0, {"lr": 0.01}, score=0.5)
        # Score 0.46 is within delta (0.5 - 0.05 = 0.45)
        result = memory.record_trial(1, {"lr": 0.02}, score=0.46)
        assert result is True

    def test_worse_score_outside_delta_rejected(self, memory: MockPersistentBestTrialMemory):
        """Trial outside min_score_delta should be rejected."""
        memory.record_trial(0, {"lr": 0.01}, score=0.5)
        # Score 0.4 is outside delta (0.5 - 0.05 = 0.45)
        result = memory.record_trial(1, {"lr": 0.02}, score=0.4)
        assert result is False
        assert len(memory.entries) == 1

    def test_top_k_limit_enforced(self, memory: MockPersistentBestTrialMemory):
        """Only top_k trials should be kept."""
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        for i, score in enumerate(scores):
            memory.record_trial(i, {"lr": 0.01 * i}, score=score)
        
        assert len(memory.entries) == 3  # top_k_trials = 3
        # Should keep highest scores
        kept_scores = {e["score"] for e in memory.entries}
        assert kept_scores == {0.7, 0.8, 0.9}

    def test_disabled_memory_rejects_all(self):
        """Disabled memory should reject all trials."""
        config = HPOMemoryConfig(enabled=False)
        memory = MockPersistentBestTrialMemory(config)
        
        result = memory.record_trial(0, {"lr": 0.01}, score=0.9)
        assert result is False
        assert len(memory.entries) == 0


class TestHPOMemoryWarmstart:
    """Test warmstart param extraction."""

    @pytest.fixture
    def memory(self) -> MockPersistentBestTrialMemory:
        # Use large min_score_delta so all trials are recorded
        config = HPOMemoryConfig(top_k_trials=5, warmstart_trials=2, min_score_delta=1.0)
        return MockPersistentBestTrialMemory(config)

    def test_warmstart_returns_top_trials(self, memory: MockPersistentBestTrialMemory):
        """Warmstart should return params from top warmstart_trials."""
        memory.record_trial(0, {"lr": 0.01}, score=0.5)
        memory.record_trial(1, {"lr": 0.02}, score=0.7)
        memory.record_trial(2, {"lr": 0.03}, score=0.6)
        
        params = memory.get_warmstart_params()
        assert len(params) == 2
        # Should be sorted by score descending
        assert params[0] == {"lr": 0.02}  # score 0.7
        assert params[1] == {"lr": 0.03}  # score 0.6

    def test_warmstart_empty_memory(self, memory: MockPersistentBestTrialMemory):
        """Empty memory should return empty list."""
        params = memory.get_warmstart_params()
        assert params == []

    def test_warmstart_fewer_than_limit(self, memory: MockPersistentBestTrialMemory):
        """If fewer trials than limit, return all."""
        memory.record_trial(0, {"lr": 0.01}, score=0.5)
        
        params = memory.get_warmstart_params()
        assert len(params) == 1


class TestHPOMemoryEdgeCases:
    """Edge cases for memory management."""

    def test_zero_min_score_delta(self):
        """Zero delta should only accept equal or better scores."""
        config = HPOMemoryConfig(min_score_delta=0.0, top_k_trials=10)
        memory = MockPersistentBestTrialMemory(config)
        
        memory.record_trial(0, {"lr": 0.01}, score=0.5)
        
        # Equal score should be accepted
        assert memory.record_trial(1, {"lr": 0.02}, score=0.5) is True
        # Worse score should be rejected
        assert memory.record_trial(2, {"lr": 0.03}, score=0.49) is False

    def test_negative_scores(self):
        """Negative scores should work correctly."""
        config = HPOMemoryConfig(min_score_delta=0.1, top_k_trials=5)
        memory = MockPersistentBestTrialMemory(config)
        
        memory.record_trial(0, {"lr": 0.01}, score=-0.5)
        # -0.5 - 0.1 = -0.6, so -0.55 should be accepted
        assert memory.record_trial(1, {"lr": 0.02}, score=-0.55) is True
        # -0.7 is outside delta
        assert memory.record_trial(2, {"lr": 0.03}, score=-0.7) is False

    def test_duplicate_params_allowed(self):
        """Same params with different scores should both be recorded."""
        config = HPOMemoryConfig(top_k_trials=5, min_score_delta=1.0)
        memory = MockPersistentBestTrialMemory(config)
        
        memory.record_trial(0, {"lr": 0.01}, score=0.5)
        memory.record_trial(1, {"lr": 0.01}, score=0.6)  # Same params
        
        assert len(memory.entries) == 2

    def test_very_large_top_k(self):
        """Large top_k should not cause issues."""
        config = HPOMemoryConfig(top_k_trials=1000, min_score_delta=1.0)
        memory = MockPersistentBestTrialMemory(config)
        
        for i in range(100):
            memory.record_trial(i, {"lr": 0.01 * i}, score=i * 0.01)
        
        assert len(memory.entries) == 100  # All recorded since < 1000

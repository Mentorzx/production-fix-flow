"""Tests for TimeBudgetEstimator (Two-Phase Logic)."""

from pff.domain.learning.dslfm.time_estimator import (
    TimeBudgetConfig,
    TimeBudgetEstimator,
)


class MockClock:
    def __init__(self):
        self.current_time = 0.0

    def __call__(self):
        return self.current_time

    def advance(self, seconds: float):
        self.current_time += seconds


def test_phase1_pruning():
    """Test Phase 1: Pruning before halfway if projection > limit after eval window."""
    config = TimeBudgetConfig(
        enabled=True,
        max_total_time_s=900.0,
        tolerance_start_s=840.0,
        tolerance_evals=2,
        eval_time_window=3,
    )
    clock = MockClock()
    estimator = TimeBudgetEstimator(
        config, total_epochs=200, validate_every=10, clock=clock
    )

    clock.advance(30.0)
    assert estimator.check_budget(current_epoch=9) is False

    clock.advance(300.0)
    assert estimator.check_budget(current_epoch=19) is False

    clock.advance(300.0)
    assert estimator.check_budget(current_epoch=29) is True


def test_phase1_passing():
    """Test Phase 1: Passing if projection < limit after eval window."""
    config = TimeBudgetConfig(enabled=True, max_total_time_s=900.0, eval_time_window=3)
    clock = MockClock()
    estimator = TimeBudgetEstimator(
        config, total_epochs=200, validate_every=10, clock=clock
    )

    clock.advance(20.0)
    assert estimator.check_budget(current_epoch=9) is False

    clock.advance(20.0)
    assert estimator.check_budget(current_epoch=19) is False

    clock.advance(20.0)
    assert estimator.check_budget(current_epoch=29) is False


def test_phase1_early_eval_prune():
    """Test Phase 1: Early prune on first eval when epochs are too slow."""
    config = TimeBudgetConfig(enabled=True, max_total_time_s=300.0, eval_time_window=3)
    clock = MockClock()
    estimator = TimeBudgetEstimator(
        config, total_epochs=200, validate_every=10, clock=clock
    )

    clock.advance(200.0)
    should_prune = estimator.check_budget(current_epoch=9)
    assert should_prune is True


def test_phase2_grace_passing():
    """Test Phase 2: Grace permitted if next step fits."""
    config = TimeBudgetConfig(
        enabled=True,
        max_total_time_s=900.0,  # 15 min
        tolerance_start_s=840.0,  # 14 min
        tolerance_evals=2,
    )
    clock = MockClock()
    estimator = TimeBudgetEstimator(
        config, total_epochs=200, validate_every=10, clock=clock
    )

    # Halfway point. Elapsed 850s (> 840s).
    # Last interval was 10s.
    # Next proj = 850 + 10 = 860 < 900.
    # OK.
    clock.advance(850.0)

    # We need to simulate the interval from _last_eval_end_time (0) to now (850).
    # But wait, interval_duration = now - start (if first check).
    # So interval is 850s.
    # Next proj = 850 + 850 >>> 900.
    # It would prune immediately if we treat the whole 850s as one interval!
    # Correct usage implies periodic checks.

    # Let's verify we are in a steady state.
    # Assume previous eval finished at 840s.
    estimator._last_eval_end_time = 840.0

    should_prune = estimator.check_budget(current_epoch=99)
    assert should_prune is False  # Fits budget
    assert estimator.tolerance_counter == 1


def test_phase2_conditional_prune():
    """Test Phase 2: Immediate prune if next step exceeds limit (even with grace evals left)."""
    config = TimeBudgetConfig(
        enabled=True,
        max_total_time_s=900.0,  # 15 min
        tolerance_start_s=840.0,  # 14 min
        tolerance_evals=2,
    )
    clock = MockClock()
    estimator = TimeBudgetEstimator(
        config, total_epochs=200, validate_every=10, clock=clock
    )

    # Halfway point. Elapsed 850s (> 840s).
    clock.advance(850.0)

    # Last eval was at 790s. Interval = 60s.
    estimator._last_eval_end_time = 790.0

    # Next proj = 850 + 60 = 910 > 900.
    # Should PRUNE immediately, even though tolerance_counter (1) <= 2.

    should_prune = estimator.check_budget(current_epoch=99)
    assert should_prune is True
    assert estimator.tolerance_counter == 1


def test_phase2_hard_limit():
    """Test Phase 2: Prune after tolerance evals exhausted."""
    config = TimeBudgetConfig(
        enabled=True,
        max_total_time_s=900.0,
        tolerance_start_s=840.0,
        tolerance_evals=1,  # Only 1 allowed
    )
    clock = MockClock()
    estimator = TimeBudgetEstimator(
        config, total_epochs=200, validate_every=10, clock=clock
    )

    clock.advance(850.0)
    estimator._last_eval_end_time = 840.0  # 10s interval

    # First check. Elapsed 850. Next proj 860 < 900. OK.
    assert estimator.check_budget(current_epoch=99) is False
    assert estimator.tolerance_counter == 1

    # Second check. Elapsed 860.
    clock.advance(10.0)
    # Counter becomes 2 > 1. Prune.
    assert estimator.check_budget(current_epoch=109) is True

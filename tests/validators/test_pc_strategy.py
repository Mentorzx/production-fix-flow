"""Tests for Probabilistic Circuit aggregation strategy."""

import numpy as np
import pytest

from pff.domain.learning.pc.strategy import ProbabilisticCircuitStrategy


def test_pc_strategy_matches_noisy_or_small_input():
    """PC strategy should behave like Noisy-OR for independent rules."""
    pc = ProbabilisticCircuitStrategy(
        fallback_to_noisy_or=True, max_rules_per_circuit=1000
    )
    confidences = np.array([0.5, 0.5, 0.2], dtype=np.float64)
    result = pc.aggregate(confidences)
    expected = 1 - (0.5 * 0.5 * 0.8)
    assert pytest.approx(result, rel=1e-6) == expected


def test_pc_strategy_fallback_on_rule_limit():
    """When rule limit is exceeded, PC should fallback to Noisy-OR."""
    pc = ProbabilisticCircuitStrategy(
        fallback_to_noisy_or=True, max_rules_per_circuit=2
    )
    confidences = np.array([0.3, 0.4, 0.5], dtype=np.float64)
    result = pc.aggregate(confidences)
    expected = 1 - (0.7 * 0.6 * 0.5)
    assert pytest.approx(result, rel=1e-6) == expected


def test_pc_strategy_no_fallback_raises():
    """If fallback is disabled, exceeding the limit should raise."""
    pc = ProbabilisticCircuitStrategy(
        fallback_to_noisy_or=False, max_rules_per_circuit=1
    )
    confidences = np.array([0.2, 0.4], dtype=np.float64)
    with pytest.raises(Exception):
        pc.aggregate(confidences)

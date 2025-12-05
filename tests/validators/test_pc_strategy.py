import numpy as np
import pytest

from pff.validators.ensembles.hierarchical.symbolic_aggregator import (
    SymbolicAggregator,
)


def test_pc_strategy_matches_noisy_or_small_input():
    """PC strategy should behave like Noisy-OR for independent rules."""
    aggregator = SymbolicAggregator(
        strategy="pc",
        params={"fallback_to_noisy_or": True, "max_rules_per_circuit": 1000},
    )
    result = aggregator.aggregate_single([0.5, 0.5, 0.2])
    expected = 1 - (0.5 * 0.5 * 0.8)
    assert pytest.approx(result.confidence, rel=1e-6) == expected
    assert result.strategy_used == "pc"


def test_pc_strategy_fallback_on_rule_limit():
    """When rule limit is exceeded, PC should fallback to Noisy-OR."""
    aggregator = SymbolicAggregator(
        strategy="pc",
        params={"fallback_to_noisy_or": True, "max_rules_per_circuit": 2},
    )
    result = aggregator.aggregate_single([0.3, 0.4, 0.5])
    expected = 1 - (0.7 * 0.6 * 0.5)
    assert pytest.approx(result.confidence, rel=1e-6) == expected


def test_pc_strategy_no_fallback_raises():
    """If fallback is disabled, exceeding the limit should raise."""
    aggregator = SymbolicAggregator(
        strategy="pc",
        params={"fallback_to_noisy_or": False, "max_rules_per_circuit": 1},
    )
    with pytest.raises(Exception):
        aggregator.aggregate_single([0.2, 0.4])

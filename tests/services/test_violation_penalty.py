"""
Tests for ViolationPenaltyCalculator.

Tests the extracted penalty calculation logic following SRP.

NOTE: The penalty calculator now uses violations_per_k_rules (violations
per 1000 rules) instead of raw violation_rate. This handles large rule
sets (18K+ rules) properly where raw violation_rate would be too small.
"""

import pytest
from pff.services.violation_penalty import (
    ViolationPenaltyCalculator,
    PenaltyConfig,
)


class TestPenaltyConfig:
    """Tests for PenaltyConfig dataclass."""

    def test_default_values(self):
        """Default config values should match expected defaults.
        
        Note: rate_floor is now in violations per 1K rules (5.0)
        and penalty_multiplier is 0.05 per violation per 1K rules.
        """
        config = PenaltyConfig()
        assert config.rate_floor == 5.0  # 5 violations per 1K rules threshold
        assert config.penalty_multiplier == 0.05  # 0.05 penalty per violation per 1K
        assert config.max_penalty == 0.65
        assert config.no_violations_bonus == 0.35
        assert config.below_threshold_bonus == 0.15
        assert config.confidence_anchor == 0.5

    def test_from_config_dict(self):
        """Config should load from dictionary."""
        custom = {
            "rate_floor": 10.0,
            "penalty_multiplier": 0.1,
            "max_penalty": 0.5,
            "no_violations_bonus": 0.4,
            "below_threshold_bonus": 0.2,
            "confidence_anchor": 0.6,
        }
        config = PenaltyConfig.from_config(custom)
        assert config.rate_floor == 10.0
        assert config.penalty_multiplier == 0.1
        assert config.max_penalty == 0.5

    def test_from_config_partial(self):
        """Missing keys should use defaults."""
        partial = {"rate_floor": 10.0}
        config = PenaltyConfig.from_config(partial)
        assert config.rate_floor == 10.0
        assert config.penalty_multiplier == 0.05  # default (new value)


class TestViolationPenaltyCalculator:
    """Tests for ViolationPenaltyCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator with default config."""
        return ViolationPenaltyCalculator(PenaltyConfig())

    def test_no_violations_returns_bonus(self, calculator):
        """Zero violations should return negative penalty (bonus)."""
        features = {
            "num_violations": 0,
            "total_rules": 18000,
            "violation_rate": 0.0,
            "violations_per_k_rules": 0.0,
            "avg_confidence": 0.0,
        }
        penalty, metadata = calculator.compute(features)

        assert penalty < 0, "No violations should give bonus (negative penalty)"
        assert penalty == -0.35
        assert metadata["penalty_reason"] == "no_violations_bonus"
        assert metadata["applied_bonus"] == 0.35

    def test_few_violations_returns_small_bonus(self, calculator):
        """Violations below rate floor (5 per 1K rules) should give small bonus."""
        features = {
            "num_violations": 50,
            "total_rules": 18000,  # 2.78 per 1K rules, below 5.0 floor
            "violation_rate": 0.00278,
            "violations_per_k_rules": 2.78,
            "avg_confidence": 0.5,
        }
        penalty, metadata = calculator.compute(features)

        assert penalty < 0, "Below-threshold violations should give bonus"
        assert penalty == -0.15
        assert metadata["penalty_reason"] == "below_threshold_bonus"

    def test_many_violations_returns_penalty(self, calculator):
        """High violation density should return positive penalty."""
        features = {
            "num_violations": 200,
            "total_rules": 18000,  # 11.1 per 1K rules
            "violation_rate": 0.0111,
            "violations_per_k_rules": 11.11,
            "avg_confidence": 0.7,
        }
        penalty, metadata = calculator.compute(features)

        assert penalty > 0, "Many violations should give penalty"
        assert metadata["penalty_reason"] == "violation_density"
        assert "applied_penalty" in metadata
        # Expected: 11.11 * 0.05 + (0.7 - 0.5) * 0.1 = 0.556 + 0.02 = 0.576
        assert 0.5 < penalty < 0.65

    def test_penalty_capped_at_max(self, calculator):
        """Penalty should not exceed max_penalty."""
        features = {
            "num_violations": 1000,
            "total_rules": 5000,  # 200 per 1K rules (extreme)
            "violation_rate": 0.2,
            "violations_per_k_rules": 200.0,
            "avg_confidence": 1.0,
        }
        penalty, _ = calculator.compute(features)

        assert penalty <= 0.65, "Penalty should be capped at max_penalty"
        assert penalty == 0.65  # Should hit the cap

    def test_high_confidence_increases_penalty(self, calculator):
        """High confidence violations should increase penalty."""
        base_features = {
            "num_violations": 100,
            "total_rules": 18000,  # 5.56 per 1K rules (just above threshold)
            "violation_rate": 0.00556,
            "violations_per_k_rules": 5.56,
        }

        # Low confidence (below anchor)
        low_conf = {**base_features, "avg_confidence": 0.3}
        penalty_low, _ = calculator.compute(low_conf)

        # High confidence (above anchor)
        high_conf = {**base_features, "avg_confidence": 0.9}
        penalty_high, _ = calculator.compute(high_conf)

        # penalty_low = 5.56 * 0.05 + 0 = 0.278
        # penalty_high = 5.56 * 0.05 + (0.9 - 0.5) * 0.1 = 0.278 + 0.04 = 0.318
        assert penalty_high > penalty_low, "Higher confidence should increase penalty"

    def test_custom_config(self):
        """Calculator should respect custom config."""
        custom_config = PenaltyConfig(
            no_violations_bonus=0.5,
            below_threshold_bonus=0.25,
        )
        calculator = ViolationPenaltyCalculator(custom_config)

        features = {"num_violations": 0, "total_rules": 100}
        penalty, metadata = calculator.compute(features)

        assert penalty == -0.5
        assert metadata["applied_bonus"] == 0.5

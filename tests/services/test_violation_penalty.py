"""
Tests for ViolationPenaltyCalculator.

Tests the extracted penalty calculation logic following SRP.
"""

import pytest
from pff.services.violation_penalty import (
    ViolationPenaltyCalculator,
    PenaltyConfig,
)


class TestPenaltyConfig:
    """Tests for PenaltyConfig dataclass."""

    def test_default_values(self):
        """Default config values should match expected defaults."""
        config = PenaltyConfig()
        assert config.rate_floor == 0.005
        assert config.penalty_multiplier == 12.0
        assert config.max_penalty == 0.45
        assert config.no_violations_bonus == 0.35
        assert config.below_threshold_bonus == 0.15
        assert config.confidence_anchor == 0.5

    def test_from_config_dict(self):
        """Config should load from dictionary."""
        custom = {
            "rate_floor": 0.01,
            "penalty_multiplier": 10.0,
            "max_penalty": 0.5,
            "no_violations_bonus": 0.4,
            "below_threshold_bonus": 0.2,
            "confidence_anchor": 0.6,
        }
        config = PenaltyConfig.from_config(custom)
        assert config.rate_floor == 0.01
        assert config.penalty_multiplier == 10.0
        assert config.max_penalty == 0.5

    def test_from_config_partial(self):
        """Missing keys should use defaults."""
        partial = {"rate_floor": 0.01}
        config = PenaltyConfig.from_config(partial)
        assert config.rate_floor == 0.01
        assert config.penalty_multiplier == 12.0  # default


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
            "total_rules": 100,
            "violation_rate": 0.0,
            "avg_confidence": 0.0,
        }
        penalty, metadata = calculator.compute(features)

        assert penalty < 0, "No violations should give bonus (negative penalty)"
        assert penalty == -0.35
        assert metadata["penalty_reason"] == "no_violations_bonus"
        assert metadata["applied_bonus"] == 0.35

    def test_few_violations_returns_small_bonus(self, calculator):
        """Violations below rate floor should give small bonus."""
        features = {
            "num_violations": 2,
            "total_rules": 1000,  # 0.2% rate, below 0.5% floor
            "violation_rate": 0.002,
            "avg_confidence": 0.5,
        }
        penalty, metadata = calculator.compute(features)

        assert penalty < 0, "Below-threshold violations should give bonus"
        assert penalty == -0.15
        assert metadata["penalty_reason"] == "below_threshold_bonus"

    def test_many_violations_returns_penalty(self, calculator):
        """High violation rate should return positive penalty."""
        features = {
            "num_violations": 100,
            "total_rules": 500,  # 20% rate
            "violation_rate": 0.2,
            "avg_confidence": 0.7,
        }
        penalty, metadata = calculator.compute(features)

        assert penalty > 0, "Many violations should give penalty"
        assert metadata["penalty_reason"] == "violation_density"
        assert "applied_penalty" in metadata

    def test_penalty_capped_at_max(self, calculator):
        """Penalty should not exceed max_penalty."""
        features = {
            "num_violations": 1000,
            "total_rules": 100,  # 1000% rate (extreme)
            "violation_rate": 10.0,
            "avg_confidence": 1.0,
        }
        penalty, _ = calculator.compute(features)

        assert penalty <= 0.45, "Penalty should be capped at max_penalty"

    def test_high_confidence_increases_penalty(self, calculator):
        """High confidence violations should increase penalty."""
        base_features = {
            "num_violations": 10,
            "total_rules": 500,
            "violation_rate": 0.02,  # 2% - low enough to not hit cap
        }

        # Low confidence
        low_conf = {**base_features, "avg_confidence": 0.3}
        penalty_low, _ = calculator.compute(low_conf)

        # High confidence
        high_conf = {**base_features, "avg_confidence": 0.9}
        penalty_high, _ = calculator.compute(high_conf)

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

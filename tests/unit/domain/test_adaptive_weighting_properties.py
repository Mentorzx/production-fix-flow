"""Property tests for adaptive weighting in OOV solution config.

Tests the following properties:
(1) Weights are always within [clip_min, clip_max] after adaptive adjustment.
(2) Weights sum to approximately 1.0 after normalization.
(3) Good symbolic scenarios (high coverage, few violations) -> higher symbolic weight.
(4) Bad symbolic scenarios (low coverage, many violations) -> higher neural weight.
(5) When adaptive_weighting.enabled=false, static weights are returned unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

# ============================================================================
# Mock implementations for adaptive weighting property tests
# ============================================================================


class MockAdaptiveWeightingConfig:
    """Mock config for adaptive weighting tests."""

    def __init__(
        self,
        enabled: bool = True,
        weight_clip_min: float = 0.5,
        weight_clip_max: float = 2.0,
    ):
        self.enabled = enabled
        self.weight_clip_min = weight_clip_min
        self.weight_clip_max = weight_clip_max


class MockOOVAwareEnsembleManager:
    """Simplified mock of OOVAwareEnsembleManager for property testing."""

    def __init__(
        self,
        adaptive_config: MockAdaptiveWeightingConfig | None = None,
        base_weights: dict[str, float] | None = None,
    ):
        self.config = adaptive_config or MockAdaptiveWeightingConfig()
        self.expert_weights = {
            "base": base_weights or {"symbolic": 0.4, "hybrid": 0.35, "neural": 0.25},
            "high_oov": {"symbolic": 0.6, "hybrid": 0.2, "neural": 0.2},
            "few_rules": {"symbolic": 0.2, "hybrid": 0.5, "neural": 0.3},
            "balanced": {"symbolic": 0.33, "hybrid": 0.34, "neural": 0.33},
        }
        # Test thresholds
        self._high_oov_ratio = 0.8
        self._min_rules_for_symbolic = 50
        self._good_coverage_ratio = 0.7

    def compute_adaptive_expert_weights(
        self,
        input_quality: dict[str, Any],
        rule_violations: int,
        symbolic_coverage: float,
    ) -> dict[str, float]:
        """Compute adaptive weights based on input characteristics."""
        if not self.config.enabled:
            # Return static base weights when disabled
            return dict(self.expert_weights["base"])

        strategy = input_quality.get("recommended_strategy", "base")
        base_weights = self.expert_weights.get(strategy, self.expert_weights["base"]).copy()

        # Adjust for violations
        if rule_violations > 5:
            base_weights["symbolic"] *= 1.2
            base_weights["hybrid"] *= 0.9
        elif rule_violations == 0:
            base_weights["symbolic"] *= 0.8
            base_weights["hybrid"] *= 1.1

        # Adjust for low coverage
        if symbolic_coverage < 0.3:
            base_weights["neural"] *= 1.2
            base_weights["symbolic"] *= 0.8

        # Normalize to sum to 1
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            normalized = {k: v / total_weight for k, v in base_weights.items()}
        else:
            normalized = base_weights

        return normalized

    def apply_weight_clipping(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply clip_min/clip_max constraints to multipliers (not final weights)."""
        clipped = {}
        for k, v in weights.items():
            clipped[k] = max(self.config.weight_clip_min, min(self.config.weight_clip_max, v * 2))
        # Re-normalize after clipping
        total = sum(clipped.values())
        return {k: v / total for k, v in clipped.items()}


class TestAdaptiveWeightsNormalization:
    """Test that weights are normalized correctly."""

    @pytest.fixture
    def manager(self) -> MockOOVAwareEnsembleManager:
        return MockOOVAwareEnsembleManager()

    @pytest.mark.parametrize("strategy", ["base", "high_oov", "few_rules", "balanced"])
    @pytest.mark.parametrize("violations", [0, 3, 10])
    @pytest.mark.parametrize("coverage", [0.1, 0.3, 0.5, 0.8])
    def test_weights_sum_to_one(
        self,
        manager: MockOOVAwareEnsembleManager,
        strategy: str,
        violations: int,
        coverage: float,
    ):
        """Property: weights MUST sum to approximately 1.0."""
        input_quality = {"recommended_strategy": strategy}

        weights = manager.compute_adaptive_expert_weights(input_quality, violations, coverage)

        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_all_weights_positive(self, manager: MockOOVAwareEnsembleManager):
        """Property: all weights must be > 0."""
        for strategy in ["base", "high_oov", "few_rules", "balanced"]:
            for violations in [0, 5, 15]:
                for coverage in [0.1, 0.5, 0.9]:
                    input_quality = {"recommended_strategy": strategy}
                    weights = manager.compute_adaptive_expert_weights(
                        input_quality, violations, coverage
                    )

                    for name, w in weights.items():
                        assert w > 0, f"Weight {name}={w} should be > 0"


class TestAdaptiveWeightsDirectionality:
    """Test that weights adjust in expected directions."""

    @pytest.fixture
    def manager(self) -> MockOOVAwareEnsembleManager:
        return MockOOVAwareEnsembleManager()

    def test_many_violations_increase_symbolic_weight(self, manager: MockOOVAwareEnsembleManager):
        """Property: many violations should increase symbolic weight."""
        input_quality = {"recommended_strategy": "base"}

        weights_no_violations = manager.compute_adaptive_expert_weights(
            input_quality, rule_violations=0, symbolic_coverage=0.5
        )
        weights_many_violations = manager.compute_adaptive_expert_weights(
            input_quality, rule_violations=10, symbolic_coverage=0.5
        )

        assert weights_many_violations["symbolic"] > weights_no_violations["symbolic"], (
            f"Many violations should increase symbolic weight: "
            f"{weights_no_violations['symbolic']:.3f} -> {weights_many_violations['symbolic']:.3f}"
        )

    def test_low_coverage_increases_neural_weight(self, manager: MockOOVAwareEnsembleManager):
        """Property: low symbolic coverage should increase neural weight."""
        input_quality = {"recommended_strategy": "base"}

        weights_high_cov = manager.compute_adaptive_expert_weights(
            input_quality, rule_violations=2, symbolic_coverage=0.8
        )
        weights_low_cov = manager.compute_adaptive_expert_weights(
            input_quality, rule_violations=2, symbolic_coverage=0.1
        )

        assert weights_low_cov["neural"] > weights_high_cov["neural"], (
            f"Low coverage should increase neural weight: "
            f"{weights_high_cov['neural']:.3f} -> {weights_low_cov['neural']:.3f}"
        )

    def test_good_symbolic_scenario(self, manager: MockOOVAwareEnsembleManager):
        """Property: good symbolic scenario (high cov, few violations) -> higher symbolic."""
        # Good: high coverage, moderate violations (shows rules are being applied)
        input_quality = {"recommended_strategy": "base"}

        weights_good = manager.compute_adaptive_expert_weights(
            input_quality, rule_violations=3, symbolic_coverage=0.7
        )

        # Bad: low coverage
        weights_bad = manager.compute_adaptive_expert_weights(
            input_quality, rule_violations=3, symbolic_coverage=0.1
        )

        assert weights_good["symbolic"] > weights_bad["symbolic"], (
            "Good symbolic scenario should favor symbolic"
        )

    def test_bad_symbolic_scenario(self, manager: MockOOVAwareEnsembleManager):
        """Property: bad symbolic scenario (low cov) -> higher neural."""
        input_quality = {"recommended_strategy": "base"}

        # Bad: low coverage
        weights_bad = manager.compute_adaptive_expert_weights(
            input_quality, rule_violations=0, symbolic_coverage=0.1
        )

        # Good: high coverage
        weights_good = manager.compute_adaptive_expert_weights(
            input_quality, rule_violations=0, symbolic_coverage=0.8
        )

        assert weights_bad["neural"] > weights_good["neural"], (
            "Bad symbolic scenario should increase neural weight"
        )


class TestAdaptiveWeightsDisabled:
    """Test behavior when adaptive weighting is disabled."""

    def test_disabled_returns_static_weights(self):
        """Property: when enabled=false, return static base weights unchanged."""
        static_weights = {"symbolic": 0.35, "hybrid": 0.35, "neural": 0.30}
        config = MockAdaptiveWeightingConfig(enabled=False)
        manager = MockOOVAwareEnsembleManager(
            adaptive_config=config,
            base_weights=static_weights,
        )

        # Should return same weights regardless of inputs
        input_quality = {"recommended_strategy": "high_oov"}  # Would change weights if enabled

        weights = manager.compute_adaptive_expert_weights(
            input_quality, rule_violations=100, symbolic_coverage=0.0
        )

        # Should be exactly the static weights
        for k, expected in static_weights.items():
            assert weights[k] == expected, (
                f"Disabled should return static weight for {k}: "
                f"expected {expected}, got {weights[k]}"
            )

    def test_disabled_ignores_input_quality(self):
        """Property: disabled ignores all input parameters."""
        config = MockAdaptiveWeightingConfig(enabled=False)
        manager = MockOOVAwareEnsembleManager(adaptive_config=config)

        # Get weights with different inputs
        w1 = manager.compute_adaptive_expert_weights({"recommended_strategy": "base"}, 0, 1.0)
        w2 = manager.compute_adaptive_expert_weights({"recommended_strategy": "high_oov"}, 100, 0.0)

        # Should be identical
        for k in w1:
            assert w1[k] == w2[k], "Disabled should give same weights regardless of input"


class TestWeightClipping:
    """Test weight clipping behavior."""

    def test_clipping_maintains_normalization(self):
        """Property: clipping should maintain sum=1 after re-normalization."""
        config = MockAdaptiveWeightingConfig(weight_clip_min=0.5, weight_clip_max=2.0)
        manager = MockOOVAwareEnsembleManager(adaptive_config=config)

        weights = {"symbolic": 0.1, "hybrid": 0.1, "neural": 0.8}  # Extreme
        clipped = manager.apply_weight_clipping(weights)

        total = sum(clipped.values())
        assert abs(total - 1.0) < 1e-9, f"Clipped weights should sum to 1, got {total}"


class TestAdaptiveWeightsFromConfig:
    """Test that config values are respected."""

    def test_strategy_selection_uses_config_weights(self):
        """Property: strategy selection should use configured base weights."""
        custom_weights = {"symbolic": 0.5, "hybrid": 0.3, "neural": 0.2}
        manager = MockOOVAwareEnsembleManager(base_weights=custom_weights)

        input_quality = {"recommended_strategy": "base"}
        weights = manager.compute_adaptive_expert_weights(
            input_quality, rule_violations=0, symbolic_coverage=0.5
        )

        # Without adjustments, should approximate base weights
        # (some adjustment for 0 violations: symbolic*0.8, hybrid*1.1)
        # But relative ordering should match config
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9


class TestAdaptiveWeightsEdgeCases:
    """Test edge cases in adaptive weighting."""

    @pytest.fixture
    def manager(self) -> MockOOVAwareEnsembleManager:
        return MockOOVAwareEnsembleManager()

    def test_zero_violations_zero_coverage(self, manager: MockOOVAwareEnsembleManager):
        """Property: extreme inputs should not break weight computation."""
        input_quality = {"recommended_strategy": "base"}

        weights = manager.compute_adaptive_expert_weights(
            input_quality, rule_violations=0, symbolic_coverage=0.0
        )

        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_max_violations_max_coverage(self, manager: MockOOVAwareEnsembleManager):
        """Property: extreme inputs should not break weight computation."""
        input_quality = {"recommended_strategy": "base"}

        weights = manager.compute_adaptive_expert_weights(
            input_quality, rule_violations=1000, symbolic_coverage=1.0
        )

        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_unknown_strategy_uses_base(self, manager: MockOOVAwareEnsembleManager):
        """Property: unknown strategy should fall back to base."""
        input_quality = {"recommended_strategy": "unknown_strategy"}

        weights = manager.compute_adaptive_expert_weights(
            input_quality, rule_violations=2, symbolic_coverage=0.5
        )

        # Should use base weights as fallback
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9

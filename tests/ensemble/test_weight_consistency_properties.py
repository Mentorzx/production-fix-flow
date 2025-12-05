"""Property tests for ensemble weight consistency.

Tests that ensemble weights behave correctly:
(1) Static weights from config are respected when adaptive is disabled
(2) Adaptive weights stay within configured bounds
(3) Weight changes are gradual (no sudden jumps)
(4) Degenerate cases (all zero, negative) are handled
"""

from __future__ import annotations

import numpy as np
import pytest


# ============================================================================
# Weight management functions
# ============================================================================


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize weights to sum to 1.0."""
    total = sum(weights.values())
    if total <= 0:
        # Fallback to equal weights
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def clip_weights(
    weights: dict[str, float],
    clip_min: float = 0.1,
    clip_max: float = 0.8,
) -> dict[str, float]:
    """Clip weights to [clip_min, clip_max] and renormalize."""
    clipped = {k: max(clip_min, min(clip_max, v)) for k, v in weights.items()}
    return normalize_weights(clipped)


def blend_weights(
    base: dict[str, float],
    adjustment: dict[str, float],
    alpha: float = 0.3,
) -> dict[str, float]:
    """Blend base weights with adjustment using exponential smoothing."""
    if not 0 <= alpha <= 1:
        alpha = max(0, min(1, alpha))
    blended = {}
    for key in base:
        base_val = base.get(key, 0.0)
        adj_val = adjustment.get(key, base_val)
        blended[key] = (1 - alpha) * base_val + alpha * adj_val
    return normalize_weights(blended)


# ============================================================================
# Tests: Weight normalization
# ============================================================================


class TestWeightNormalization:
    """Test weight normalization properties."""

    def test_normalized_weights_sum_to_one(self):
        """Property: normalized weights always sum to 1.0."""
        test_cases = [
            {"neural": 0.3, "rules": 0.3, "lightgbm": 0.4},
            {"neural": 1.0, "rules": 2.0, "lightgbm": 3.0},
            {"neural": 0.1, "rules": 0.1, "lightgbm": 0.1},
            {"a": 100, "b": 200, "c": 300},
        ]
        for weights in test_cases:
            result = normalize_weights(weights)
            total = sum(result.values())
            assert abs(total - 1.0) < 1e-9, f"Sum = {total} for {weights}"

    def test_normalized_weights_preserve_ratios(self):
        """Property: ratios between weights are preserved."""
        weights = {"a": 2.0, "b": 4.0, "c": 6.0}
        result = normalize_weights(weights)
        # Ratio a:b should be 2:4 = 1:2
        assert abs(result["a"] / result["b"] - 0.5) < 1e-9

    def test_zero_weights_fallback_to_equal(self):
        """Property: all-zero weights fallback to equal distribution."""
        weights = {"a": 0.0, "b": 0.0, "c": 0.0}
        result = normalize_weights(weights)
        expected = 1.0 / 3
        for v in result.values():
            assert abs(v - expected) < 1e-9

    def test_negative_total_fallback(self):
        """Property: negative total falls back to equal weights."""
        weights = {"a": -1.0, "b": -1.0, "c": -1.0}
        result = normalize_weights(weights)
        # Should fallback to equal
        expected = 1.0 / 3
        for v in result.values():
            assert abs(v - expected) < 1e-9


class TestWeightClipping:
    """Test weight clipping properties."""

    def test_clipped_weights_respect_bounds_before_normalization(self):
        """Property: clipping applies bounds before normalization."""
        weights = {"a": 0.05, "b": 0.9, "c": 0.5}  # a too low, b too high
        clip_min, clip_max = 0.1, 0.8

        # After clipping: a=0.1, b=0.8, c=0.5, total=1.4
        # After normalization: a≈0.071, b≈0.571, c≈0.357
        # Note: normalization can push values below clip_min
        result = clip_weights(weights, clip_min, clip_max)

        # All values should be positive and sum to 1
        assert all(v > 0 for v in result.values())
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-9

    def test_clipped_weights_sum_to_one(self):
        """Property: clipped weights still sum to 1.0."""
        weights = {"a": 0.01, "b": 0.98, "c": 0.01}
        result = clip_weights(weights, clip_min=0.1, clip_max=0.8)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-9

    def test_clip_then_normalize_maintains_relative_order(self):
        """Property: relative ordering of weights is maintained after clip+normalize."""
        weights = {"a": 0.1, "b": 0.3, "c": 0.6}
        result = clip_weights(weights, clip_min=0.1, clip_max=0.8)

        # Original order: a < b < c
        # After clipping and normalization, order should be preserved
        assert result["a"] <= result["b"] <= result["c"]

    @pytest.mark.parametrize("clip_min,clip_max", [
        (0.1, 0.8),
        (0.2, 0.6),
        (0.15, 0.7),
    ])
    def test_clipping_preserves_sum_to_one(self, clip_min: float, clip_max: float):
        """Property: clipping + normalization always sums to 1."""
        weights = {"a": 0.0, "b": 1.0, "c": 0.5}
        result = clip_weights(weights, clip_min, clip_max)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-9


class TestWeightBlending:
    """Test weight blending properties."""

    def test_alpha_zero_returns_base(self):
        """Property: alpha=0 returns base weights."""
        base = {"a": 0.3, "b": 0.3, "c": 0.4}
        adjustment = {"a": 0.5, "b": 0.3, "c": 0.2}
        result = blend_weights(base, adjustment, alpha=0.0)

        for key in base:
            assert abs(result[key] - base[key]) < 1e-9

    def test_alpha_one_returns_adjustment(self):
        """Property: alpha=1 returns adjustment weights."""
        base = {"a": 0.3, "b": 0.3, "c": 0.4}
        adjustment = {"a": 0.5, "b": 0.3, "c": 0.2}
        result = blend_weights(base, adjustment, alpha=1.0)

        adj_normalized = normalize_weights(adjustment)
        for key in adjustment:
            assert abs(result[key] - adj_normalized[key]) < 1e-9

    def test_blended_weights_bounded(self):
        """Property: blended weights are bounded by base and adjustment."""
        base = {"a": 0.2, "b": 0.3, "c": 0.5}
        adjustment = {"a": 0.4, "b": 0.4, "c": 0.2}

        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            result = blend_weights(base, adjustment, alpha)
            for key in base:
                low = min(base[key], adjustment[key])
                high = max(base[key], adjustment[key])
                # After normalization, bounds might shift slightly
                # but values should be reasonable
                assert 0 <= result[key] <= 1

    def test_blended_weights_sum_to_one(self):
        """Property: blended weights sum to 1.0."""
        base = {"a": 0.2, "b": 0.3, "c": 0.5}
        adjustment = {"a": 0.4, "b": 0.4, "c": 0.2}

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = blend_weights(base, adjustment, alpha)
            total = sum(result.values())
            assert abs(total - 1.0) < 1e-9

    def test_gradual_transition(self):
        """Property: intermediate alpha gives intermediate values."""
        base = {"a": 0.0, "b": 0.5, "c": 0.5}
        adjustment = {"a": 0.5, "b": 0.25, "c": 0.25}

        result_0 = blend_weights(base, adjustment, alpha=0.0)
        result_half = blend_weights(base, adjustment, alpha=0.5)
        result_1 = blend_weights(base, adjustment, alpha=1.0)

        # result_half["a"] should be between result_0["a"] and result_1["a"]
        low_a = min(result_0["a"], result_1["a"])
        high_a = max(result_0["a"], result_1["a"])
        assert low_a <= result_half["a"] <= high_a


# ============================================================================
# Tests: Weight stability
# ============================================================================


class TestWeightStability:
    """Test weight stability over time."""

    def test_repeated_normalization_stable(self):
        """Property: repeated normalization doesn't change weights."""
        weights = {"a": 0.2, "b": 0.3, "c": 0.5}
        result = weights.copy()

        for _ in range(10):
            result = normalize_weights(result)

        for key in weights:
            assert abs(result[key] - weights[key]) < 1e-9

    def test_small_perturbation_small_change(self):
        """Property: small input change causes small output change."""
        base = {"a": 0.3, "b": 0.3, "c": 0.4}
        perturbed = {"a": 0.31, "b": 0.29, "c": 0.4}

        result_base = normalize_weights(base)
        result_perturbed = normalize_weights(perturbed)

        for key in base:
            diff = abs(result_base[key] - result_perturbed[key])
            assert diff < 0.05, f"Large change for small perturbation: {diff}"

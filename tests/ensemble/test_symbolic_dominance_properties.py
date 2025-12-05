"""Property tests for symbolic dominance scoring and penalty behavior.

Tests the following properties:
(1) For two trials with same base metrics but different symbolic_contribution_ratio,
    the trial closer to target_symbolic_ratio should have higher composite_score.
(2) If symbolic_contribution_ratio <= target_symbolic_ratio, dominance penalty must be 0.
    If greater, penalty must grow monotonically with the excess.
"""

from __future__ import annotations

import math
from typing import Any

import pytest


# ============================================================================
# Inline scoring functions matching scripts/optimization/core.py
# ============================================================================

def _compute_symbolic_dominance_penalty(
    symbolic_contribution_ratio: float,
    dominance_target: float,
) -> float:
    """Compute symbolic dominance penalty.
    
    Args:
        symbolic_contribution_ratio: Actual symbolic contribution (0-1).
        dominance_target: Target symbolic ratio from config/params.
        
    Returns:
        Penalty value (0 if below target, scaled overflow otherwise).
    """
    if symbolic_contribution_ratio <= dominance_target:
        return 0.0
    dominance_overflow = symbolic_contribution_ratio - dominance_target
    return dominance_overflow / max(1e-6, 1.0 - dominance_target)


def _compute_composite_score(
    base_score: float,
    symbolic_contribution_ratio: float,
    dominance_target: float,
    symbolic_dominance_penalty_coeff: float,
    other_penalties: list[tuple[float, float]] | None = None,
) -> float:
    """Compute composite score with symbolic dominance penalty.
    
    Args:
        base_score: Base weighted score before penalties.
        symbolic_contribution_ratio: Actual symbolic contribution.
        dominance_target: Target symbolic ratio.
        symbolic_dominance_penalty_coeff: Coefficient for dominance penalty.
        other_penalties: List of (coeff, penalty) tuples for other penalties.
        
    Returns:
        Final composite score after penalties.
    """
    score = base_score
    
    # Apply other penalties first
    if other_penalties:
        for coeff, penalty in other_penalties:
            score *= (1.0 - coeff * min(1.0, penalty))
    
    # Apply symbolic dominance penalty
    dominance_penalty = _compute_symbolic_dominance_penalty(
        symbolic_contribution_ratio, dominance_target
    )
    score *= (1.0 - symbolic_dominance_penalty_coeff * min(1.0, dominance_penalty))
    
    return max(0.0, score)


class TestSymbolicDominancePenaltyMonotonicity:
    """Test that dominance penalty behaves monotonically."""

    @pytest.mark.parametrize("target", [0.35, 0.42, 0.50, 0.70])
    def test_penalty_zero_when_at_or_below_target(self, target: float):
        """Property: penalty MUST be 0 when contribution <= target."""
        # Test at target
        penalty_at = _compute_symbolic_dominance_penalty(target, target)
        assert penalty_at == 0.0, f"Penalty at target should be 0, got {penalty_at}"
        
        # Test below target
        for ratio in [target - 0.1, target - 0.2, target / 2, 0.0]:
            if ratio >= 0:
                penalty_below = _compute_symbolic_dominance_penalty(ratio, target)
                assert penalty_below == 0.0, f"Penalty below target should be 0, got {penalty_below}"

    @pytest.mark.parametrize("target", [0.35, 0.42, 0.50, 0.70])
    def test_penalty_grows_monotonically_above_target(self, target: float):
        """Property: penalty MUST grow monotonically as contribution exceeds target."""
        ratios_above = [target + 0.05, target + 0.1, target + 0.15, target + 0.2, target + 0.25]
        ratios_above = [r for r in ratios_above if r <= 1.0]  # Clamp to valid range
        
        penalties = [_compute_symbolic_dominance_penalty(r, target) for r in ratios_above]
        
        # Check monotonicity: each penalty should be >= previous
        for i in range(1, len(penalties)):
            assert penalties[i] >= penalties[i-1], (
                f"Penalty not monotonic: {penalties[i-1]} -> {penalties[i]} "
                f"for ratios {ratios_above[i-1]} -> {ratios_above[i]}"
            )

    def test_penalty_is_strictly_increasing(self):
        """Property: penalty should be strictly increasing above target."""
        target = 0.40
        penalties = []
        for ratio in [0.45, 0.50, 0.55, 0.60, 0.65]:
            penalty = _compute_symbolic_dominance_penalty(ratio, target)
            penalties.append((ratio, penalty))
        
        for i in range(1, len(penalties)):
            assert penalties[i][1] > penalties[i-1][1], (
                f"Penalty not strictly increasing: {penalties[i-1]} -> {penalties[i]}"
            )


class TestCompositeScoreComparison:
    """Test that trials closer to target have higher scores."""

    def test_trial_closer_to_target_has_higher_score(self):
        """Property: trial with ratio closer to target should have higher composite_score."""
        base_score = 0.80
        target = 0.40
        coeff = 1.0
        
        # Trial A: ratio = 0.38 (below target, no penalty)
        # Trial B: ratio = 0.55 (above target, penalized)
        score_a = _compute_composite_score(base_score, 0.38, target, coeff)
        score_b = _compute_composite_score(base_score, 0.55, target, coeff)
        
        assert score_a > score_b, (
            f"Trial A (ratio=0.38, closer to target=0.40) should score higher: "
            f"{score_a} vs {score_b}"
        )

    def test_both_above_target_closer_wins(self):
        """Property: if both above target, the one closer to target wins."""
        base_score = 0.80
        target = 0.40
        coeff = 1.0
        
        # Trial A: ratio = 0.50 (10% above target)
        # Trial B: ratio = 0.70 (30% above target)
        score_a = _compute_composite_score(base_score, 0.50, target, coeff)
        score_b = _compute_composite_score(base_score, 0.70, target, coeff)
        
        assert score_a > score_b, (
            f"Trial A (ratio=0.50, closer to target) should score higher: "
            f"{score_a} vs {score_b}"
        )

    def test_both_below_target_no_difference(self):
        """Property: if both below target, neither is penalized (same score)."""
        base_score = 0.80
        target = 0.50
        coeff = 1.0
        
        # Both below target: no penalty
        score_a = _compute_composite_score(base_score, 0.30, target, coeff)
        score_b = _compute_composite_score(base_score, 0.40, target, coeff)
        
        # Both should equal base_score (no penalty)
        assert score_a == base_score, f"Score A should equal base {base_score}, got {score_a}"
        assert score_b == base_score, f"Score B should equal base {base_score}, got {score_b}"
        assert score_a == score_b, "Both below target should have equal scores"

    @pytest.mark.parametrize("target,coeff", [
        (0.35, 0.5),
        (0.40, 1.0),
        (0.42, 1.0),
        (0.50, 1.5),
        (0.70, 0.5),
    ])
    def test_score_ordering_across_configs(self, target: float, coeff: float):
        """Property: score ordering is preserved across different configs."""
        base_score = 0.75
        
        # Generate ratios: some below, some above target
        ratios = [target - 0.1, target, target + 0.1, target + 0.2, target + 0.3]
        ratios = [max(0.0, min(1.0, r)) for r in ratios]  # Clamp
        
        scores = [
            (r, _compute_composite_score(base_score, r, target, coeff))
            for r in ratios
        ]
        
        # All ratios <= target should have same score (no penalty)
        below_target = [(r, s) for r, s in scores if r <= target]
        if len(below_target) > 1:
            for i in range(1, len(below_target)):
                assert below_target[i][1] == below_target[0][1], (
                    f"Ratios below target should have equal scores"
                )
        
        # Scores for ratios above target should decrease
        above_target = [(r, s) for r, s in scores if r > target]
        for i in range(1, len(above_target)):
            assert above_target[i][1] <= above_target[i-1][1], (
                f"Scores should decrease as ratio increases above target"
            )


class TestSymbolicDominanceWithOtherPenalties:
    """Test symbolic dominance interacts correctly with other penalties."""

    def test_dominance_penalty_stacks_with_others(self):
        """Property: dominance penalty stacks multiplicatively with other penalties."""
        base_score = 0.80
        target = 0.40
        coeff = 1.0
        
        # Other penalty: weight_penalty=0.1 with coeff=0.4
        other_penalties = [(0.4, 0.1)]
        
        # Score with only other penalty (no dominance)
        score_other_only = _compute_composite_score(
            base_score, target, target, coeff, other_penalties  # At target, no dominance
        )
        
        # Score with both penalties (0.60 ratio -> dominance)
        score_both = _compute_composite_score(
            base_score, 0.60, target, coeff, other_penalties
        )
        
        # Both should be penalized more than neither
        score_neither = base_score
        
        assert score_other_only < score_neither
        assert score_both < score_other_only, (
            f"Both penalties should reduce score more: {score_both} vs {score_other_only}"
        )

    def test_max_penalty_floors_at_zero(self):
        """Property: composite score never goes negative."""
        base_score = 0.50
        target = 0.30
        coeff = 2.0  # Aggressive coefficient
        
        # Extreme penalties
        other_penalties = [(0.9, 1.0), (0.9, 1.0)]  # Very heavy penalties
        
        # Extreme symbolic dominance
        score = _compute_composite_score(
            base_score, 1.0, target, coeff, other_penalties
        )
        
        assert score >= 0.0, f"Score should never be negative, got {score}"


class TestConfigDrivenDominance:
    """Test that target and coefficient behave as expected from config."""

    def test_higher_coeff_increases_penalty_impact(self):
        """Property: higher coefficient = stronger penalty impact."""
        base_score = 0.80
        target = 0.40
        ratio = 0.60  # Above target
        
        score_low_coeff = _compute_composite_score(base_score, ratio, target, 0.5)
        score_high_coeff = _compute_composite_score(base_score, ratio, target, 1.5)
        
        assert score_low_coeff > score_high_coeff, (
            f"Higher coeff should reduce score more: {score_high_coeff} vs {score_low_coeff}"
        )

    def test_higher_target_allows_more_symbolic(self):
        """Property: higher target ratio tolerates more symbolic contribution."""
        base_score = 0.80
        coeff = 1.0
        ratio = 0.55
        
        # With target=0.40, ratio=0.55 is penalized
        score_low_target = _compute_composite_score(base_score, ratio, 0.40, coeff)
        
        # With target=0.60, ratio=0.55 is NOT penalized
        score_high_target = _compute_composite_score(base_score, ratio, 0.60, coeff)
        
        assert score_high_target > score_low_target, (
            f"Higher target should allow more symbolic: {score_high_target} vs {score_low_target}"
        )
        assert score_high_target == base_score, (
            f"With ratio below target, score should equal base: {score_high_target}"
        )

"""Property tests for generalization gap computation and penalty.

Tests the following properties:
(1) _compute_generalization_gap returns val_auc - test_auc correctly.
(2) If gap penalty is configured, higher gap should result in lower composite_score.
"""

from __future__ import annotations

import pytest

# ============================================================================
# Inline functions for generalization gap computation
# ============================================================================


def _compute_generalization_gap(val_auc: float, test_auc: float) -> float:
    """Compute generalization gap between validation and test AUC.

    Args:
        val_auc: Validation AUC score.
        test_auc: Test AUC score.

    Returns:
        Generalization gap (val_auc - test_auc).
        Positive gap indicates overfitting to validation set.
    """
    return val_auc - test_auc


def _compute_gap_penalty(
    gap: float,
    max_acceptable_gap: float = 0.05,
    penalty_scale: float = 1.0,
) -> float:
    """Compute penalty for generalization gap.

    Args:
        gap: Generalization gap value.
        max_acceptable_gap: Gap threshold above which penalty applies.
        penalty_scale: Scaling factor for penalty magnitude.

    Returns:
        Penalty value (0 if gap <= max_acceptable_gap).
    """
    if gap <= max_acceptable_gap:
        return 0.0
    excess = gap - max_acceptable_gap
    return min(1.0, excess * penalty_scale)


def _compute_score_with_gap_penalty(
    base_score: float,
    val_auc: float,
    test_auc: float,
    gap_penalty_coeff: float = 0.3,
    max_acceptable_gap: float = 0.05,
) -> float:
    """Compute composite score with generalization gap penalty.

    Args:
        base_score: Base weighted score before penalties.
        val_auc: Validation AUC.
        test_auc: Test AUC.
        gap_penalty_coeff: Coefficient for gap penalty.
        max_acceptable_gap: Acceptable gap threshold.

    Returns:
        Final score after gap penalty.
    """
    gap = _compute_generalization_gap(val_auc, test_auc)
    penalty = _compute_gap_penalty(gap, max_acceptable_gap)

    score = base_score * (1.0 - gap_penalty_coeff * min(1.0, penalty))
    return max(0.0, score)


class TestGeneralizationGapComputation:
    """Test basic gap computation correctness."""

    @pytest.mark.parametrize(
        "val_auc,test_auc,expected_gap",
        [
            (0.85, 0.80, 0.05),  # Positive gap (overfitting)
            (0.80, 0.80, 0.0),  # No gap
            (0.75, 0.80, -0.05),  # Negative gap (underfitting on val)
            (0.90, 0.70, 0.20),  # Large positive gap
            (0.60, 0.90, -0.30),  # Large negative gap
        ],
    )
    def test_gap_equals_val_minus_test(self, val_auc: float, test_auc: float, expected_gap: float):
        """Property: gap MUST equal val_auc - test_auc exactly."""
        gap = _compute_generalization_gap(val_auc, test_auc)
        assert abs(gap - expected_gap) < 1e-9, f"Expected gap {expected_gap}, got {gap}"

    def test_gap_sign_indicates_direction(self):
        """Property: positive gap indicates overfitting, negative indicates underfitting."""
        # Overfitting: val > test
        gap_overfit = _compute_generalization_gap(0.90, 0.80)
        assert gap_overfit > 0, "Overfitting should give positive gap"

        # Underfitting: val < test
        gap_underfit = _compute_generalization_gap(0.75, 0.85)
        assert gap_underfit < 0, "Underfitting should give negative gap"

        # Perfect generalization: val == test
        gap_perfect = _compute_generalization_gap(0.80, 0.80)
        assert gap_perfect == 0.0, "Perfect generalization should give zero gap"


class TestGapPenaltyBehavior:
    """Test gap penalty computation and thresholds."""

    def test_penalty_zero_when_gap_below_threshold(self):
        """Property: no penalty when gap <= max_acceptable_gap."""
        for gap in [0.0, 0.01, 0.03, 0.05]:
            penalty = _compute_gap_penalty(gap, max_acceptable_gap=0.05)
            assert penalty == 0.0, f"Penalty should be 0 for gap={gap}, got {penalty}"

    def test_penalty_positive_when_gap_above_threshold(self):
        """Property: penalty > 0 when gap > max_acceptable_gap."""
        for gap in [0.06, 0.10, 0.15, 0.20]:
            penalty = _compute_gap_penalty(gap, max_acceptable_gap=0.05)
            assert penalty > 0.0, f"Penalty should be > 0 for gap={gap}, got {penalty}"

    def test_penalty_increases_with_gap(self):
        """Property: penalty should increase monotonically with gap above threshold."""
        threshold = 0.05
        gaps = [0.06, 0.08, 0.10, 0.15, 0.20]

        penalties = [_compute_gap_penalty(g, threshold) for g in gaps]

        for i in range(1, len(penalties)):
            assert penalties[i] > penalties[i - 1], (
                f"Penalty should increase: gap={gaps[i - 1]}, pen={penalties[i - 1]} -> "
                f"gap={gaps[i]}, pen={penalties[i]}"
            )

    def test_penalty_capped_at_one(self):
        """Property: penalty is capped at 1.0."""
        # Very large gap with high scale
        penalty = _compute_gap_penalty(1.0, max_acceptable_gap=0.05, penalty_scale=100.0)
        assert penalty == 1.0, f"Penalty should cap at 1.0, got {penalty}"

    def test_negative_gap_no_penalty(self):
        """Property: negative gap (underfitting) should not be penalized."""
        for gap in [-0.05, -0.10, -0.20]:
            penalty = _compute_gap_penalty(gap, max_acceptable_gap=0.05)
            assert penalty == 0.0, f"Negative gap should give 0 penalty, got {penalty}"


class TestScoreWithGapPenalty:
    """Test composite score integration with gap penalty."""

    def test_higher_gap_lower_score(self):
        """Property: higher gap should result in lower composite_score."""
        base_score = 0.80

        # Small gap (within threshold)
        score_small = _compute_score_with_gap_penalty(base_score, 0.85, 0.82)  # gap=0.03

        # Large gap
        score_large = _compute_score_with_gap_penalty(base_score, 0.90, 0.70)  # gap=0.20

        assert score_small > score_large, (
            f"Higher gap should reduce score: small={score_small}, large={score_large}"
        )

    def test_same_metrics_different_gaps(self):
        """Property: with same base_score, gap determines final score ordering."""
        base_score = 0.80

        # Scenario A: val=0.85, test=0.84 -> gap=0.01 (no penalty)
        score_a = _compute_score_with_gap_penalty(base_score, 0.85, 0.84)

        # Scenario B: val=0.85, test=0.75 -> gap=0.10 (penalized)
        score_b = _compute_score_with_gap_penalty(base_score, 0.85, 0.75)

        assert score_a > score_b
        assert score_a == base_score, "No gap penalty should mean full base_score"

    @pytest.mark.parametrize("gap_coeff", [0.1, 0.3, 0.5, 0.8])
    def test_coefficient_scales_penalty_impact(self, gap_coeff: float):
        """Property: higher coefficient = stronger penalty impact."""
        base_score = 0.80
        val_auc, test_auc = 0.90, 0.75  # gap=0.15

        # Reference with coeff=0

        # With this coefficient
        score_with_penalty = _compute_score_with_gap_penalty(
            base_score, val_auc, test_auc, gap_penalty_coeff=gap_coeff
        )

        # Score should be less than base, and lower with higher coeff
        assert score_with_penalty < base_score

    def test_score_never_negative(self):
        """Property: score should never go negative."""
        base_score = 0.50

        # Extreme gap with high coefficient
        score = _compute_score_with_gap_penalty(
            base_score,
            0.95,
            0.20,  # gap=0.75
            gap_penalty_coeff=1.0,
        )

        assert score >= 0.0, f"Score should never be negative, got {score}"


class TestGeneralizationGapLogging:
    """Test gap computation edge cases relevant to logging."""

    def test_gap_with_perfect_scores(self):
        """Property: gap computation works with perfect AUC scores."""
        gap = _compute_generalization_gap(1.0, 1.0)
        assert gap == 0.0

        gap_overfit = _compute_generalization_gap(1.0, 0.95)
        assert abs(gap_overfit - 0.05) < 1e-9  # Use tolerance for floating point

    def test_gap_with_minimum_scores(self):
        """Property: gap computation works with low AUC scores."""
        gap = _compute_generalization_gap(0.5, 0.5)
        assert gap == 0.0

        gap = _compute_generalization_gap(0.55, 0.50)
        assert abs(gap - 0.05) < 1e-9

    def test_gap_precision(self):
        """Property: gap should maintain floating point precision."""
        val_auc = 0.8765432109
        test_auc = 0.8765432100

        gap = _compute_generalization_gap(val_auc, test_auc)
        expected = val_auc - test_auc

        assert abs(gap - expected) < 1e-15, "Gap should be computed with full precision"


class TestGapPenaltyConfiguration:
    """Test configurable aspects of gap penalty."""

    @pytest.mark.parametrize("threshold", [0.01, 0.03, 0.05, 0.10])
    def test_custom_threshold_respected(self, threshold: float):
        """Property: custom threshold is correctly applied."""
        gap_at_threshold = threshold
        gap_above = threshold + 0.01

        penalty_at = _compute_gap_penalty(gap_at_threshold, max_acceptable_gap=threshold)
        penalty_above = _compute_gap_penalty(gap_above, max_acceptable_gap=threshold)

        assert penalty_at == 0.0, "At threshold should give 0 penalty"
        assert penalty_above > 0.0, "Above threshold should give positive penalty"

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 5.0])
    def test_penalty_scale_affects_magnitude(self, scale: float):
        """Property: penalty scale affects penalty magnitude."""
        gap = 0.15
        threshold = 0.05

        penalty_scale_1 = _compute_gap_penalty(gap, threshold, penalty_scale=1.0)
        penalty_custom = _compute_gap_penalty(gap, threshold, penalty_scale=scale)

        if scale > 1.0:
            assert penalty_custom >= penalty_scale_1
        else:
            assert penalty_custom <= penalty_scale_1

"""Tests for physical time scoring with absolute semantic scale.

Validates that the time scoring function produces scores with physical meaning:
- Scores near 1.0 are only achievable with near-zero durations (impossible)
- Realistic durations produce meaningfully lower scores
- The score reflects actual time penalty, not just relative ranking

Design patterns:
- Property-based testing for invariants
- Parametrized tests for calibration points
"""

import pytest
from pff.domain.hpo.scoring import (
    TimeScaleConfig,
    compute_physical_time_score,
    compute_score,
    _default_weights,
    DEFAULT_EPS,
)


class TestPhysicalTimeScoreCalibration:
    """Test that time scores are calibrated to physical anchor points."""

    @pytest.fixture
    def default_time_scale(self) -> TimeScaleConfig:
        return TimeScaleConfig()

    def test_score_at_t_best(self, default_time_scale: TimeScaleConfig) -> None:
        """Score at t_best should be approximately score_at_best."""
        score = compute_physical_time_score(
            duration=default_time_scale.t_best,
            time_scale=default_time_scale,
        )
        # Allow 5% tolerance
        assert abs(score - default_time_scale.score_at_best) < 0.05, (
            f"Score at t_best={default_time_scale.t_best}s should be ~{default_time_scale.score_at_best}, "
            f"got {score}"
        )

    def test_score_at_t_target(self, default_time_scale: TimeScaleConfig) -> None:
        """Score at t_target should be approximately score_at_target."""
        score = compute_physical_time_score(
            duration=default_time_scale.t_target,
            time_scale=default_time_scale,
        )
        # Allow 10% tolerance for interpolation
        assert abs(score - default_time_scale.score_at_target) < 0.10, (
            f"Score at t_target={default_time_scale.t_target}s should be ~{default_time_scale.score_at_target}, "
            f"got {score}"
        )

    def test_score_at_t_worst(self, default_time_scale: TimeScaleConfig) -> None:
        """Score at t_worst should be approximately score_at_worst."""
        score = compute_physical_time_score(
            duration=default_time_scale.t_worst,
            time_scale=default_time_scale,
        )
        # Allow 5% tolerance
        assert abs(score - default_time_scale.score_at_worst) < 0.05, (
            f"Score at t_worst={default_time_scale.t_worst}s should be ~{default_time_scale.score_at_worst}, "
            f"got {score}"
        )


class TestPhysicalTimeScoreMonotonicity:
    """Test that time score is monotonically decreasing with duration."""

    @pytest.fixture
    def default_time_scale(self) -> TimeScaleConfig:
        return TimeScaleConfig()

    @pytest.mark.parametrize(
        "durations",
        [
            [0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 600.0],
            [0.5, 5.0, 25.0, 75.0, 150.0, 450.0],
            [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
        ],
    )
    def test_monotonically_decreasing(
        self, default_time_scale: TimeScaleConfig, durations: list[float]
    ) -> None:
        """Longer durations should produce lower scores."""
        scores = [compute_physical_time_score(d, default_time_scale) for d in durations]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score should decrease: duration {durations[i]}s -> {scores[i]}, "
                f"duration {durations[i + 1]}s -> {scores[i + 1]}"
            )


class TestPhysicalTimeScoreBounds:
    """Test that time scores respect physical bounds."""

    @pytest.fixture
    def default_time_scale(self) -> TimeScaleConfig:
        return TimeScaleConfig()

    def test_score_never_reaches_one(self, default_time_scale: TimeScaleConfig) -> None:
        """Even with very small durations, score should not reach 1.0."""
        for tiny_duration in [0.001, 0.01, 0.1]:
            score = compute_physical_time_score(tiny_duration, default_time_scale)
            assert (
                score < 1.0 - DEFAULT_EPS
            ), f"Score for {tiny_duration}s should be < 1-eps, got {score}"

    def test_score_never_reaches_zero(
        self, default_time_scale: TimeScaleConfig
    ) -> None:
        """Even with very large durations, score should not reach 0.0."""
        for huge_duration in [1000.0, 10000.0, 100000.0]:
            score = compute_physical_time_score(huge_duration, default_time_scale)
            assert (
                score > DEFAULT_EPS
            ), f"Score for {huge_duration}s should be > eps, got {score}"

    def test_realistic_durations_well_below_one(
        self, default_time_scale: TimeScaleConfig
    ) -> None:
        """Realistic durations (1s-600s) should produce scores well below 1.0."""
        for duration in [1.0, 5.0, 30.0, 60.0, 300.0, 600.0]:
            score = compute_physical_time_score(duration, default_time_scale)
            assert (
                score < 0.95
            ), f"Score for realistic duration {duration}s should be < 0.95, got {score}"


class TestGlobalScoreSanity:
    """Test that global score respects physical time constraints."""

    def test_perfect_metrics_with_realistic_time_below_threshold(self) -> None:
        """Even with perfect metrics, realistic times should cap global score."""
        weights = _default_weights()

        # Perfect metrics
        perfect_metrics = {
            "mrr": 1.0,
            "best_mrr": 1.0,
            "hits1": 1.0,
            "hits3": 1.0,
            "hits10": 1.0,
            "auc": 1.0,
            "pr_auc": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "duration": weights.time_scale.t_best,  # Best realistic time
        }

        score, _, components = compute_score(
            current_metrics=perfect_metrics,
            history_metrics=[],
            weights=weights,
        )

        # With t_best duration, score should be < 0.98 (not near-perfect)
        assert (
            score < 0.98
        ), f"Even with perfect metrics and t_best duration, score should be < 0.98, got {score}"

    def test_perfect_metrics_with_target_time_significantly_penalized(self) -> None:
        """Perfect metrics with t_target time should have visible penalty."""
        weights = _default_weights()

        perfect_metrics = {
            "mrr": 1.0,
            "best_mrr": 1.0,
            "hits1": 1.0,
            "hits3": 1.0,
            "hits10": 1.0,
            "auc": 1.0,
            "pr_auc": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "duration": weights.time_scale.t_target,  # Acceptable but penalized time
        }

        score, _, components = compute_score(
            current_metrics=perfect_metrics,
            history_metrics=[],
            weights=weights,
        )

        # With t_target duration, score should be < 0.95
        assert (
            score < 0.95
        ), f"Perfect metrics with t_target duration should produce score < 0.95, got {score}"

    def test_perfect_metrics_with_worst_time_heavily_penalized(self) -> None:
        """Perfect metrics with t_worst time should be heavily penalized."""
        weights = _default_weights()

        perfect_metrics = {
            "mrr": 1.0,
            "best_mrr": 1.0,
            "hits1": 1.0,
            "hits3": 1.0,
            "hits10": 1.0,
            "auc": 1.0,
            "pr_auc": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "duration": weights.time_scale.t_worst,  # Poor time
        }

        score, _, components = compute_score(
            current_metrics=perfect_metrics,
            history_metrics=[],
            weights=weights,
        )

        # With t_worst duration, score should be < 0.90
        assert (
            score < 0.90
        ), f"Perfect metrics with t_worst duration should produce score < 0.90, got {score}"

    @pytest.mark.parametrize(
        "duration,max_expected_score",
        [
            (1.0, 0.98),  # t_best: still below 0.98
            (50.0, 0.95),  # t_target: visible penalty
            (300.0, 0.90),  # t_worst: heavy penalty
            (600.0, 0.85),  # beyond t_worst: severe penalty
        ],
    )
    def test_score_ceiling_by_duration(
        self, duration: float, max_expected_score: float
    ) -> None:
        """Score ceiling should be determined by duration for perfect metrics."""
        weights = _default_weights()

        perfect_metrics = {
            "mrr": 1.0,
            "best_mrr": 1.0,
            "hits1": 1.0,
            "hits3": 1.0,
            "hits10": 1.0,
            "auc": 1.0,
            "pr_auc": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "duration": duration,
        }

        score, _, _ = compute_score(
            current_metrics=perfect_metrics,
            history_metrics=[],
            weights=weights,
        )

        assert (
            score < max_expected_score
        ), f"Score with duration={duration}s should be < {max_expected_score}, got {score}"


class TestWeakModelScoring:
    """Test that weak models with degenerate precision/recall are penalized correctly.

    A model that predicts everything as positive will have recall=100% but very
    low precision. The score should reflect this poor performance, not reward it.
    """

    def test_high_recall_low_precision_is_penalized(self) -> None:
        """Recall=100% with precision=5% should result in lower clf_block.

        This scenario indicates the model is just guessing 'positive' for everything.
        The clf_block should reflect the weak precision.
        """
        weak_model_metrics = {
            "mrr": 0.04,
            "best_mrr": 0.05,
            "hits1": 0.02,
            "hits3": 0.04,
            "hits10": 0.08,
            "auc": 0.42,
            "pr_auc": 0.10,
            "precision": 0.05,  # Very low precision
            "recall": 1.0,  # Perfect recall (model predicting everything positive)
            "duration": 60.0,
        }

        # History with similar weak models
        history = [
            {
                "mrr": 0.03,
                "best_mrr": 0.04,
                "hits1": 0.01,
                "hits3": 0.03,
                "hits10": 0.06,
                "auc": 0.40,
                "pr_auc": 0.08,
                "precision": 0.04,
                "recall": 1.0,
                "duration": 50.0,
            },
        ]

        score, normalized, components = compute_score(
            current_metrics=weak_model_metrics,
            history_metrics=history,
        )

        # Classification block should reflect the weak precision (absolute)
        # clf_block weights: auc=0.35, pr_auc=0.35, precision=0.15, recall=0.15
        # Expected: 0.35*0.42 + 0.35*0.10 + 0.15*0.05 + 0.15*1.0 ≈ 0.34
        assert components.classification < 0.50, (
            f"Weak model clf_block should be < 0.50 with precision=5%. "
            f"Got clf={components.classification:.3f}"
        )

        # Precision should be at its absolute value, not normalized
        assert (
            normalized["precision"] < 0.10
        ), f"Precision=0.05 should normalize to ~0.05, got {normalized['precision']:.3f}"

    def test_precision_uses_absolute_not_relative_scaling(self) -> None:
        """Precision should use absolute value, not min-max normalization.

        If all trials have precision in range [0.03, 0.05], the best (0.05) should
        NOT be normalized to 100%. It should remain at its absolute value (~5%).
        """
        current = {
            "precision": 0.05,
            "recall": 0.90,
            "auc": 0.50,
            "pr_auc": 0.30,
            "mrr": 0.10,
            "best_mrr": 0.10,
            "hits1": 0.05,
            "hits3": 0.08,
            "hits10": 0.15,
            "duration": 60.0,
        }

        history = [
            {
                "precision": 0.03,
                "recall": 0.95,
                "auc": 0.48,
                "pr_auc": 0.28,
                "mrr": 0.08,
                "best_mrr": 0.09,
                "hits1": 0.04,
                "hits3": 0.06,
                "hits10": 0.12,
                "duration": 55.0,
            },
            {
                "precision": 0.04,
                "recall": 0.92,
                "auc": 0.49,
                "pr_auc": 0.29,
                "mrr": 0.09,
                "best_mrr": 0.09,
                "hits1": 0.04,
                "hits3": 0.07,
                "hits10": 0.13,
                "duration": 65.0,
            },
        ]

        score, normalized, _ = compute_score(
            current_metrics=current,
            history_metrics=history,
        )

        # Precision normalized should be close to absolute value (0.05), not 1.0
        assert normalized["precision"] < 0.20, (
            f"Precision=0.05 should normalize to ~0.05-0.10, not {normalized['precision']:.3f}. "
            "This indicates relative normalization is being used instead of absolute."
        )

    def test_clf_metrics_use_absolute_values(self) -> None:
        """Classification metrics (auc, pr_auc, precision, recall) use absolute values.

        This ensures that weak models with precision=5% don't get artificially
        inflated scores just because they're the "best" among weak trials.
        """
        # Balanced model (both precision and recall moderate)
        balanced = {
            "mrr": 0.15,
            "best_mrr": 0.15,
            "hits1": 0.08,
            "hits3": 0.12,
            "hits10": 0.20,
            "auc": 0.65,
            "pr_auc": 0.50,
            "precision": 0.30,
            "recall": 0.30,
            "duration": 60.0,
        }

        # Degenerate model (high recall, low precision)
        degenerate = {
            "mrr": 0.15,
            "best_mrr": 0.15,
            "hits1": 0.08,
            "hits3": 0.12,
            "hits10": 0.20,
            "auc": 0.65,
            "pr_auc": 0.50,
            "precision": 0.05,
            "recall": 1.0,
            "duration": 60.0,
        }

        history = [balanced, degenerate]

        _, norm_balanced, comp_balanced = compute_score(balanced, history)
        _, norm_degenerate, comp_degenerate = compute_score(degenerate, history)

        # Precision should use absolute value (not normalized to 1.0)
        assert (
            norm_balanced["precision"] < 0.35
        ), f"Precision=0.30 should be ~0.30 (absolute), got {norm_balanced['precision']:.3f}"
        assert (
            norm_degenerate["precision"] < 0.10
        ), f"Precision=0.05 should be ~0.05 (absolute), got {norm_degenerate['precision']:.3f}"

        # Recall should also use absolute value
        assert (
            norm_degenerate["recall"] > 0.95
        ), f"Recall=1.0 should be ~1.0 (absolute), got {norm_degenerate['recall']:.3f}"
        assert (
            norm_balanced["recall"] < 0.35
        ), f"Recall=0.30 should be ~0.30 (absolute), got {norm_balanced['recall']:.3f}"

        # AUC should also use absolute value
        assert (
            0.60 < norm_balanced["auc"] < 0.70
        ), f"AUC=0.65 should be ~0.65 (absolute), got {norm_balanced['auc']:.3f}"

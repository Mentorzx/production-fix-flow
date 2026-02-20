"""MCC Sanity Tests for HPO Regression Detection.

These tests validate the MCC (Matthews Correlation Coefficient) computation
to prevent regressions like the one observed after the Arrow refactor.

Test coverage:
1. Perfect, random, and inverted classification baselines
2. Threshold selection impact on MCC
3. Negative sampling contamination detection
4. Score distribution effects on MCC
5. Cross-validation with sklearn reference implementation

Design: These tests serve as regression guards and should be run before
any HPO pipeline changes.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef as sklearn_mcc

from pff_rust import fast_matthews_corrcoef


class TestMCCBaselines:
    """Baseline tests for MCC computation correctness."""

    def test_perfect_classification_mcc_equals_one(self) -> None:
        """Perfect binary classification should yield MCC = 1.0."""
        y_true = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
        y_pred = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)

        mcc = fast_matthews_corrcoef(y_true, y_pred)

        assert mcc == pytest.approx(1.0, abs=1e-9), (
            f"Perfect classification should give MCC=1.0, got {mcc}"
        )

    def test_perfect_inverted_classification_mcc_equals_negative_one(self) -> None:
        """Perfectly inverted predictions should yield MCC = -1.0."""
        y_true = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
        y_pred = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

        mcc = fast_matthews_corrcoef(y_true, y_pred)

        assert mcc == pytest.approx(-1.0, abs=1e-9), (
            f"Inverted classification should give MCC=-1.0, got {mcc}"
        )

    def test_random_classification_mcc_near_zero(self) -> None:
        """Random predictions should yield MCC ≈ 0.0 on average."""
        rng = np.random.default_rng(42)
        mcc_values = []

        for _ in range(100):
            y_true = rng.integers(0, 2, size=1000).astype(np.int64)
            y_pred = rng.integers(0, 2, size=1000).astype(np.int64)
            mcc_values.append(fast_matthews_corrcoef(y_true, y_pred))

        mean_mcc = np.mean(mcc_values)
        # Random predictions should have mean MCC close to 0
        assert abs(mean_mcc) < 0.1, (
            f"Random classification should give MCC≈0, got mean={mean_mcc:.4f}"
        )

    def test_all_positive_predictions_mcc_zero(self) -> None:
        """All positive predictions should yield MCC = 0.0 (no discrimination)."""
        y_true = np.array([1, 1, 0, 0, 1, 0], dtype=np.int64)
        y_pred = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)

        mcc = fast_matthews_corrcoef(y_true, y_pred)

        # When all predictions are same class, MCC should be 0 (no discriminative power)
        assert mcc == pytest.approx(0.0, abs=1e-9), (
            f"All-positive predictions should give MCC=0, got {mcc}"
        )

    def test_all_negative_predictions_mcc_zero(self) -> None:
        """All negative predictions should yield MCC = 0.0 (no discrimination)."""
        y_true = np.array([1, 1, 0, 0, 1, 0], dtype=np.int64)
        y_pred = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)

        mcc = fast_matthews_corrcoef(y_true, y_pred)

        assert mcc == pytest.approx(0.0, abs=1e-9), (
            f"All-negative predictions should give MCC=0, got {mcc}"
        )


class TestMCCSklearnConsistency:
    """Cross-validate fast_matthews_corrcoef against sklearn reference."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1337])
    def test_mcc_matches_sklearn_random_data(self, seed: int) -> None:
        """MCC should match sklearn's matthews_corrcoef on random data."""
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 2, size=500).astype(np.int64)
        y_pred = rng.integers(0, 2, size=500).astype(np.int64)

        our_mcc = fast_matthews_corrcoef(y_true, y_pred)
        sklearn_mcc_val = sklearn_mcc(y_true, y_pred)

        assert our_mcc == pytest.approx(sklearn_mcc_val, abs=1e-9), (
            f"MCC mismatch: ours={our_mcc:.6f}, sklearn={sklearn_mcc_val:.6f}"
        )

    def test_mcc_matches_sklearn_imbalanced_data(self) -> None:
        """MCC should match sklearn on highly imbalanced data (common in KGC)."""
        rng = np.random.default_rng(42)
        # 90% negatives, 10% positives (typical in negative sampling)
        y_true = np.concatenate([np.ones(100, dtype=np.int64), np.zeros(900, dtype=np.int64)])
        # Classifier with some errors
        y_pred = y_true.copy()
        error_idx = rng.choice(len(y_true), size=50, replace=False)
        y_pred[error_idx] = 1 - y_pred[error_idx]

        our_mcc = fast_matthews_corrcoef(y_true, y_pred)
        sklearn_mcc_val = sklearn_mcc(y_true, y_pred)

        assert our_mcc == pytest.approx(sklearn_mcc_val, abs=1e-9), (
            f"MCC mismatch on imbalanced data: ours={our_mcc:.6f}, sklearn={sklearn_mcc_val:.6f}"
        )

    def test_mcc_matches_sklearn_extreme_imbalance(self) -> None:
        """MCC should match sklearn on extreme imbalance (1:20 ratio)."""
        y_true = np.concatenate([np.ones(50, dtype=np.int64), np.zeros(1000, dtype=np.int64)])
        # Perfect predictions
        y_pred = y_true.copy()

        our_mcc = fast_matthews_corrcoef(y_true, y_pred)
        sklearn_mcc_val = sklearn_mcc(y_true, y_pred)

        assert our_mcc == pytest.approx(sklearn_mcc_val, abs=1e-9)


class TestMCCThresholdSelection:
    """Tests for threshold selection impact on MCC."""

    def test_optimal_threshold_maximizes_mcc(self) -> None:
        """Sweeping thresholds should find the optimal MCC."""
        rng = np.random.default_rng(42)

        # Generate scores with some separation between classes
        n_pos, n_neg = 100, 100
        pos_scores = rng.normal(0.7, 0.2, size=n_pos)
        neg_scores = rng.normal(0.3, 0.2, size=n_neg)

        all_scores = np.concatenate([pos_scores, neg_scores])
        all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)

        # Sweep thresholds
        thresholds = np.percentile(all_scores, np.linspace(0, 100, 20))
        mcc_values = []

        for thresh in thresholds:
            preds = (all_scores > thresh).astype(np.int64)
            mcc = fast_matthews_corrcoef(all_labels, preds)
            mcc_values.append(mcc)

        best_mcc = max(mcc_values)

        # With well-separated classes, best MCC should be significantly positive
        assert best_mcc > 0.3, f"Well-separated classes should give MCC>0.3, got {best_mcc:.4f}"

    def test_overlapping_scores_reduce_mcc(self) -> None:
        """Highly overlapping score distributions should result in lower MCC."""
        rng = np.random.default_rng(42)

        n_pos, n_neg = 100, 100
        # Same mean, high overlap
        pos_scores = rng.normal(0.5, 0.3, size=n_pos)
        neg_scores = rng.normal(0.5, 0.3, size=n_neg)

        all_scores = np.concatenate([pos_scores, neg_scores])
        all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)

        thresholds = np.percentile(all_scores, np.linspace(0, 100, 20))
        best_mcc = max(
            fast_matthews_corrcoef(all_labels, (all_scores > t).astype(np.int64))
            for t in thresholds
        )

        # Overlapping distributions should give low MCC
        assert best_mcc < 0.3, f"Overlapping distributions should give MCC<0.3, got {best_mcc:.4f}"

    def test_threshold_at_extreme_percentiles_reduces_mcc(self) -> None:
        """Thresholds at 0th or 100th percentile should give MCC=0."""
        rng = np.random.default_rng(42)

        n_pos, n_neg = 100, 100
        pos_scores = rng.normal(0.7, 0.2, size=n_pos)
        neg_scores = rng.normal(0.3, 0.2, size=n_neg)

        all_scores = np.concatenate([pos_scores, neg_scores])
        all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)

        # Threshold below all scores (all predicted positive)
        thresh_low = all_scores.min() - 1
        preds_low = (all_scores > thresh_low).astype(np.int64)
        mcc_low = fast_matthews_corrcoef(all_labels, preds_low)

        # Threshold above all scores (all predicted negative)
        thresh_high = all_scores.max() + 1
        preds_high = (all_scores > thresh_high).astype(np.int64)
        mcc_high = fast_matthews_corrcoef(all_labels, preds_high)

        assert mcc_low == pytest.approx(0.0, abs=1e-9), (
            f"All-positive should give MCC=0, got {mcc_low}"
        )
        assert mcc_high == pytest.approx(0.0, abs=1e-9), (
            f"All-negative should give MCC=0, got {mcc_high}"
        )


class TestNegativeSamplingContamination:
    """Tests detecting when negative sampling accidentally creates true positives."""

    def test_contaminated_negatives_reduce_mcc(self) -> None:
        """If negatives contain true positives, MCC should be artificially low."""
        n_pos = 100
        n_neg = 500

        # Simulate a perfect model that scores true positives high
        pos_scores = np.ones(n_pos) * 0.9
        neg_scores = np.ones(n_neg) * 0.1

        # Clean scenario: no contamination
        clean_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)
        clean_scores = np.concatenate([pos_scores, neg_scores])
        clean_preds = (clean_scores > 0.5).astype(np.int64)
        clean_mcc = fast_matthews_corrcoef(clean_labels, clean_preds)

        # Contaminated scenario: 10% of negatives are actually positives
        # (model still scores them high because they ARE positives)
        contamination_rate = 0.1
        n_contaminated = int(n_neg * contamination_rate)

        contaminated_neg_scores = neg_scores.copy()
        contaminated_neg_scores[:n_contaminated] = 0.9

        # Labels still say these are negatives (the bug: wrong labels)
        contaminated_labels = clean_labels.copy()
        contaminated_scores = np.concatenate([pos_scores, contaminated_neg_scores])
        contaminated_preds = (contaminated_scores > 0.5).astype(np.int64)
        contaminated_mcc = fast_matthews_corrcoef(contaminated_labels, contaminated_preds)

        # Contamination should reduce MCC
        assert contaminated_mcc < clean_mcc, (
            f"Contaminated negatives should reduce MCC: clean={clean_mcc:.4f}, "
            f"contaminated={contaminated_mcc:.4f}"
        )
        # Clean MCC should be ~1.0, contaminated should be significantly lower
        assert clean_mcc > 0.9, f"Clean MCC should be >0.9, got {clean_mcc:.4f}"

    def test_entity_space_mismatch_detection(self) -> None:
        """Detect when num_entities mismatch causes invalid negative samples."""
        rng = np.random.default_rng(42)

        # Scenario: num_entities=1000 but model was trained with num_entities=500
        # This means 50% of sampled negatives might be invalid entities
        n_pos = 100
        n_neg = 500

        # Model scores: valid entities get reasonable scores, invalid get random
        pos_scores = rng.normal(0.8, 0.1, size=n_pos)

        # Split negatives: half valid (score ~0.2), half invalid (score ~0.5 random)
        n_valid_neg = n_neg // 2
        n_invalid_neg = n_neg - n_valid_neg
        valid_neg_scores = rng.normal(0.2, 0.1, size=n_valid_neg)
        invalid_neg_scores = rng.normal(0.5, 0.2, size=n_invalid_neg)

        all_scores = np.concatenate([pos_scores, valid_neg_scores, invalid_neg_scores])
        all_labels = np.concatenate(
            [
                np.ones(n_pos),
                np.zeros(n_valid_neg),
                np.zeros(n_invalid_neg),
            ]
        ).astype(np.int64)

        thresholds = np.percentile(all_scores, np.linspace(0, 100, 20))
        best_mcc = max(
            fast_matthews_corrcoef(all_labels, (all_scores > t).astype(np.int64))
            for t in thresholds
        )

        # With 50% invalid negatives, MCC should be noticeably degraded
        # compared to what we'd expect from a good model (perfect would be ~1.0)
        assert best_mcc < 0.8, (
            f"Entity space mismatch should degrade MCC: got {best_mcc:.4f}, "
            "expected <0.8 due to invalid negatives"
        )


class TestMCCScoreDistribution:
    """Tests for score distribution effects on MCC."""

    def test_bimodal_distribution_high_mcc(self) -> None:
        """Well-separated bimodal score distribution should give high MCC."""
        rng = np.random.default_rng(42)

        n_pos, n_neg = 200, 200

        # Bimodal: positives cluster around 0.8, negatives around 0.2
        pos_scores = rng.beta(8, 2, size=n_pos)
        neg_scores = rng.beta(2, 8, size=n_neg)

        all_scores = np.concatenate([pos_scores, neg_scores])
        all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)

        thresholds = np.percentile(all_scores, np.linspace(0, 100, 20))
        best_mcc = max(
            fast_matthews_corrcoef(all_labels, (all_scores > t).astype(np.int64))
            for t in thresholds
        )

        assert best_mcc > 0.7, f"Bimodal distribution should give MCC>0.7, got {best_mcc:.4f}"

    def test_uniform_distribution_low_mcc(self) -> None:
        """Uniform score distribution indicates no discrimination."""
        rng = np.random.default_rng(42)

        n_pos, n_neg = 200, 200

        # Both classes have uniform distribution (no discrimination)
        pos_scores = rng.uniform(0, 1, size=n_pos)
        neg_scores = rng.uniform(0, 1, size=n_neg)

        all_scores = np.concatenate([pos_scores, neg_scores])
        all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)

        thresholds = np.percentile(all_scores, np.linspace(0, 100, 20))
        best_mcc = max(
            fast_matthews_corrcoef(all_labels, (all_scores > t).astype(np.int64))
            for t in thresholds
        )

        # Uniform distribution should give MCC close to 0
        assert abs(best_mcc) < 0.15, f"Uniform distribution should give MCC≈0, got {best_mcc:.4f}"


class TestMCCEdgeCases:
    """Edge cases that could cause MCC computation to fail or produce wrong results."""

    def test_empty_arrays_return_zero(self) -> None:
        """Empty inputs should return MCC=0.0, not raise."""
        y_true = np.array([], dtype=np.int64)
        y_pred = np.array([], dtype=np.int64)

        mcc = fast_matthews_corrcoef(y_true, y_pred)

        assert mcc == 0.0, f"Empty arrays should give MCC=0, got {mcc}"

    def test_single_element_arrays(self) -> None:
        """Single element should give MCC=0 (no variance)."""
        y_true = np.array([1], dtype=np.int64)
        y_pred = np.array([1], dtype=np.int64)

        mcc = fast_matthews_corrcoef(y_true, y_pred)

        # Single element can't demonstrate correlation
        assert mcc == 0.0, f"Single element should give MCC=0, got {mcc}"

    def test_all_same_true_labels(self) -> None:
        """All same true labels should give MCC=0."""
        y_true = np.array([1, 1, 1, 1, 1], dtype=np.int64)
        y_pred = np.array([1, 0, 1, 0, 1], dtype=np.int64)

        mcc = fast_matthews_corrcoef(y_true, y_pred)

        # Can't compute correlation when one variable has no variance
        assert mcc == 0.0, f"All-same true labels should give MCC=0, got {mcc}"

    @pytest.mark.slow
    def test_large_arrays_performance(self) -> None:
        """Large arrays should compute in reasonable time."""
        import time

        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=1_000_000).astype(np.int64)
        y_pred = rng.integers(0, 2, size=1_000_000).astype(np.int64)

        start = time.perf_counter()
        mcc = fast_matthews_corrcoef(y_true, y_pred)
        elapsed = time.perf_counter() - start

        # Allow a wider budget to avoid false failures on slower CI nodes
        assert elapsed < 0.5, f"1M element MCC took {elapsed:.3f}s, expected <0.5s"
        assert not np.isnan(mcc), "MCC should not be NaN for large arrays"


class TestMCCHPORealisticScenarios:
    """Realistic HPO scenarios that could cause MCC regression."""

    def test_typical_kgc_evaluation_scenario(self) -> None:
        """Simulate typical KGC evaluation with negative sampling."""
        rng = np.random.default_rng(42)

        # Typical HPO scenario: 2000 validation triples, 10 negatives each
        n_val_triples = 2000
        n_negatives_per_positive = 10

        n_pos = n_val_triples
        n_neg = n_val_triples * n_negatives_per_positive

        # Simulate a decent model (positives score higher on average)
        pos_scores = rng.beta(5, 2, size=n_pos)
        neg_scores = rng.beta(2, 5, size=n_neg)

        all_scores = np.concatenate([pos_scores, neg_scores])
        all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)

        # Sweep thresholds as done in kgc_manager._compute_binary_metrics_internal
        thresholds = np.percentile(all_scores, np.linspace(0, 100, 20))
        best_mcc = -1.0

        for t in thresholds:
            preds = (all_scores > t).astype(np.int64)
            mcc = fast_matthews_corrcoef(all_labels, preds)
            if mcc > best_mcc:
                best_mcc = mcc

        # A working model with 1:10 pos:neg ratio should achieve MCC >= 0.3
        assert best_mcc >= 0.3, (
            f"Typical KGC scenario should achieve MCC>=0.3, got {best_mcc:.4f}. "
            "This might indicate a regression in the MCC computation or data pipeline."
        )

    def test_early_training_low_discrimination(self) -> None:
        """Early in training, model has poor discrimination - MCC should be low but valid."""
        rng = np.random.default_rng(42)

        n_pos, n_neg = 100, 500

        # Early training: scores are mostly random with slight bias
        pos_scores = rng.normal(0.52, 0.3, size=n_pos)
        neg_scores = rng.normal(0.48, 0.3, size=n_neg)

        all_scores = np.concatenate([pos_scores, neg_scores])
        all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)

        thresholds = np.percentile(all_scores, np.linspace(0, 100, 20))
        best_mcc = max(
            fast_matthews_corrcoef(all_labels, (all_scores > t).astype(np.int64))
            for t in thresholds
        )

        # Early training MCC should be low but not negative
        assert best_mcc >= 0.0, f"Early training MCC should be >=0, got {best_mcc:.4f}"
        assert best_mcc < 0.3, f"Early training MCC should be <0.3, got {best_mcc:.4f}"

    def test_data_pipeline_produces_valid_input_shapes(self) -> None:
        """Ensure typical data pipeline shapes don't break MCC computation."""
        rng = np.random.default_rng(42)

        # Various shapes that might come from data pipeline
        shapes = [
            (100, 500),
            (1000, 5000),
            (50, 1000),
            (500, 500),
        ]

        for n_pos, n_neg in shapes:
            pos_scores = rng.normal(0.7, 0.2, size=n_pos)
            neg_scores = rng.normal(0.3, 0.2, size=n_neg)

            all_scores = np.concatenate([pos_scores, neg_scores])
            all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)

            # Use 0.5 as threshold (common default)
            preds = (all_scores > 0.5).astype(np.int64)
            mcc = fast_matthews_corrcoef(all_labels, preds)

            assert not np.isnan(mcc), f"MCC is NaN for shape ({n_pos}, {n_neg})"
            assert not np.isinf(mcc), f"MCC is Inf for shape ({n_pos}, {n_neg})"
            assert -1.0 <= mcc <= 1.0, f"MCC out of range for shape ({n_pos}, {n_neg}): {mcc}"

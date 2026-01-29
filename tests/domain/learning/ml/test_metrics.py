from __future__ import annotations

import numpy as np

from pff.domain.learning.ml.metrics import BinaryMetricsInputs, compute_binary_metrics


class _Backend:
    @staticmethod
    def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

    @staticmethod
    def auc(x: np.ndarray, y: np.ndarray) -> float:
        # Trapezoidal rule for sorted x
        return float(np.trapz(y, x))

    @staticmethod
    def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
        # Simple AP approximation: mean score over positives
        positives = y_true == 1
        if not np.any(positives):
            return 0.0
        return float(np.mean(y_score[positives]))

    @staticmethod
    def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if denom == 0:
            return 0.0
        return float((tp * tn - fp * fn) / np.sqrt(denom))

    @staticmethod
    def precision_recall_curve(
        y_true: np.ndarray, y_score: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Minimal monotonic curve for deterministic tests.
        thresholds = np.array([0.3, 0.6], dtype=np.float64)
        precisions = np.array([0.7, 0.8, 1.0], dtype=np.float64)
        recalls = np.array([1.0, 0.6, 0.2], dtype=np.float64)
        return precisions, recalls, thresholds

    @staticmethod
    def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
        # Simple rank-based proxy (not exact, but deterministic for tests)
        order = np.argsort(y_score)
        ranks = np.argsort(order) + 1
        pos = y_true == 1
        n_pos = int(np.sum(pos))
        n_neg = int(np.sum(~pos))
        if n_pos == 0 or n_neg == 0:
            return 0.5
        sum_ranks = float(np.sum(ranks[pos]))
        return (sum_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def test_compute_binary_metrics_outputs_expected_keys():
    labels = np.array([1, 0, 1, 0], dtype=np.int64)
    scores = np.array([0.9, 0.2, 0.7, 0.1], dtype=np.float64)
    metrics = compute_binary_metrics(
        BinaryMetricsInputs(labels=labels, prob_scores=scores),
        backend=_Backend(),
    )

    for key in [
        "auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "decision_threshold",
        "mcc",
        "accuracy",
        "ap",
        "tp",
        "tn",
        "fp",
        "fn",
        "vp",
        "vn",
    ]:
        assert key in metrics


def test_compute_binary_metrics_empty_inputs():
    metrics = compute_binary_metrics(
        BinaryMetricsInputs(
            labels=np.array([], dtype=np.int64),
            prob_scores=np.array([], dtype=np.float64),
        ),
        backend=_Backend(),
    )
    assert metrics == {}

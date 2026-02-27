"""Pure metric computation helpers for ML evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pff_rust import compute_ece


class BinaryMetricsBackend(Protocol):
    """Represent BinaryMetricsBackend."""

    def accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Execute accuracy score.

        Args:
            y_true: Input value used by this callable.
            y_pred: Input value used by this callable.
        """
        ...

    def auc(self, x: np.ndarray, y: np.ndarray) -> float:
        """Execute auc.

        Args:
            x: Input value used by this callable.
            y: Input value used by this callable.
        """
        ...

    def average_precision_score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Execute average precision score.

        Args:
            y_true: Input value used by this callable.
            y_score: Input value used by this callable.
        """
        ...

    def matthews_corrcoef(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Execute matthews corrcoef.

        Args:
            y_true: Input value used by this callable.
            y_pred: Input value used by this callable.
        """
        ...

    def precision_recall_curve(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute precision recall curve.

        Args:
            y_true: Input value used by this callable.
            y_score: Input value used by this callable.
        """
        ...

    def roc_auc_score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Execute roc auc score.

        Args:
            y_true: Input value used by this callable.
            y_score: Input value used by this callable.
        """
        ...


@dataclass(frozen=True)
class BinaryMetricsInputs:
    """Represent BinaryMetricsInputs."""

    labels: np.ndarray
    prob_scores: np.ndarray
    thresholds_from_pr: bool = True
    decision_threshold: float | None = None
    n_bins: int = 15


def compute_binary_metrics(
    inputs: BinaryMetricsInputs,
    backend: BinaryMetricsBackend,
) -> dict[str, float]:
    """Execute compute binary metrics.



    Args:

        inputs: Input value used by this callable.

        backend: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    labels = np.asarray(inputs.labels, dtype=np.int64)
    prob_scores = np.asarray(inputs.prob_scores, dtype=np.float64)
    if labels.size == 0 or prob_scores.size == 0:
        return {}

    eps = 1e-12
    prob_scores = np.clip(prob_scores, eps, 1.0 - eps)
    metrics: dict[str, float] = {}

    metrics.update(
        _compute_calibration_metrics(labels, prob_scores, n_bins=inputs.n_bins)
    )

    try:
        metrics["auc"] = float(backend.roc_auc_score(labels, prob_scores))
    except Exception:
        metrics["auc"] = 0.5

    try:
        pr_metrics = _compute_pr_metrics(
            labels,
            prob_scores,
            backend=backend,
            thresholds_from_pr=inputs.thresholds_from_pr,
            decision_threshold=inputs.decision_threshold,
        )
        metrics.update(pr_metrics)
    except Exception:
        metrics.update({"mcc": 0.0, "pr_auc": 0.0, "accuracy": 0.0, "ap": 0.0})

    return metrics


def _compute_calibration_metrics(
    labels: np.ndarray, prob_scores: np.ndarray, *, n_bins: int
) -> dict[str, float]:
    """Execute compute calibration metrics.



    Args:

        labels: Input value used by this callable.

        prob_scores: Input value used by this callable.

        n_bins: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    metrics: dict[str, float] = {}
    try:
        metrics["brier"] = float(np.mean((prob_scores - labels) ** 2))
        metrics["nll"] = float(
            -np.mean(
                labels * np.log(prob_scores)
                + (1.0 - labels) * np.log(1.0 - prob_scores)
            )
        )
        metrics["ece"] = float(
            compute_ece(
                prob_scores.astype(np.float64),
                labels.astype(np.float64),
                int(max(1, n_bins)),
            )
        )
    except Exception:
        return {}
    return metrics


def _compute_pr_metrics(
    labels: np.ndarray,
    prob_scores: np.ndarray,
    *,
    backend: BinaryMetricsBackend,
    thresholds_from_pr: bool,
    decision_threshold: float | None,
) -> dict[str, float]:
    """Execute compute pr metrics.



    Args:

        labels: Input value used by this callable.

        prob_scores: Input value used by this callable.

        backend: Input value used by this callable.

        thresholds_from_pr: Input value used by this callable.

        decision_threshold: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    precisions, recalls, thresholds = backend.precision_recall_curve(
        labels, prob_scores
    )
    metrics: dict[str, float] = {}

    if len(precisions) <= 1 or len(recalls) <= 1:
        return {"mcc": 0.0, "pr_auc": 0.0, "accuracy": 0.0, "ap": 0.0}

    sorted_indices = np.argsort(recalls)
    sorted_recalls = recalls[sorted_indices]
    sorted_precisions = precisions[sorted_indices]

    unique_mask = np.diff(sorted_recalls, prepend=-1) != 0
    unique_recalls = sorted_recalls[unique_mask]
    unique_precisions = sorted_precisions[unique_mask]

    if len(unique_recalls) >= 2:
        pr_auc = backend.auc(unique_recalls, unique_precisions)
    else:
        pr_auc = 0.0
    metrics["pr_auc"] = float(pr_auc)

    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (
        precisions[:-1] + recalls[:-1] + 1e-12
    )
    best_idx = int(np.argmax(f1_scores))
    metrics["precision"] = float(precisions[best_idx])
    metrics["recall"] = float(recalls[best_idx])
    metrics["f1"] = float(f1_scores[best_idx])

    threshold = decision_threshold if decision_threshold is not None else 0.5
    if thresholds_from_pr and len(thresholds) > best_idx:
        threshold = float(thresholds[best_idx])
    metrics["decision_threshold"] = float(threshold)

    binary_preds = (prob_scores > threshold).astype(np.int64)
    confusion = compute_confusion_counts(labels, binary_preds)
    metrics.update(confusion)

    metrics["mcc"] = float(backend.matthews_corrcoef(labels, binary_preds))
    metrics["accuracy"] = float(backend.accuracy_score(labels, binary_preds))
    metrics["ap"] = float(backend.average_precision_score(labels, prob_scores))

    return metrics


def compute_confusion_counts(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Execute compute confusion counts.



    Args:

        labels: Input value used by this callable.

        preds: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    labels_arr = np.asarray(labels, dtype=np.int64)
    preds_arr = np.asarray(preds, dtype=np.int64)
    tp = int(np.sum((labels_arr == 1) & (preds_arr == 1)))
    tn = int(np.sum((labels_arr == 0) & (preds_arr == 0)))
    fp = int(np.sum((labels_arr == 0) & (preds_arr == 1)))
    fn = int(np.sum((labels_arr == 1) & (preds_arr == 0)))
    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "vp": float(tp),
        "vn": float(tn),
    }

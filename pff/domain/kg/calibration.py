from pathlib import Path
from typing import Any

import joblib
import numpy as np

from pff.shared import logger

_sklearn_isotonic = None
_sklearn_linear = None
_sklearn_model_selection = None


def _require_sklearn_isotonic():
    """Lazy import sklearn.isotonic."""
    global _sklearn_isotonic
    if _sklearn_isotonic is None:
        try:
            from sklearn import isotonic as _mod
        except ImportError as exc:
            raise RuntimeError("sklearn não disponível para calibração.") from exc
        _sklearn_isotonic = _mod
    return _sklearn_isotonic


def _require_sklearn_linear():
    """Lazy import sklearn.linear_model."""
    global _sklearn_linear
    if _sklearn_linear is None:
        try:
            from sklearn import linear_model as _mod
        except ImportError as exc:
            raise RuntimeError("sklearn não disponível para calibração.") from exc
        _sklearn_linear = _mod
    return _sklearn_linear


def _require_sklearn_model_selection():
    """Lazy import sklearn.model_selection."""
    global _sklearn_model_selection
    if _sklearn_model_selection is None:
        try:
            from sklearn import model_selection as _mod
        except ImportError as exc:
            raise RuntimeError("sklearn não disponível para calibração.") from exc
        _sklearn_model_selection = _mod
    return _sklearn_model_selection

"""
Score calibration module for Knowledge Graph Completion.

This module provides calibration methods to transform raw scores
into well-calibrated probabilities.
"""


class CalibrationError(Exception):
    """Exception raised for errors in the calibration process."""

    pass


class ModelNotFittedError(CalibrationError):
    """Exception raised when trying to use a model that hasn't been fitted."""

    pass


class ScoreCalibrator:
    """Calibrates raw scores to produce well-calibrated probabilities."""

    def __init__(self, method: str = "platt"):
        """
        Initialize calibrator with specified method.

        Args:
            method: Calibration method - "platt" (sigmoid), "isotonic", or "both"
        """
        if method not in ["platt", "isotonic", "both"]:
            raise ValueError(
                f"Unsupported calibration method: {method}. Use 'platt', 'isotonic', or 'both'"
            )

        self.method = method
        self.platt_model = None
        self.isotonic_model = None
        self.is_fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ScoreCalibrator":
        """
        Fit calibration model on training data.

        Args:
            scores: Raw scores from the model
            labels: True binary labels (0 or 1)

        Returns:
            Self for method chaining
        """
        scores_reshaped = self._reshape_scores(scores)

        if self.method in ["platt", "both"]:
            self._fit_platt_model(scores_reshaped, labels)

        if self.method in ["isotonic", "both"]:
            self._fit_isotonic_model(scores, labels)

        self.is_fitted = True
        return self

    def _fit_platt_model(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit Platt scaling (logistic regression) model."""
        linear = _require_sklearn_linear()
        self.platt_model = linear.LogisticRegression()
        self.platt_model.fit(scores, labels)
        logger.debug("Platt scaling calibration fitted")

    def _fit_isotonic_model(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit isotonic regression model."""
        isotonic = _require_sklearn_isotonic()
        self.isotonic_model = isotonic.IsotonicRegression(out_of_bounds="clip", n_jobs=-1)
        self.isotonic_model.fit(scores.ravel(), labels)
        logger.debug("Isotonic regression calibration fitted")

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Transform raw scores to calibrated probabilities.

        Args:
            scores: Raw scores to calibrate

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ModelNotFittedError("Calibrator must be fitted before transform")

        scores_reshaped = self._reshape_scores(scores)

        if self.method == "platt":
            return self._transform_platt(scores_reshaped)
        elif self.method == "isotonic":
            return self._transform_isotonic(scores)
        else:  # self.method == "both"
            return self._transform_both(scores_reshaped)

    def _transform_platt(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling transformation."""
        if self.platt_model is None:
            raise ModelNotFittedError("Platt model is not fitted. Call 'fit' first.")
        return self.platt_model.predict_proba(scores)[:, 1]

    def _transform_isotonic(self, scores: np.ndarray) -> np.ndarray:
        """Apply isotonic regression transformation."""
        if self.isotonic_model is None:
            raise ModelNotFittedError("Isotonic model is not fitted. Call 'fit' first.")
        return self.isotonic_model.transform(scores.ravel())

    def _transform_both(self, scores: np.ndarray) -> np.ndarray:
        """Apply both Platt scaling and isotonic regression."""
        if self.platt_model is None or self.isotonic_model is None:
            raise ModelNotFittedError(
                "Both Platt and Isotonic models must be fitted. Call 'fit' first."
            )
        # Apply Platt first, then isotonic
        platt_probs = self.platt_model.predict_proba(scores)[:, 1]
        return self.isotonic_model.transform(platt_probs)

    def fit_transform(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(scores, labels).transform(scores)

    def cross_val_calibrate(
        self, scores: np.ndarray, labels: np.ndarray, cv: int = 5
    ) -> np.ndarray:
        """
        Perform cross-validated calibration to avoid overfitting.

        Args:
            scores: Raw scores
            labels: True labels
            cv: Number of cross-validation folds

        Returns:
            Cross-validated calibrated probabilities
        """
        if self.method == "platt":
            return self._cross_val_platt(scores, labels, cv)
        else:
            return self._cross_val_manual(scores, labels, cv)

    def _cross_val_platt(
        self, scores: np.ndarray, labels: np.ndarray, cv: int
    ) -> np.ndarray:
        """Perform cross-validation for Platt scaling using sklearn's built-in method."""
        linear = _require_sklearn_linear()
        model_selection = _require_sklearn_model_selection()
        scores_reshaped = self._reshape_scores(scores)
        model = linear.LogisticRegression()
        return model_selection.cross_val_predict(
            model, scores_reshaped, labels, cv=cv, method="predict_proba"
        )[:, 1]

    def _cross_val_manual(
        self, scores: np.ndarray, labels: np.ndarray, cv: int
    ) -> np.ndarray:
        """Perform manual cross-validation for isotonic regression or combined methods."""
        model_selection = _require_sklearn_model_selection()
        calibrated = np.zeros_like(scores)
        kfold = model_selection.KFold(n_splits=cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kfold.split(scores):
            temp_calibrator = ScoreCalibrator(method=self.method)
            temp_calibrator.fit(scores[train_idx], labels[train_idx])
            calibrated[val_idx] = temp_calibrator.transform(scores[val_idx])

        return calibrated

    def _reshape_scores(self, scores: np.ndarray) -> np.ndarray:
        """Ensure scores are properly shaped for model input."""
        return scores.reshape(-1, 1) if scores.ndim == 1 else scores

    def save(self, path: Path) -> None:
        """
        Save calibration model to disk.

        Args:
            path: File path where the model will be saved
        """
        model_data = {
            "method": self.method,
            "platt_model": self.platt_model,
            "isotonic_model": self.isotonic_model,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(model_data, path)
        logger.info(f"Calibrador salvo em {path}")

    @classmethod
    def load(cls, path: Path) -> "ScoreCalibrator":
        """
        Load calibration model from disk.

        Args:
            path: File path where the model was saved

        Returns:
            Loaded ScoreCalibrator instance
        """
        model_data = joblib.load(path)
        calibrator = cls(method=model_data["method"])
        calibrator.platt_model = model_data["platt_model"]
        calibrator.isotonic_model = model_data["isotonic_model"]
        calibrator.is_fitted = model_data["is_fitted"]
        return calibrator


class ScoreCalibratorBuilder:
    """Builder for ScoreCalibrator.

    Pattern: Builder
    - Encapsulates construction from configuration dictionaries.
    - Keeps method validation centralized.
    """

    @staticmethod
    def from_config(config: dict[str, Any] | None) -> ScoreCalibrator:
        method = "platt"
        if isinstance(config, dict):
            method = config.get("method", method)
        return ScoreCalibrator(method=method)


def find_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1",
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> tuple[float, dict]:
    """
    Find optimal threshold for binary classification.

    Args:
        scores: Predicted scores/probabilities
        labels: True binary labels
        metric: Metric to optimize (f1, precision, recall, or balanced_accuracy)
        min_precision: Minimum required precision (optional)
        min_recall: Minimum required recall (optional)

    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    from sklearn.metrics import precision_recall_curve  # noqa: PLC0415

    # Validate inputs
    valid_metrics = ["f1", "precision", "recall", "balanced_accuracy"]
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric: {metric}. Choose from {valid_metrics}")

    # Get precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)

    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    # Apply constraints if specified
    valid_mask = _apply_threshold_constraints(
        precisions, recalls, min_precision, min_recall
    )

    # Select optimal threshold based on metric
    best_idx, optimal_threshold = _select_optimal_threshold(
        scores, labels, thresholds, precisions, recalls, f1_scores, valid_mask, metric
    )

    # Calculate metrics at optimal threshold
    metrics = _calculate_metrics(
        optimal_threshold, precisions, recalls, f1_scores, labels, best_idx
    )

    _log_threshold_results(optimal_threshold, metrics)

    return optimal_threshold, metrics


def _apply_threshold_constraints(
    precisions: np.ndarray,
    recalls: np.ndarray,
    min_precision: float | None,
    min_recall: float | None,
) -> np.ndarray:
    """Apply minimum precision and recall constraints to thresholds."""
    valid_mask = np.ones(len(precisions) - 1, dtype=bool)

    if min_precision is not None:
        valid_mask &= precisions[:-1] >= min_precision

    if min_recall is not None:
        valid_mask &= recalls[:-1] >= min_recall

    if not valid_mask.any():
        logger.warning("No thresholds satisfy the constraints, using defaults")
        valid_mask = np.ones(len(precisions) - 1, dtype=bool)

    return valid_mask


def _select_optimal_threshold(  # noqa: PLR0913
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    precisions: np.ndarray,
    recalls: np.ndarray,
    f1_scores: np.ndarray,
    valid_mask: np.ndarray,
    metric: str,
) -> tuple[int, float]:
    """Select the optimal threshold based on the specified metric."""
    from sklearn.metrics import balanced_accuracy_score  # noqa: PLC0415

    if metric == "f1":
        best_idx = np.argmax(f1_scores[:-1] * valid_mask)
    elif metric == "precision":
        best_idx = np.argmax(precisions[:-1] * valid_mask)
    elif metric == "recall":
        best_idx = np.argmax(recalls[:-1] * valid_mask)
    elif metric == "balanced_accuracy":
        ba_scores = np.array(
            [
                balanced_accuracy_score(labels, (scores >= thresh).astype(int))
                for thresh in thresholds
            ]
        )
        best_idx = np.argmax(ba_scores * valid_mask)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return int(best_idx), thresholds[best_idx]


def _calculate_metrics(  # noqa: PLR0913
    threshold: float,
    precisions: np.ndarray,
    recalls: np.ndarray,
    f1_scores: np.ndarray,
    labels: np.ndarray,
    best_idx: int,
) -> dict:
    """Calculate and return metrics for the selected threshold."""
    return {
        "threshold": float(threshold),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1": float(f1_scores[best_idx]),
        "support": int(np.sum(labels)),
    }


def _log_threshold_results(threshold: float, metrics: dict) -> None:
    """Log the optimal threshold and resulting metrics."""
    logger.info(f"Optimal threshold: {threshold:.4f}")
    logger.info(
        f"Metrics: Precision={metrics['precision']:.4f}, "
        f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
    )

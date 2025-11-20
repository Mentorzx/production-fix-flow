"""Calibration utilities for score normalization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class ScoreCalibrator:
    """One-dimensional score calibrator based on Platt scaling.

    The helper fits a logistic regression over raw scores so that model outputs
    approximate probabilities. Only the learned coefficients are stored, making
    serialization lightweight.

    Attributes:
        method: Calibration technique name (currently only ``"platt"``).
        coef_: Learned coefficient from the logistic regression.
        intercept_: Learned intercept from the logistic regression.
        is_fitted_: Flag indicating whether ``fit`` has been executed.
    """

    method: str = "platt"
    coef_: float | None = None
    intercept_: float | None = None
    is_fitted_: bool = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit the calibrator using binary labels.

        Args:
            scores (np.ndarray): Raw model scores of shape ``[n_samples]``.
            labels (np.ndarray): Binary labels in ``{0, 1}``.

        Raises:
            ValueError: If the input is empty or lacks both classes.
        """

        if scores.size == 0:
            raise ValueError("Calibration requires at least one score")

        scores_arr = np.asarray(scores, dtype=float).reshape(-1, 1)
        labels_arr = np.asarray(labels, dtype=int).ravel()

        if np.unique(labels_arr).size < 2:
            raise ValueError("Calibration requires both positive and negative samples")

        model = LogisticRegression(max_iter=1000)
        model.fit(scores_arr, labels_arr)
        self.coef_ = float(model.coef_.ravel()[0])
        self.intercept_ = float(model.intercept_.ravel()[0])
        self.is_fitted_ = True

    def transform(self, scores: np.ndarray | list[float]) -> np.ndarray:
        """Transform raw scores into calibrated probabilities.

        Args:
            scores (np.ndarray | list[float]): Raw scores to transform.

        Returns:
            np.ndarray: Calibrated probabilities with the same shape as ``scores``.
        """

        scores_arr = np.asarray(scores, dtype=float)
        if not self.is_fitted_ or self.coef_ is None or self.intercept_ is None:
            return 1.0 / (1.0 + np.exp(-scores_arr))

        logits = self.coef_ * scores_arr + self.intercept_
        return 1.0 / (1.0 + np.exp(-logits))

    def transform_single(self, score: float) -> float:
        """Transform a single score into probability.

        Args:
            score (float): Raw score.

        Returns:
            float: Calibrated probability.
        """

        return float(self.transform(np.array([score]))[0])

    def to_dict(self) -> dict[str, float | str | bool]:
        """Serialize the calibrator state.

        Returns:
            dict[str, float | str | bool]: Payload containing coefficients and metadata.
        """

        return {
            "method": self.method,
            "coef": self.coef_,
            "intercept": self.intercept_,
            "is_fitted": self.is_fitted_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float | str | bool]) -> "ScoreCalibrator":
        """Create a calibrator from persisted state.

        Args:
            payload (dict[str, float | str | bool]): Serialized calibrator.

        Returns:
            ScoreCalibrator: Restored calibrator instance.
        """

        calibrator = cls(method=str(payload.get("method", "platt")))
        calibrator.coef_ = payload.get("coef")  # type: ignore[assignment]
        calibrator.intercept_ = payload.get("intercept")  # type: ignore[assignment]
        calibrator.is_fitted_ = bool(payload.get("is_fitted", False))
        return calibrator

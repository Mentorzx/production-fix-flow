#!/usr/bin/env python3
"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/domain/audit/calibration.py

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

_sklearn_isotonic = None
_sklearn_linear = None


def _require_sklearn_isotonic():
    """Lazy import sklearn.isotonic."""
    global _sklearn_isotonic
    if _sklearn_isotonic is None:
        try:
            from sklearn import isotonic as _mod
        except ImportError as exc:
            raise RuntimeError("sklearn not available for calibration.") from exc
        _sklearn_isotonic = _mod
    return _sklearn_isotonic


def _require_sklearn_linear():
    """Lazy import sklearn.linear_model."""
    global _sklearn_linear
    if _sklearn_linear is None:
        try:
            from sklearn import linear_model as _mod
        except ImportError as exc:
            raise RuntimeError("sklearn not available for calibration.") from exc
        _sklearn_linear = _mod
    return _sklearn_linear


MIN_UNIQUE_LABELS = 2


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration configuration loaded from `config/audit/audit.yaml`."""

    method: str = "isotonic"
    ece_bins: int = 15
    min_samples_per_relation: int = 200
    clip_eps: float = 1e-12

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CalibrationConfig:
        """Execute from dict.



        Args:

            data: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        return CalibrationConfig(
            method=data.get("method", "isotonic"),
            ece_bins=data.get("ece_bins", 15),
            min_samples_per_relation=data.get("min_samples_per_relation", 200),
            clip_eps=data.get("clip_eps", 1e-12),
        )

    @classmethod
    def load(cls) -> CalibrationConfig:
        """Mock load from config."""
        return cls()


class Calibrator(ABC):
    """Base class for all calibrators used in the audit layer."""

    @abstractmethod
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Transform raw scores to calibrated probabilities."""
        ...


class PlattCalibrator(Calibrator):
    """Platt scaling (Logistic Regression) calibrator."""

    def __init__(self, coef: float, intercept: float):
        """Execute init.



        Args:

            coef: Input value used by this callable.

            intercept: Input value used by this callable.

        """

        self.coef = coef
        self.intercept = intercept

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Execute transform.



        Args:

            scores: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        z = self.coef * scores + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


class IsotonicCalibrator(Calibrator):
    """Isotonic regression calibrator using piecewise linear interpolation."""

    def __init__(self, x: list[float], y: list[float]):
        """Execute init.



        Args:

            x: Input value used by this callable.

            y: Input value used by this callable.

        """

        self.x = np.array(x)
        self.y = np.array(y)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Execute transform.



        Args:

            scores: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if len(self.x) < 2:
            raise ValueError("Isotonic calibrator requires at least 2 points")
        return np.interp(scores, self.x, self.y)  # type: ignore[no-any-return]


def calibrator_from_dict(data: dict[str, Any] | None) -> Calibrator:
    """Reconstruct a Calibrator object from a dictionary payload."""
    if data is None:
        raise ValueError("Calibration payload is required")

    method = data.get("method", "platt")
    if method == "isotonic":
        return IsotonicCalibrator(x=data.get("x", []), y=data.get("y", []))
    else:
        return PlattCalibrator(
            coef=float(data.get("coef", 1.0)),
            intercept=float(data.get("intercept", 0.0)),
        )


def fit_per_relation_calibrators(
    scores: np.ndarray,
    labels: np.ndarray,
    relations: np.ndarray,
    config: CalibrationConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit calibrators for each relation and a global one."""
    if config is None:
        config = CalibrationConfig()

    results: dict[str, dict[str, Any]] = {}

    global_model = _fit_single_calibrator(scores, labels, config)
    results["__global__"] = {
        "model": global_model,
        "n_samples": len(scores),
    }

    unique_rels = np.unique(relations)
    for rel in unique_rels:
        mask = relations == rel
        rel_scores = scores[mask]
        rel_labels = labels[mask]

        if len(rel_scores) >= config.min_samples_per_relation:
            rel_model = _fit_single_calibrator(rel_scores, rel_labels, config)
            if rel_model:
                results[str(rel)] = {
                    "model": rel_model,
                    "n_samples": len(rel_scores),
                }

    return results


def _fit_single_calibrator(
    scores: np.ndarray, labels: np.ndarray, config: CalibrationConfig
) -> dict[str, Any] | None:
    """Helper to fit a single sklearn model and return its parameters."""
    if len(np.unique(labels)) < MIN_UNIQUE_LABELS:
        return None
    if config.method == "isotonic":
        mod = _require_sklearn_isotonic().IsotonicRegression(out_of_bounds="clip")
        mod.fit(scores, labels)
        return {
            "method": "isotonic",
            "x": (mod.X_thresholds_.tolist() if hasattr(mod, "X_thresholds_") else []),
            "y": (mod.y_thresholds_.tolist() if hasattr(mod, "y_thresholds_") else []),
            "is_fitted": True,
        }
    # C=1e12 ≈ no regularization for Platt scaling.
    # We avoid C=np.inf because sklearn 1.8 internally converts it to penalty=None
    # which triggers its own deprecation warning.
    mod = _require_sklearn_linear().LogisticRegression(C=1e12)
    mod.fit(scores.reshape(-1, 1), labels)
    coef = float(np.ravel(mod.coef_).item())
    intercept = float(np.ravel(mod.intercept_).item())
    return {
        "method": "platt",
        "coef": coef,
        "intercept": intercept,
        "is_fitted": True,
    }

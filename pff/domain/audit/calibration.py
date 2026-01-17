"""Calibration utilities for converting ranking scores into probabilities.

Design patterns:
    - Strategy: supports multiple calibration methods (Platt, Isotonic).
    - Builder: constructs per-relation calibration models + metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pff.shared import FileManager
from pff.shared.core.config import AUDIT_CONFIG_PATH
from pff.shared.acceleration.numba_kernels import compute_ece_numba, NUMBA_AVAILABLE

_sklearn_isotonic = None
_sklearn_linear = None


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


MIN_UNIQUE_LABELS = 2


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration configuration loaded from `config/audit/audit.yaml`."""

    method: str = "isotonic"
    ece_bins: int = 15
    min_samples_per_relation: int = 200
    clip_eps: float = 1e-12

    @staticmethod
    def load(file_manager: FileManager | None = None) -> CalibrationConfig:
        fm = file_manager or FileManager()
        try:
            cfg_obj = fm.read(AUDIT_CONFIG_PATH, return_native=True)
        except FileNotFoundError:
            return CalibrationConfig()
        if not isinstance(cfg_obj, dict):
            return CalibrationConfig()
        audit_cfg = cfg_obj.get("audit", cfg_obj)
        if not isinstance(audit_cfg, dict):
            return CalibrationConfig()
        cal_cfg = audit_cfg.get("calibration", {})
        if not isinstance(cal_cfg, dict):
            return CalibrationConfig()
        return CalibrationConfig(
            method=str(cal_cfg.get("method", "isotonic")),
            ece_bins=int(cal_cfg.get("ece_bins", 15)),
            min_samples_per_relation=int(cal_cfg.get("min_samples_per_relation", 200)),
            clip_eps=float(cal_cfg.get("clip_eps", 1e-12)),
        )


@dataclass(frozen=True)
class PlattModel:
    """Platt scaling model (logistic regression over raw scores)."""

    coef: float
    intercept: float

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores_arr = np.asarray(scores, dtype=np.float64)
        logits = self.coef * scores_arr + self.intercept
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "platt",
            "coef": float(self.coef),
            "intercept": float(self.intercept),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> PlattModel:
        return PlattModel(coef=float(payload["coef"]), intercept=float(payload["intercept"]))


@dataclass(frozen=True)
class IsotonicModel:
    """Isotonic regression model stored as piecewise linear thresholds."""

    x_thresholds: list[float]
    y_thresholds: list[float]

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores_arr = np.asarray(scores, dtype=np.float64)
        x = np.asarray(self.x_thresholds, dtype=np.float64)
        y = np.asarray(self.y_thresholds, dtype=np.float64)
        if x.size == 0 or y.size == 0:
            return 1.0 / (1.0 + np.exp(-scores_arr))
        return np.interp(scores_arr, x, y, left=float(y[0]), right=float(y[-1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "isotonic",
            "x_thresholds": list(self.x_thresholds),
            "y_thresholds": list(self.y_thresholds),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> IsotonicModel:
        return IsotonicModel(
            x_thresholds=[float(x) for x in payload["x_thresholds"]],
            y_thresholds=[float(y) for y in payload["y_thresholds"]],
        )


def _clip_probs(probs: np.ndarray, *, eps: float) -> np.ndarray:
    eps = float(eps)
    return np.clip(probs, eps, 1.0 - eps)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray, *, eps: float) -> float:
    p = _clip_probs(np.asarray(probs, dtype=np.float64), eps=eps)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int,
) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n_bins = max(1, int(n_bins))

    if NUMBA_AVAILABLE:
        return compute_ece_numba(p, y, n_bins)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, edges[1:-1], right=True)

    ece = 0.0
    n = float(p.size) if p.size else 1.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        acc = float(np.mean(y[mask]))
        conf = float(np.mean(p[mask]))
        ece += float(np.sum(mask)) / n * abs(acc - conf)
    return float(ece)


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> PlattModel:
    linear = _require_sklearn_linear()
    x = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(labels, dtype=np.int64).ravel()
    model = linear.LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(x, y)
    coef = float(model.coef_.ravel()[0])
    intercept = float(model.intercept_.ravel()[0])
    return PlattModel(coef=coef, intercept=intercept)


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicModel:
    isotonic = _require_sklearn_isotonic()
    x = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    model = isotonic.IsotonicRegression(out_of_bounds="clip")
    model.fit(x, y)
    return IsotonicModel(
        x_thresholds=[float(v) for v in model.X_thresholds_.tolist()],
        y_thresholds=[float(v) for v in model.y_thresholds_.tolist()],
    )


def fit_calibrator(
    scores: np.ndarray, labels: np.ndarray, *, method: str
) -> PlattModel | IsotonicModel:
    method_norm = str(method).lower().strip()
    if method_norm == "platt":
        return fit_platt(scores, labels)
    if method_norm == "isotonic":
        return fit_isotonic(scores, labels)
    raise ValueError(f"Unsupported calibration method: {method!r}")


def calibrator_from_dict(payload: dict[str, Any]) -> PlattModel | IsotonicModel:
    method = str(payload.get("method", "")).lower()
    if method == "platt":
        return PlattModel.from_dict(payload)
    if method == "isotonic":
        return IsotonicModel.from_dict(payload)
    raise ValueError(f"Unsupported calibration payload method: {method!r}")


def calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int,
    eps: float,
) -> dict[str, float]:
    p = _clip_probs(np.asarray(probs, dtype=np.float64), eps=eps)
    y = np.asarray(labels, dtype=np.int64).ravel()
    return {
        "brier": brier_score(p, y),
        "nll": negative_log_likelihood(p, y, eps=eps),
        "ece": expected_calibration_error(p, y, n_bins=n_bins),
    }


def fit_per_relation_calibrators(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    relations: np.ndarray,
    config: CalibrationConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit global + per-relation calibrators.

    Returns a dictionary keyed by relation name plus "__global__". Each value
    contains:
        - model: serialized calibrator
        - metrics: brier/nll/ece on the same data (for auditing; not CV)
        - n: sample count used
    """

    cfg = config or CalibrationConfig.load()
    scores_arr = np.asarray(scores, dtype=np.float64).ravel()
    labels_arr = np.asarray(labels, dtype=np.int64).ravel()
    rel_arr = np.asarray(relations).astype(str)

    result: dict[str, dict[str, Any]] = {}

    global_model = fit_calibrator(scores_arr, labels_arr, method=cfg.method)
    global_probs = global_model.transform(scores_arr)
    result["__global__"] = {
        "model": global_model.to_dict(),
        "metrics": calibration_metrics(
            global_probs, labels_arr, n_bins=cfg.ece_bins, eps=cfg.clip_eps
        ),
        "n": int(scores_arr.size),
    }

    for relation in sorted(set(rel_arr.tolist())):
        mask = rel_arr == relation
        if int(np.sum(mask)) < int(cfg.min_samples_per_relation):
            continue
        rel_scores = scores_arr[mask]
        rel_labels = labels_arr[mask]
        if np.unique(rel_labels).size < MIN_UNIQUE_LABELS:
            continue
        model = fit_calibrator(rel_scores, rel_labels, method=cfg.method)
        probs = model.transform(rel_scores)
        result[relation] = {
            "model": model.to_dict(),
            "metrics": calibration_metrics(
                probs, rel_labels, n_bins=cfg.ece_bins, eps=cfg.clip_eps
            ),
            "n": int(rel_scores.size),
        }

    return result

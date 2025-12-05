"""Calibration and uncertainty metrics utilities.

Provides small, dependency-free helpers to compute Expected Calibration Error
and prediction entropy for probabilistic classifiers. These helpers live in
the utils layer to keep services/validators free of metric implementations.
"""

from __future__ import annotations

import numpy as np


def compute_ece(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) for binary probabilities.

    Args:
        probabilities: Predicted probabilities in [0, 1], shape (n_samples,).
        labels: True binary labels {0, 1}, shape (n_samples,).
        n_bins: Number of calibration bins (default: 15).

    Returns:
        Expected calibration error in [0, 1] (lower is better).

    Raises:
        ValueError: If inputs have different lengths or n_bins < 1.
    """
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must have the same length")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y_true = np.asarray(labels, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(probs)

    for i in range(n_bins):
        start, end = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= start) & (probs < end) if i < n_bins - 1 else (probs >= start) & (probs <= end)
        if not np.any(mask):
            continue
        bin_confidence = probs[mask].mean()
        bin_accuracy = y_true[mask].mean()
        bin_prob = mask.mean()
        ece += bin_prob * abs(bin_accuracy - bin_confidence)

    return float(ece)


def prediction_entropy(probabilities: np.ndarray, *, average: bool = True) -> float | np.ndarray:
    """Compute entropy of binary probabilities.

    Args:
        probabilities: Predicted probabilities in [0, 1], shape (n_samples,).
        average: If True, return the mean entropy; otherwise return per-sample values.

    Returns:
        Mean entropy (float) if average=True, else array of entropies.
    """
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    entropies = -probs * np.log(probs) - (1.0 - probs) * np.log(1.0 - probs)
    if average:
        return float(entropies.mean())
    return entropies

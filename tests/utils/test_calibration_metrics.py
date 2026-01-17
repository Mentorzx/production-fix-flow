from __future__ import annotations

import numpy as np
import pytest

from tests.support.calibration_metrics import compute_ece, prediction_entropy


def test_ece_perfect_calibration_zero_error() -> None:
    probs = np.array([1e-6, 1.0 - 1e-6])
    labels = np.array([0, 1])
    ece = compute_ece(probs, labels, n_bins=2)
    assert ece == pytest.approx(0.0, abs=1e-5)


def test_ece_detects_miscalibration() -> None:
    probs = np.array([0.9, 0.9, 0.9, 0.9])
    labels = np.array([0, 0, 0, 1])
    ece = compute_ece(probs, labels, n_bins=2)
    assert ece > 0.0


def test_prediction_entropy_mean_and_per_sample() -> None:
    probs = np.array([0.5, 0.9, 0.1])
    mean_entropy = prediction_entropy(probs, average=True)
    per_sample = prediction_entropy(probs, average=False)
    assert per_sample.shape == probs.shape
    assert mean_entropy == pytest.approx(per_sample.mean())


def test_compute_ece_validates_inputs() -> None:
    with pytest.raises(ValueError):
        compute_ece(np.array([0.1, 0.2]), np.array([1]), n_bins=2)
    with pytest.raises(ValueError):
        compute_ece(np.array([0.1, 0.2]), np.array([1, 0]), n_bins=0)

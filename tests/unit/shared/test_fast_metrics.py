from __future__ import annotations

import numpy as np
from sklearn.metrics import matthews_corrcoef
from pff_rust import fast_mcc_sweep


def test_fast_mcc_sweep_equivalence():
    """Ensure fast_mcc_sweep matches sklearn results exactly."""
    np.random.seed(42)
    n_samples = 1000
    scores = np.random.randn(n_samples).astype(np.float64)
    labels = np.random.randint(0, 2, n_samples).astype(np.int64)
    thresholds = np.linspace(-2, 2, 10).astype(np.float64)

    # 1. Accelerated result
    mcc_fast, vp, vn, fp, fn, best_t = fast_mcc_sweep(labels, scores, thresholds)

    # 2. Manual Sklearn sweep
    best_mcc_sk = -2.0
    for t in thresholds:
        preds = (scores > t).astype(int)
        mcc = matthews_corrcoef(labels, preds)
        if mcc > best_mcc_sk:
            best_mcc_sk = mcc

    assert abs(mcc_fast - best_mcc_sk) < 1e-10


def test_fast_mcc_handles_empty():
    """Ensure fast_mcc_sweep handles edge cases."""
    labels = np.zeros(0, dtype=np.int64)
    scores = np.zeros(0, dtype=np.float64)
    thresholds = np.array([0.5], dtype=np.float64)

    mcc, vp, vn, fp, fn, _ = fast_mcc_sweep(labels, scores, thresholds)
    assert mcc == 0.0 or mcc == -2.0  # -2.0 is our init value for 'not improved'

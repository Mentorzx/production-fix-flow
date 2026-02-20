"""Classification metrics should match sklearn within tolerance."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


def test_classification_metrics_match_sklearn() -> None:
    sklearn = pytest.importorskip("sklearn")
    pytest.importorskip("pff_rust")

    from pff.domain.kg.pipeline import MetricsCalculator

    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int64)
    y_scores = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.4, 0.95], dtype=np.float64)

    df = pl.DataFrame({"is_true": y_true, "score": y_scores})
    calc = MetricsCalculator(config=None, top_k=10)
    metrics = calc._calculate_classification_metrics(df, calibrated=False)

    roc_auc = sklearn.metrics.roc_auc_score(y_true, y_scores)
    pr_auc = sklearn.metrics.average_precision_score(y_true, y_scores)

    assert metrics["roc_auc"] == pytest.approx(roc_auc, rel=1e-9, abs=1e-9)
    assert metrics["pr_auc"] == pytest.approx(pr_auc, rel=1e-9, abs=1e-9)

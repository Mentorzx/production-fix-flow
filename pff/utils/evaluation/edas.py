"""
EDAS-based scoring utilities for multi-criteria evaluation.

Design Pattern: Strategy/Utility
- Provides a compute_score helper that can be swapped in pipeline scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pff.utils.core.logger import logger


@dataclass
class EDASResult:
    """Holds EDAS intermediate metrics for introspection."""

    score: float
    positive_distance: float
    negative_distance: float
    reference: float


class KGEDASEvaluator:
    """
    EDAS evaluator for KG ensemble metrics.

    Uses a reference value (default 0.5) and computes positive and negative
    distances to derive a final score in [0, 1].
    """

    def __init__(self, reference_value: float = 0.5) -> None:
        self.reference_value = reference_value

    def compute_score(
        self, metrics: dict[str, Any], weights: dict[str, float] | None = None
    ) -> EDASResult:
        """
        Compute EDAS score with optional weights.

        Missing metrics default to 0.0 to keep the evaluator robust.

        Args:
            metrics: Mapping of metric name -> normalized value [0, 1].
            weights: Optional weights per metric (default uniform).

        Returns:
            EDASResult with score and distances.
        """
        if not metrics:
            return EDASResult(score=0.0, positive_distance=0.0, negative_distance=0.0, reference=self.reference_value)

        metric_names = list(metrics.keys())
        values = np.array([float(metrics.get(k, 0.0) or 0.0) for k in metric_names], dtype=np.float64)
        weights_arr = None
        if weights:
            weights_arr = np.array([float(weights.get(k, weights.get("*", 1.0))) for k in metric_names], dtype=np.float64)
        else:
            weights_arr = np.ones_like(values, dtype=np.float64)

        ref = float(self.reference_value)
        pos_dist = np.maximum(0.0, values - ref) * weights_arr
        neg_dist = np.maximum(0.0, ref - values) * weights_arr

        max_pos = np.sum(np.maximum(0.0, 1.0 - ref) * weights_arr)
        max_neg = np.sum(np.maximum(0.0, ref) * weights_arr)

        pos_score = 0.0 if max_pos == 0 else np.sum(pos_dist) / max_pos
        neg_score = 0.0 if max_neg == 0 else np.sum(neg_dist) / max_neg

        final_score = float((pos_score + (1.0 - neg_score)) / 2.0)
        final_score = max(0.0, min(1.0, final_score))

        logger.debug(
            f"EDAS: ref={ref:.3f} pos={pos_score:.3f} neg={neg_score:.3f} score={final_score:.3f}"
        )
        return EDASResult(
            score=final_score,
            positive_distance=float(pos_score),
            negative_distance=float(neg_score),
            reference=ref,
        )

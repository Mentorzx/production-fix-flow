"""
Probabilistic Circuit inference (minimal).

Current implementation executes a Noisy-OR style aggregation on top
of the compiled circuit metadata, preserving determinism and allowing
future replacement with a full sum-product forward pass.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pff.domain.learning.pc.compiler import CompiledCircuit


class PCInferenceEngine:
    """Execute inference on a compiled probabilistic circuit."""

    def __init__(self, normalize_weights: bool = True) -> None:
        """Execute init.



        Args:

            normalize_weights: Optional input value.

        """

        self.normalize_weights = normalize_weights

    def infer(
        self,
        circuit: CompiledCircuit,
        confidences: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Run a lightweight forward pass."""
        if circuit.rule_count == 0 or len(confidences) == 0:
            return 0.0

        scores = np.clip(confidences, 0.0, 1.0 - 1e-9)
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if self.normalize_weights and np.sum(w) > 0:
                w = w / np.sum(w)
            w = w[: len(scores)]
            scores = np.clip(scores * w, 0.0, 1.0 - 1e-9)

        complement = 1.0 - scores
        return float(1.0 - np.prod(complement))

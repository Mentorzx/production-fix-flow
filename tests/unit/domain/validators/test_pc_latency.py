"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/validators/test_pc_latency.py

"""

from __future__ import annotations

import time

import numpy as np

from pff.domain.learning.ml.aggregation_strategies import NoisyOrStrategy
from pff.domain.learning.pc.strategy import ProbabilisticCircuitStrategy


def test_pc_latency_within_factor():
    """Execute test pc latency within factor.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    rng = np.random.default_rng(42)
    confidences = rng.random(256).astype(np.float64)
    weights = rng.random(256).astype(np.float64)

    pc = ProbabilisticCircuitStrategy(max_rules_per_circuit=512, compilation_timeout_ms=1000)
    noisy = NoisyOrStrategy()

    loops = 200
    start = time.perf_counter()
    for _ in range(loops):
        noisy.aggregate(confidences, weights)
    noisy_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(loops):
        pc.aggregate(confidences, weights)
    pc_time = time.perf_counter() - start

    assert pc_time <= noisy_time * 5

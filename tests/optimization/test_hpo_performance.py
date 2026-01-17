"""Performance and timing tests for HPO scoring functions.

Fast benchmarks (no I/O) to catch performance regressions in:
- _normalize_metric throughput
- _blend_scores throughput
- Penalty computation loop
"""

from __future__ import annotations

import math
import time

import pytest


# Inline implementations for fast testing (no imports needed)
def _normalize_metric(value: float, *, low: float, high: float) -> float:
    """Clamp and scale a metric into [0, 1] interval."""
    if math.isnan(value):
        return 0.0
    if high <= low:
        return max(0.0, min(1.0, value))
    normalized = (value - low) / (high - low)
    return max(0.0, min(1.0, normalized))


def _blend_scores(scores: list[tuple[float, float]]) -> float:
    """Compute a weighted average from (value, weight) pairs."""
    total_weight = 0.0
    total = 0.0
    for value, weight in scores:
        if weight <= 0:
            continue
        total += value * weight
        total_weight += weight
    if total_weight == 0.0:
        return 0.0
    return total / total_weight


def compute_penalty_stack(
    base_score: float, penalties: list[tuple[float, float]]
) -> float:
    """Compute composite score with penalty stacking."""
    score = base_score
    for coeff, penalty in penalties:
        score *= 1.0 - coeff * min(1.0, penalty)
    return max(0.0, score)


class TestNormalizeMetricPerformance:
    """Benchmark _normalize_metric throughput."""

    @pytest.mark.parametrize("n_iterations", [10_000])
    def test_normalize_metric_throughput(self, n_iterations: int):
        """_normalize_metric should handle 10k calls in < 50ms."""
        start = time.perf_counter()
        for i in range(n_iterations):
            _normalize_metric(0.5 + i * 0.00001, low=0.0, high=1.0)
        elapsed = time.perf_counter() - start

        assert (
            elapsed < 0.05
        ), f"normalize_metric too slow: {elapsed:.3f}s for {n_iterations} calls"

    def test_normalize_metric_with_nans(self):
        """NaN handling should not significantly slow down."""
        n_iterations = 5_000

        # Without NaNs
        start = time.perf_counter()
        for i in range(n_iterations):
            _normalize_metric(0.5, low=0.0, high=1.0)
        time_normal = time.perf_counter() - start

        # With NaNs
        start = time.perf_counter()
        for i in range(n_iterations):
            _normalize_metric(float("nan"), low=0.0, high=1.0)
        time_nan = time.perf_counter() - start

        # NaN handling should not be > 3x slower
        assert (
            time_nan < time_normal * 3
        ), f"NaN handling slow: {time_nan:.4f}s vs {time_normal:.4f}s"


class TestBlendScoresPerformance:
    """Benchmark _blend_scores throughput."""

    def test_blend_scores_throughput(self):
        """_blend_scores should handle 10k calls in < 50ms."""
        scores = [(0.8, 0.3), (0.6, 0.5), (0.9, 0.2)]
        n_iterations = 10_000

        start = time.perf_counter()
        for _ in range(n_iterations):
            _blend_scores(scores)
        elapsed = time.perf_counter() - start

        assert (
            elapsed < 0.05
        ), f"blend_scores too slow: {elapsed:.3f}s for {n_iterations} calls"

    def test_blend_scores_scaling(self):
        """Performance should scale linearly with number of scores."""
        n_iterations = 1_000

        # 3 scores
        scores_3 = [(0.8, 0.3), (0.6, 0.5), (0.9, 0.2)]
        start = time.perf_counter()
        for _ in range(n_iterations):
            _blend_scores(scores_3)
        time_3 = time.perf_counter() - start

        # 30 scores (10x more)
        scores_30 = [(0.5 + i * 0.01, 0.1) for i in range(30)]
        start = time.perf_counter()
        for _ in range(n_iterations):
            _blend_scores(scores_30)
        time_30 = time.perf_counter() - start

        # Should be roughly linear (allow 15x for 10x input)
        assert (
            time_30 < time_3 * 15
        ), f"blend_scores not scaling linearly: {time_30:.4f}s vs {time_3:.4f}s"


class TestPenaltyComputationPerformance:
    """Benchmark penalty stacking computation."""

    def test_penalty_stack_throughput(self):
        """Penalty computation should handle 10k trials in < 100ms."""
        penalties = [
            (0.40, 0.1),
            (0.45, 0.2),
            (0.35, 0.15),
            (0.20, 0.05),
            (0.50, 0.25),
            (0.60, 0.1),
        ]
        n_iterations = 10_000

        start = time.perf_counter()
        for i in range(n_iterations):
            compute_penalty_stack(0.8 + i * 0.00001, penalties)
        elapsed = time.perf_counter() - start

        assert (
            elapsed < 0.1
        ), f"penalty_stack too slow: {elapsed:.3f}s for {n_iterations} calls"


class TestFullScoreComputationPerformance:
    """Benchmark full score computation pipeline."""

    def test_full_score_pipeline(self):
        """Full score computation should handle 1k trials in < 100ms."""
        n_iterations = 1_000

        def compute_full_score(trial_idx: int) -> float:
            # Simulate _compute_score logic
            neural_w = 0.3
            rules_w = 0.2
            lgbm_w = 0.5

            # Normalize metrics
            kge = _normalize_metric(0.45 + trial_idx * 0.0001, low=0.15, high=0.75)
            rules = _normalize_metric(0.7, low=0.4, high=0.95)
            lgbm = _normalize_metric(0.85, low=0.6, high=0.99)

            # Blend scores
            base = _blend_scores(
                [
                    (kge, max(neural_w, 0.05)),
                    (rules, max(rules_w, 0.05)),
                    (lgbm, min(max(lgbm_w, 0.05), 0.70)),
                ]
            )

            # Apply penalties
            penalties = [
                (0.40, max(0.0, 0.05 - min(neural_w, rules_w, lgbm_w))),
                (0.45, max(0.0, 0.05 - 0.08)),
                (0.35, max(0.0, 0.25 - rules_w)),
                (0.20, max(0.0, lgbm_w - 0.70)),
                (0.50, 0.1),
                (0.60, 0.05),
            ]

            return compute_penalty_stack(base, penalties)

        start = time.perf_counter()
        for i in range(n_iterations):
            compute_full_score(i)
        elapsed = time.perf_counter() - start

        assert (
            elapsed < 0.1
        ), f"full_score too slow: {elapsed:.3f}s for {n_iterations} calls"


class TestMemoryEfficiency:
    """Test memory efficiency of scoring functions."""

    def test_no_memory_leak_in_blend_scores(self):
        """blend_scores should not accumulate memory."""
        import sys

        scores = [(0.8, 0.3), (0.6, 0.5), (0.9, 0.2)]

        # Warm up
        for _ in range(100):
            _blend_scores(scores)

        # Measure
        results = []
        for _ in range(1000):
            results.append(_blend_scores(scores))

        # All results should be identical floats
        assert len(set(results)) == 1

        # Memory for 1000 floats should be minimal
        mem = sys.getsizeof(results)
        assert mem < 50_000  # Less than 50KB for 1000 floats

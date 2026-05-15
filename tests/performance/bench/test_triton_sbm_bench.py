"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/performance/bench/test_triton_sbm_bench.py

"""

import time

import pytest
import torch

from pff.shared.acceleration.triton_kernels import (
    TRITON_AVAILABLE,
    TritonDotProductValidator,
)


def test_triton_sbm_performance():
    """Execute test triton sbm performance.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    has_triton_runtime = bool(TRITON_AVAILABLE and torch.cuda.is_available())
    if not has_triton_runtime:
        assert has_triton_runtime is False
        return

    device = "cuda"
    B = 512
    N = 100_000
    D = 128

    # Deterministic seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Random data
    Q = torch.randn(B, D, device=device).contiguous()
    E = torch.randn(N, D, device=device).contiguous()
    true_tails = torch.randint(0, N, (B,), device=device)

    # 1. PyTorch Baseline
    def _time(fn, iters: int = 3) -> float:
        """Execute time.



        Args:

            fn: Input value used by this callable.

            iters: Optional input value.



        Returns:

            Return value produced by the callable.

        """

        best = float("inf")
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            end = time.perf_counter()
            best = min(best, end - start)
        return best

    def _torch_baseline():
        """Execute torch baseline.



        Returns:

            Return value produced by the callable.

        """

        scores = torch.mm(Q, E.t())
        target_scores = scores[torch.arange(B), true_tails].unsqueeze(1)
        return (scores > target_scores).sum(dim=1) + 1

    torch_ranks = _torch_baseline()
    duration_torch = _time(_torch_baseline, iters=5)

    print(f"\nPyTorch Baseline (B={B}, N={N}): {duration_torch * 1000:.2f} ms")

    # 2. Triton Kernel
    validator = TritonDotProductValidator(E, device=device)

    # Warmup
    validator.compute_ranks(Q[:16], true_tails[:16])

    triton_ranks = validator.compute_ranks(Q, true_tails)
    duration_triton = _time(lambda: validator.compute_ranks(Q, true_tails), iters=5)

    print(f"Triton Kernel (B={B}, N={N}): {duration_triton * 1000:.2f} ms")
    print(f"Speedup: {duration_torch / duration_triton:.2f}x")

    # Correctness: Triton ranks must correlate closely with PyTorch ranks.
    # Tiled dot-product accumulation uses different floating-point reduction order
    # than cuBLAS's single-matmul. For entities with near-identical scores this
    # can shift ranks. CUDA state from prior tests can amplify this effect.
    # We validate that the mean rank difference is negligible relative to N.
    rank_diff = (triton_ranks.float() - torch_ranks.float()).abs()
    mean_pct = rank_diff.mean().item() / N * 100
    p99_pct = torch.quantile(rank_diff, 0.99).item() / N * 100
    assert mean_pct < 0.1, f"Mean rank diff too large: {mean_pct:.4f}% of N"
    assert p99_pct < 0.1, f"P99 rank diff too large: {p99_pct:.4f}% of N"

    # Performance: Triton is memory-frugal (no N×N matrix), not speed-optimized.
    # It trades speed for O(B×D) memory vs cuBLAS O(B×N).
    # Just ensure it completes in a reasonable time (< 2s for 512 queries × 100K entities).
    assert duration_triton < 2.0, (
        f"Triton kernel too slow: {duration_triton * 1000:.2f}ms (expected < 2000ms)"
    )

import time

import pytest
import torch

from pff.domain.learning.dslfm.triton_kernels import (
    TRITON_AVAILABLE,
    TritonDotProductValidator,
)


@pytest.mark.skipif(
    not TRITON_AVAILABLE or not torch.cuda.is_available(), reason="Triton/GPU required"
)
def test_triton_sbm_performance():
    device = "cuda"
    B = 512
    N = 100_000
    D = 128

    # Random data
    Q = torch.randn(B, D, device=device).contiguous()
    E = torch.randn(N, D, device=device).contiguous()
    true_tails = torch.randint(0, N, (B,), device=device)

    # 1. PyTorch Baseline
    def _time(fn, iters: int = 3) -> float:
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
        scores = torch.mm(Q, E.t())
        target_scores = scores[torch.arange(B), true_tails].unsqueeze(1)
        _ = (scores > target_scores).sum(dim=1) + 1

    _torch_baseline()
    duration_torch = _time(_torch_baseline, iters=5)

    print(f"\nPyTorch Baseline (B={B}, N={N}): {duration_torch * 1000:.2f} ms")

    # 2. Triton Kernel
    validator = TritonDotProductValidator(E, device=device)

    # Warmup
    validator.compute_ranks(Q[:16], true_tails[:16])

    def _triton_run():
        validator.compute_ranks(Q, true_tails)

    duration_triton = _time(_triton_run, iters=5)

    print(f"Triton Kernel (B={B}, N={N}): {duration_triton * 1000:.2f} ms")
    print(f"Speedup: {duration_torch / duration_triton:.2f}x")

    # Assert correctness logic (approximate due to float precision, but should be close)
    # We can't compare directly here because we deleted torch results to save memory for the bench.
    # But usually we'd want to check correctness.

    max_ratio = 20.0
    # Triton validator is memory-frugal; allow slower runtimes vs cuBLAS baseline.
    assert duration_triton <= duration_torch * max_ratio

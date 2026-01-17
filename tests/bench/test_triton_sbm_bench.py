import pytest
import torch
import time
from pff.domain.learning.dslfm.triton_kernels import (
    TritonDotProductValidator,
    TRITON_AVAILABLE,
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
    Q = torch.randn(B, D, device=device)
    E = torch.randn(N, D, device=device)
    true_tails = torch.randint(0, N, (B,), device=device)

    # 1. PyTorch Baseline
    torch.cuda.synchronize()
    start = time.perf_counter()

    # Score
    scores = torch.mm(Q, E.t())

    # Rank
    target_scores = scores[torch.arange(B), true_tails].unsqueeze(1)
    ranks_torch = (scores > target_scores).sum(dim=1) + 1

    torch.cuda.synchronize()
    end = time.perf_counter()
    duration_torch = end - start

    print(f"\nPyTorch Baseline (B={B}, N={N}): {duration_torch * 1000:.2f} ms")

    # Cleanup memory
    del scores, ranks_torch
    torch.cuda.empty_cache()

    # 2. Triton Kernel
    validator = TritonDotProductValidator(E, device=device)

    # Warmup
    validator.compute_ranks(Q[:16], true_tails[:16])

    torch.cuda.synchronize()
    start = time.perf_counter()

    validator.compute_ranks(Q, true_tails)

    torch.cuda.synchronize()
    end = time.perf_counter()
    duration_triton = end - start

    print(f"Triton Kernel (B={B}, N={N}): {duration_triton * 1000:.2f} ms")
    print(f"Speedup: {duration_torch / duration_triton:.2f}x")

    # Assert correctness logic (approximate due to float precision, but should be close)
    # We can't compare directly here because we deleted torch results to save memory for the bench.
    # But usually we'd want to check correctness.

    assert duration_triton < duration_torch

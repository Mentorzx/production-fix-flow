"""Test numerical stability of IBP KL divergence under mixed precision.

This test reproduces NaN/Inf issues when computing KL divergence with
float16/bfloat16 inputs, especially under autocast (AMP).
"""

from __future__ import annotations

import pytest
import torch

from pff.domain.learning.dslfm.vae import IndianBuffetProcessPrior


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ibp_kl_finite_with_extreme_probs(dtype: torch.dtype) -> None:
    """Test KL divergence remains finite with extreme probability values.

    Args:
        dtype: Data type for q_z tensor.
    """
    prior = IndianBuffetProcessPrior(alpha=1.0, max_communities=8)

    # Test cases: values very close to 0, 1, and in between
    batch_size = 4
    device = torch.device("cpu")

    # Case 1: q_z near 0
    q_z_near_zero = torch.tensor(
        [[1e-6, 0.1, 0.5, 0.9, 1.0 - 1e-6, 0.2, 0.3, 0.4]],
        dtype=dtype,
        device=device,
    ).expand(batch_size, -1)

    # Case 2: q_z near 1
    q_z_near_one = torch.tensor(
        [[1.0 - 1e-6, 0.9, 0.5, 0.1, 1e-6, 0.8, 0.7, 0.6]],
        dtype=dtype,
        device=device,
    ).expand(batch_size, -1)

    # Case 3: q_z exactly 0 or 1 (edge case)
    q_z_exact = torch.tensor(
        [[0.0, 1.0, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).expand(batch_size, -1)

    for q_z in [q_z_near_zero, q_z_near_one, q_z_exact]:
        kl = prior.kl_divergence(q_z)

        assert torch.isfinite(kl), f"KL divergence must be finite, got {kl.item()}"
        assert kl.item() >= -1e-6, (
            f"KL divergence must be non-negative (allowing small numerical error), got {kl.item()}"
        )


def test_ibp_kl_stable_under_autocast_cpu() -> None:
    """Test KL divergence stability under CPU autocast (bfloat16)."""
    prior = IndianBuffetProcessPrior(alpha=1.0, max_communities=16)
    device = torch.device("cpu")

    # Create q_z with values that could cause underflow in float16
    q_z = torch.rand(8, 16, device=device) * 0.99 + 0.005

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        kl = prior.kl_divergence(q_z)

    assert torch.isfinite(kl), f"KL divergence must be finite under autocast, got {kl.item()}"
    assert kl.item() >= -1e-6, f"KL divergence must be non-negative, got {kl.item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ibp_kl_stable_under_autocast_cuda() -> None:
    """Test KL divergence stability under CUDA autocast (float16)."""
    prior = IndianBuffetProcessPrior(alpha=1.0, max_communities=16)
    device = torch.device("cuda")
    prior = prior.to(device)

    # Create q_z with values that could cause underflow in float16
    q_z = torch.rand(8, 16, device=device) * 0.99 + 0.005

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        kl = prior.kl_divergence(q_z)

    assert torch.isfinite(kl), f"KL divergence must be finite under autocast, got {kl.item()}"
    assert kl.item() >= -1e-6, f"KL divergence must be non-negative, got {kl.item()}"


def test_ibp_kl_consistency_float32_vs_mixed() -> None:
    """Test that KL divergence is consistent between float32 and mixed precision."""
    prior = IndianBuffetProcessPrior(alpha=1.0, max_communities=16)
    device = torch.device("cpu")

    q_z = torch.rand(8, 16, device=device) * 0.98 + 0.01

    # Compute in float32
    kl_f32 = prior.kl_divergence(q_z.float())

    # Compute under autocast (should internally use float32 for stability)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        kl_mixed = prior.kl_divergence(q_z.bfloat16())

    # Values should be close (allowing for small numerical differences)
    diff = abs(kl_f32.item() - kl_mixed.item())
    assert diff < 0.1, f"KL divergence should be consistent, diff={diff}"

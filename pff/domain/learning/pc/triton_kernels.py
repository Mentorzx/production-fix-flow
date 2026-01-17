"""
Triton-accelerated kernels for Probabilistic Circuits (PC2).

Provides fused kernels for exact inference on Hidden Chow-Liu Trees (HCLT),
avoiding massive intermediate tensor materialization during the forward pass.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


if TRITON_AVAILABLE:

    @triton.jit
    def _pc2_forward_kernel(
        pos_probs_ptr,
        parents_ptr,
        root_probs_ptr,
        cond_probs_ptr,
        output_y0_ptr,
        output_y1_ptr,
        prior_y0,
        prior_y1,
        NumAttrs,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute log P(Z, Y) for PC2 structure in a single fused pass."""
        pid = tl.program_id(0)

        row_offset = pid * NumAttrs

        acc_y0 = prior_y0
        acc_y1 = prior_y1

        for i in range(NumAttrs):
            p_val = tl.load(pos_probs_ptr + row_offset + i)
            p_neg = 1.0 - p_val

            parent_idx = tl.load(parents_ptr + i)

            if parent_idx == -1:
                r_y0 = tl.load(root_probs_ptr + i * 2 + 0)
                r_y1 = tl.load(root_probs_ptr + i * 2 + 1)

                term_y0 = p_val * tl.log(r_y0) + p_neg * tl.log(1.0 - r_y0)
                term_y1 = p_val * tl.log(r_y1) + p_neg * tl.log(1.0 - r_y1)

                acc_y0 += term_y0
                acc_y1 += term_y1
            else:
                p_parent = tl.load(pos_probs_ptr + row_offset + parent_idx)

                cp_p0_y0 = tl.load(cond_probs_ptr + i * 4 + 0 * 2 + 0)
                cp_p1_y0 = tl.load(cond_probs_ptr + i * 4 + 1 * 2 + 0)

                cp_p0_y1 = tl.load(cond_probs_ptr + i * 4 + 0 * 2 + 1)
                cp_p1_y1 = tl.load(cond_probs_ptr + i * 4 + 1 * 2 + 1)

                log_p1_y0 = p_val * tl.log(cp_p1_y0) + p_neg * tl.log(1.0 - cp_p1_y0)
                log_p0_y0 = p_val * tl.log(cp_p0_y0) + p_neg * tl.log(1.0 - cp_p0_y0)

                term_y0 = p_parent * log_p1_y0 + (1.0 - p_parent) * log_p0_y0
                acc_y0 += term_y0

                log_p1_y1 = p_val * tl.log(cp_p1_y1) + p_neg * tl.log(1.0 - cp_p1_y1)
                log_p0_y1 = p_val * tl.log(cp_p0_y1) + p_neg * tl.log(1.0 - cp_p0_y1)

                term_y1 = p_parent * log_p1_y1 + (1.0 - p_parent) * log_p0_y1
                acc_y1 += term_y1

        tl.store(output_y0_ptr + pid, acc_y0)
        tl.store(output_y1_ptr + pid, acc_y1)


def pc2_forward_triton(
    pos_probs: torch.Tensor,  # [Batch, NumAttrs]
    parents: torch.Tensor,  # [NumAttrs]
    root_probs: torch.Tensor,  # [NumAttrs, 2]
    cond_probs: torch.Tensor,  # [NumAttrs, 2, 2]
    log_prior: torch.Tensor,  # [2]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute PC2 forward pass using Triton."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")

    batch_size, num_attrs = pos_probs.shape

    out_y0 = torch.empty(batch_size, device=pos_probs.device, dtype=torch.float32)
    out_y1 = torch.empty(batch_size, device=pos_probs.device, dtype=torch.float32)

    grid = (batch_size,)

    prior_y0 = float(log_prior[0].item())
    prior_y1 = float(log_prior[1].item())

    _pc2_forward_kernel[grid](
        pos_probs,
        parents,
        root_probs,
        cond_probs,
        out_y0,
        out_y1,
        prior_y0,
        prior_y1,
        num_attrs,
        BLOCK_SIZE=128,
    )

    return out_y0, out_y1

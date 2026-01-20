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
    def _log_sigmoid(x):
        return -tl.log(1.0 + tl.exp(-x))

    @triton.jit
    def _log_one_minus_sigmoid(x):
        return -tl.log(1.0 + tl.exp(x))

    @triton.jit
    def _pc2_forward_kernel(  # noqa: N803
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

    @triton.jit
    def _pc2_matrix_forward_kernel(  # noqa: N803
        heads_probs_ptr,
        tails_probs_ptr,
        parents_ptr,
        root_probs_ptr,
        cond_probs_ptr,
        output_y1_ptr,
        prior_y1,
        NumHeads,
        NumTails,
        NumAttrs,
    ):
        """Compute log P(Z, Y=1) for all pairs (head, tail) in a single fused pass.

        Z is computed as 0.5 * (Z_head + Z_tail).
        """
        row_h = tl.program_id(0)
        row_t = tl.program_id(1)

        if row_h >= NumHeads or row_t >= NumTails:
            return

        acc_y1 = prior_y1

        for i in range(NumAttrs):
            z_h = tl.load(heads_probs_ptr + row_h * NumAttrs + i)
            z_t = tl.load(tails_probs_ptr + row_t * NumAttrs + i)

            p_val = 0.5 * (z_h + z_t)
            p_neg = 1.0 - p_val

            parent_idx = tl.load(parents_ptr + i)

            if parent_idx == -1:
                r_y1 = tl.load(root_probs_ptr + i * 2 + 1)
                term_y1 = p_val * tl.log(r_y1) + p_neg * tl.log(1.0 - r_y1)
                acc_y1 += term_y1
            else:
                z_h_p = tl.load(heads_probs_ptr + row_h * NumAttrs + parent_idx)
                z_t_p = tl.load(tails_probs_ptr + row_t * NumAttrs + parent_idx)
                p_parent = 0.5 * (z_h_p + z_t_p)

                cp_p0_y1 = tl.load(cond_probs_ptr + i * 4 + 0 * 2 + 1)
                cp_p1_y1 = tl.load(cond_probs_ptr + i * 4 + 1 * 2 + 1)

                log_p1_y1 = p_val * tl.log(cp_p1_y1) + p_neg * tl.log(1.0 - cp_p1_y1)
                log_p0_y1 = p_val * tl.log(cp_p0_y1) + p_neg * tl.log(1.0 - cp_p0_y1)

                term_y1 = p_parent * log_p1_y1 + (1.0 - p_parent) * log_p0_y1
                acc_y1 += term_y1

        tl.store(output_y1_ptr + row_h * NumTails + row_t, acc_y1)


def pc2_forward_triton(
    pos_probs: torch.Tensor,
    parents: torch.Tensor,
    root_probs: torch.Tensor,
    cond_probs: torch.Tensor,
    log_prior: torch.Tensor,
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


def pc2_matrix_forward_triton(
    heads_probs: torch.Tensor,
    tails_probs: torch.Tensor,
    parents: torch.Tensor,
    root_probs: torch.Tensor,
    cond_probs: torch.Tensor,
    log_prior_y1: float,
) -> torch.Tensor:
    """Execute pairwise PC2 forward pass (all pairs) using Triton."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")

    num_heads, num_attrs = heads_probs.shape
    num_tails = tails_probs.size(0)

    output = torch.empty((num_heads, num_tails), device=heads_probs.device, dtype=torch.float32)

    grid = (num_heads, num_tails)

    _pc2_matrix_forward_kernel[grid](
        heads_probs,
        tails_probs,
        parents,
        root_probs,
        cond_probs,
        output,
        log_prior_y1,
        num_heads,
        num_tails,
        num_attrs,
    )

    return output

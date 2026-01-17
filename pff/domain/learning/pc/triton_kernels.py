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
        pos_probs_ptr,  # [Batch, NumAttrs]
        parents_ptr,  # [NumAttrs]
        root_probs_ptr,  # [NumAttrs, 2] -> [attr, y]
        cond_probs_ptr,  # [NumAttrs, 2, 2] -> [attr, parent_val, y]
        output_y0_ptr,  # [Batch]
        output_y1_ptr,  # [Batch]
        prior_y0,  # scalar
        prior_y1,  # scalar
        NumAttrs,  # int
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute log P(Z, Y) for PC2 structure in a single fused pass."""
        pid = tl.program_id(0)

        # Offset for the current row in pos_probs
        row_offset = pid * NumAttrs

        # Initialize accumulators with priors
        acc_y0 = prior_y0
        acc_y1 = prior_y1

        # Iterate over all attributes
        # Since NumAttrs is usually small-ish (e.g., 512-2048), a simple loop
        # within the thread is efficient enough and avoids block reductions.
        # We process in chunks of BLOCK_SIZE if needed, but here we just loop linearly.
        # Note: Depending on NumAttrs, loop unrolling might happen.

        for i in range(NumAttrs):
            # Load current attribute prob P(Z_i = 1)
            # We assume inputs are already clamped/safe
            p_val = tl.load(pos_probs_ptr + row_offset + i)
            p_neg = 1.0 - p_val

            # Load parent index
            parent_idx = tl.load(parents_ptr + i)

            if parent_idx == -1:
                # Root Logic
                # root_probs: [NumAttrs, 2] -> offset: i * 2 + y
                r_y0 = tl.load(root_probs_ptr + i * 2 + 0)
                r_y1 = tl.load(root_probs_ptr + i * 2 + 1)

                # log P(Z_i | Y=0)
                term_y0 = p_val * tl.log(r_y0) + p_neg * tl.log(1.0 - r_y0)
                # log P(Z_i | Y=1)
                term_y1 = p_val * tl.log(r_y1) + p_neg * tl.log(1.0 - r_y1)

                acc_y0 += term_y0
                acc_y1 += term_y1
            else:
                # Child Logic
                # Load parent prob P(Z_parent = 1)
                # Random access within the row
                p_parent = tl.load(pos_probs_ptr + row_offset + parent_idx)

                # Conditional Probs: [NumAttrs, ParentVal, Y]
                # Stride: i * 4 + parent_val * 2 + y

                # P(Z_i=1 | P=0, Y=0)
                cp_p0_y0 = tl.load(cond_probs_ptr + i * 4 + 0 * 2 + 0)
                # P(Z_i=1 | P=1, Y=0)
                cp_p1_y0 = tl.load(cond_probs_ptr + i * 4 + 1 * 2 + 0)

                # P(Z_i=1 | P=0, Y=1)
                cp_p0_y1 = tl.load(cond_probs_ptr + i * 4 + 0 * 2 + 1)
                # P(Z_i=1 | P=1, Y=1)
                cp_p1_y1 = tl.load(cond_probs_ptr + i * 4 + 1 * 2 + 1)

                # Compute terms for Y=0
                # If parent=1: log P(Z_i | P=1, Y=0)
                log_p1_y0 = p_val * tl.log(cp_p1_y0) + p_neg * tl.log(1.0 - cp_p1_y0)
                # If parent=0: log P(Z_i | P=0, Y=0)
                log_p0_y0 = p_val * tl.log(cp_p0_y0) + p_neg * tl.log(1.0 - cp_p0_y0)

                # Expectation over parent state
                term_y0 = p_parent * log_p1_y0 + (1.0 - p_parent) * log_p0_y0
                acc_y0 += term_y0

                # Compute terms for Y=1
                log_p1_y1 = p_val * tl.log(cp_p1_y1) + p_neg * tl.log(1.0 - cp_p1_y1)
                log_p0_y1 = p_val * tl.log(cp_p0_y1) + p_neg * tl.log(1.0 - cp_p0_y1)

                term_y1 = p_parent * log_p1_y1 + (1.0 - p_parent) * log_p0_y1
                acc_y1 += term_y1

        # Store results
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

    # Outputs
    out_y0 = torch.empty(batch_size, device=pos_probs.device, dtype=torch.float32)
    out_y1 = torch.empty(batch_size, device=pos_probs.device, dtype=torch.float32)

    # Kernel config
    # We use 1 block per batch item.
    # For very large batches, this is plenty of parallelism.
    grid = (batch_size,)

    # Scalars
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
        BLOCK_SIZE=128,  # Hint, unused in simple loop but good practice
    )

    return out_y0, out_y1

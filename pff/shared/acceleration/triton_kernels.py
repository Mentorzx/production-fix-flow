from __future__ import annotations

import time
from pathlib import Path

import torch

from pff.shared.system.cuda import is_cuda_available
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = is_cuda_available()
except ImportError:
    TRITON_AVAILABLE = False
    triton, tl = None, None

_AUTOTUNE_CACHE: dict[tuple[int, int], int] = {}


def _next_power_of_2(n: int) -> int:
    """Return next power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def _benchmark_block_n(
    *,
    entity_re: torch.Tensor,
    entity_im: torch.Tensor,
    gamma: float,
    block_n: int,
    runs: int = 3,
    warmup: int = 1,
) -> float:
    if not TRITON_AVAILABLE or not entity_re.is_cuda:
        return float("inf")

    batch_size = min(128, int(entity_re.shape[0]))
    dim = int(entity_re.shape[1])
    device = entity_re.device
    query_re = torch.randn((batch_size, dim), device=device, dtype=entity_re.dtype)
    query_im = torch.randn((batch_size, dim), device=device, dtype=entity_re.dtype)
    tails = torch.randint(
        low=0,
        high=int(entity_re.shape[0]),
        size=(batch_size,),
        device=device,
        dtype=torch.int32,
    )

    ranks_out = torch.empty(batch_size, dtype=torch.int32, device=device)
    grid = (batch_size,)

    for _ in range(warmup):
        _dslfm_rank_kernel[grid](
            query_re,
            query_im,
            entity_re,
            entity_im,
            tails,
            ranks_out,
            gamma,
            NUM_ENTITIES=int(entity_re.shape[0]),
            DIM=dim,
            BLOCK_N=block_n,
            BLOCK_D=_next_power_of_2(dim),
        )
    torch.cuda.synchronize(device=device)

    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        _dslfm_rank_kernel[grid](
            query_re,
            query_im,
            entity_re,
            entity_im,
            tails,
            ranks_out,
            gamma,
            NUM_ENTITIES=int(entity_re.shape[0]),
            DIM=dim,
            BLOCK_N=block_n,
            BLOCK_D=_next_power_of_2(dim),
        )
        torch.cuda.synchronize(device=device)
        timings.append((time.perf_counter() - start) * 1000.0)

    return min(timings) if timings else float("inf")


def autotune_block_n(
    *,
    entity_re: torch.Tensor,
    entity_im: torch.Tensor,
    gamma: float,
    candidates: list[int] | None = None,
    bench_output_dir: Path | None = None,
) -> int:
    key = (int(entity_re.shape[0]), int(entity_re.shape[1]))
    cached = _AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    if candidates is None:
        candidates = [256, 512, 1024, 2048]

    timings: dict[int, float] = {}
    for block_n in candidates:
        timings[block_n] = _benchmark_block_n(
            entity_re=entity_re,
            entity_im=entity_im,
            gamma=gamma,
            block_n=block_n,
        )

    best_block = min(timings, key=lambda k: timings[k])
    _AUTOTUNE_CACHE[key] = best_block

    if bench_output_dir is not None:
        try:
            output_dir = bench_output_dir
            FileManager().ensure_dir(output_dir)
            import polars as pl

            payload = {
                "num_entities": key[0],
                "embedding_dim": key[1],
                "block_n": best_block,
                "timings_ms": timings,
            }
            df = pl.DataFrame([payload])
            FileManager().save(df, output_dir / "triton_block_sizes.parquet")
        except Exception as exc:
            logger.debug(f"Failed to persist Triton autotune data: {exc}")

    return best_block


if TRITON_AVAILABLE:

    @triton.jit
    def _dslfm_rank_kernel(
        Q_re_ptr,
        Q_im_ptr,
        E_re_ptr,
        E_im_ptr,
        T_idx_ptr,
        Rank_out_ptr,
        gamma,
        NUM_ENTITIES,
        DIM,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < DIM

        q_re = tl.load(Q_re_ptr + pid * DIM + offs_d, mask=mask_d, other=0.0)
        q_im = tl.load(Q_im_ptr + pid * DIM + offs_d, mask=mask_d, other=0.0)

        t_idx = tl.load(T_idx_ptr + pid)
        t_re = tl.load(E_re_ptr + t_idx * DIM + offs_d, mask=mask_d, other=0.0)
        t_im = tl.load(E_im_ptr + t_idx * DIM + offs_d, mask=mask_d, other=0.0)

        diff_re_target = q_re - t_re
        diff_im_target = q_im - t_im
        dist_sq_target = tl.sum(diff_re_target * diff_re_target + diff_im_target * diff_im_target)
        score_target = gamma - tl.sqrt(dist_sq_target)

        rank_acc = 0
        for block_start in range(0, NUM_ENTITIES, BLOCK_N):
            offs_n = block_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < NUM_ENTITIES

            e_re_ptrs = E_re_ptr + offs_n[:, None] * DIM + offs_d[None, :]
            e_im_ptrs = E_im_ptr + offs_n[:, None] * DIM + offs_d[None, :]
            mask_2d = mask_n[:, None] & mask_d[None, :]

            e_re_block = tl.load(e_re_ptrs, mask=mask_2d, other=0.0)
            e_im_block = tl.load(e_im_ptrs, mask=mask_2d, other=0.0)

            diff_re = q_re[None, :] - e_re_block
            diff_im = q_im[None, :] - e_im_block

            dist_sq = tl.sum(diff_re * diff_re + diff_im * diff_im, axis=1)
            scores = gamma - tl.sqrt(dist_sq)

            is_better = (scores > score_target) & mask_n
            rank_acc += tl.sum(is_better.to(tl.int32))

        tl.store(Rank_out_ptr + pid, (rank_acc + 1).to(tl.int32))

    @triton.jit
    def _dot_product_rank_kernel(
        Q_ptr,
        E_ptr,
        T_idx_ptr,
        Rank_out_ptr,
        NUM_ENTITIES,
        DIM,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < DIM

        q_vec = tl.load(Q_ptr + pid * DIM + offs_d, mask=mask_d, other=0.0)
        t_idx = tl.load(T_idx_ptr + pid)
        t_vec = tl.load(E_ptr + t_idx * DIM + offs_d, mask=mask_d, other=0.0)
        score_target = tl.sum(q_vec * t_vec)

        rank_acc = 0
        for block_start in range(0, NUM_ENTITIES, BLOCK_N):
            offs_n = block_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < NUM_ENTITIES

            e_ptrs = E_ptr + (offs_n[:, None] * DIM) + offs_d[None, :]
            mask_2d = mask_n[:, None] & mask_d[None, :]
            e_block = tl.load(e_ptrs, mask=mask_2d, other=0.0)

            scores = tl.sum(q_vec[None, :] * e_block, axis=1)
            is_better = (scores > score_target) & mask_n
            rank_acc += tl.sum(is_better.to(tl.int32))

        tl.store(Rank_out_ptr + pid, (rank_acc + 1).to(tl.int32))

    @triton.jit
    def fused_training_loss_kernel(
        H_re_ptr,
        H_im_ptr,
        Cos_ptr,
        Sin_ptr,
        T_re_ptr,
        T_im_ptr,
        Loss_out_ptr,
        gamma,
        N_BATCH,
        DIM,
        BLOCK_BATCH: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        o_b, o_d = pid * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH), tl.arange(0, BLOCK_D)
        m_b, m_d = o_b < N_BATCH, o_d < DIM

        h_re = tl.load(
            H_re_ptr + o_b[:, None] * DIM + o_d[None, :],
            mask=m_b[:, None] & m_d[None, :],
            other=0.0,
        )
        h_im = tl.load(
            H_im_ptr + o_b[:, None] * DIM + o_d[None, :],
            mask=m_b[:, None] & m_d[None, :],
            other=0.0,
        )
        cos = tl.load(
            Cos_ptr + o_b[:, None] * DIM + o_d[None, :], mask=m_b[:, None] & m_d[None, :], other=0.0
        )
        sin = tl.load(
            Sin_ptr + o_b[:, None] * DIM + o_d[None, :], mask=m_b[:, None] & m_d[None, :], other=0.0
        )

        q_re, q_im = h_re * cos - h_im * sin, h_re * sin + h_im * cos
        t_re = tl.load(
            T_re_ptr + o_b[:, None] * DIM + o_d[None, :],
            mask=m_b[:, None] & m_d[None, :],
            other=0.0,
        )
        t_im = tl.load(
            T_im_ptr + o_b[:, None] * DIM + o_d[None, :],
            mask=m_b[:, None] & m_d[None, :],
            other=0.0,
        )

        diff_re, diff_im = q_re - t_re, q_im - t_im
        s = gamma - tl.sqrt(tl.sum(diff_re * diff_re + diff_im * diff_im, axis=1))

        loss = tl.where(s > 0, -tl.log(1.0 + tl.exp(-s)), s - tl.log(1.0 + tl.exp(s)))
        tl.store(Loss_out_ptr + pid, tl.sum(loss))

    @triton.jit
    def _fused_subsample_kernel(
        input_ptr,
        output_ptr,
        n_rows,
        n_cols,
        k,
        seed,
        stride_row,
        stride_col,
        stride_out_row,
        stride_out_col,
        BLOCK_K: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= n_rows:
            return
        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < k
        rand_indices = tl.abs(tl.randint(seed + row_idx, offsets)) % n_cols
        input_offsets = row_idx * stride_row + rand_indices * stride_col
        vals = tl.load(input_ptr + input_offsets, mask=mask)
        tl.store(output_ptr + row_idx * stride_out_row + offsets * stride_out_col, vals, mask=mask)

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
        pid = tl.program_id(0)
        row_offset = pid * NumAttrs
        acc_y0, acc_y1 = prior_y0, prior_y1
        for i in range(NumAttrs):
            p_val = tl.load(pos_probs_ptr + row_offset + i)
            p_neg = 1.0 - p_val
            parent_idx = tl.load(parents_ptr + i)
            if parent_idx == -1:
                r_y0 = tl.load(root_probs_ptr + i * 2 + 0)
                r_y1 = tl.load(root_probs_ptr + i * 2 + 1)
                acc_y0 += p_val * tl.log(r_y0) + p_neg * tl.log(1.0 - r_y0)
                acc_y1 += p_val * tl.log(r_y1) + p_neg * tl.log(1.0 - r_y1)
            else:
                p_parent = tl.load(pos_probs_ptr + row_offset + parent_idx)
                cp_p0_y0 = tl.load(cond_probs_ptr + i * 4 + 0 * 2 + 0)
                cp_p1_y0 = tl.load(cond_probs_ptr + i * 4 + 1 * 2 + 0)
                cp_p0_y1 = tl.load(cond_probs_ptr + i * 4 + 0 * 2 + 1)
                cp_p1_y1 = tl.load(cond_probs_ptr + i * 4 + 1 * 2 + 1)
                log_p1_y0 = p_val * tl.log(cp_p1_y0) + p_neg * tl.log(1.0 - cp_p1_y0)
                log_p0_y0 = p_val * tl.log(cp_p0_y0) + p_neg * tl.log(1.0 - cp_p0_y0)
                acc_y0 += p_parent * log_p1_y0 + (1.0 - p_parent) * log_p0_y0
                log_p1_y1 = p_val * tl.log(cp_p1_y1) + p_neg * tl.log(1.0 - cp_p1_y1)
                log_p0_y1 = p_val * tl.log(cp_p0_y1) + p_neg * tl.log(1.0 - cp_p0_y1)
                acc_y1 += p_parent * log_p1_y1 + (1.0 - p_parent) * log_p0_y1
        tl.store(output_y0_ptr + pid, acc_y0)
        tl.store(output_y1_ptr + pid, acc_y1)

    @triton.jit
    def _pc2_matrix_forward_kernel(
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
        row_h, row_t = tl.program_id(0), tl.program_id(1)
        if row_h >= NumHeads or row_t >= NumTails:
            return
        acc_y1 = prior_y1
        for i in range(NumAttrs):
            p_val = 0.5 * (
                tl.load(heads_probs_ptr + row_h * NumAttrs + i)
                + tl.load(tails_probs_ptr + row_t * NumAttrs + i)
            )
            p_neg = 1.0 - p_val
            parent_idx = tl.load(parents_ptr + i)
            if parent_idx == -1:
                acc_y1 += p_val * tl.log(tl.load(root_probs_ptr + i * 2 + 1)) + p_neg * tl.log(
                    1.0 - tl.load(root_probs_ptr + i * 2 + 1)
                )
            else:
                p_parent = 0.5 * (
                    tl.load(heads_probs_ptr + row_h * NumAttrs + parent_idx)
                    + tl.load(tails_probs_ptr + row_t * NumAttrs + parent_idx)
                )
                log_p1_y1 = p_val * tl.log(
                    tl.load(cond_probs_ptr + i * 4 + 1 * 2 + 1)
                ) + p_neg * tl.log(1.0 - tl.load(cond_probs_ptr + i * 4 + 1 * 2 + 1))
                log_p0_y1 = p_val * tl.log(
                    tl.load(cond_probs_ptr + i * 4 + 0 * 2 + 1)
                ) + p_neg * tl.log(1.0 - tl.load(cond_probs_ptr + i * 4 + 0 * 2 + 1))
                acc_y1 += p_parent * log_p1_y1 + (1.0 - p_parent) * log_p0_y1
        tl.store(output_y1_ptr + row_h * NumTails + row_t, acc_y1)

    @triton.jit
    def _fused_rand_mask_kernel(
        scores_ptr,
        output_ptr,
        seed,
        N_COLS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        row_scores, row_out = scores_ptr + row_id * N_COLS, output_ptr + row_id * N_COLS
        row_seed = seed + row_id * 54321
        for off in range(0, N_COLS, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N_COLS
            is_valid = tl.load(row_scores + cols, mask=mask, other=float("-inf")) > -3.40282e38
            rand_float = ((row_seed + cols * 12345) * 1103515245 + 12345).to(
                tl.float32
            ) / 2147483648.0
            tl.store(row_out + cols, tl.where(is_valid, rand_float, -1.0), mask=mask)


def is_triton_available() -> bool:
    return TRITON_AVAILABLE


def fused_random_subsample_triton(
    scores: torch.Tensor, k: int, *, seed: int | None = None
) -> torch.Tensor:
    if not TRITON_AVAILABLE or not scores.is_cuda:
        batch_size, num_candidates = scores.shape
        gen = torch.Generator(device=scores.device)
        if seed is not None:
            gen.manual_seed(seed)
        if torch.isinf(scores).any():
            rand_keys = torch.rand(batch_size, num_candidates, device=scores.device, generator=gen)
            rand_keys = torch.where(
                torch.isfinite(scores), rand_keys, torch.full_like(rand_keys, -1.0)
            )
            _, idx = torch.topk(rand_keys, k=k, dim=1)
            return scores.gather(1, idx)
        else:
            idx = torch.stack(
                [
                    torch.randperm(num_candidates, device=scores.device, generator=gen)[:k]
                    for _ in range(batch_size)
                ]
            )
            return scores.gather(1, idx)

    batch_size, num_candidates = scores.shape
    rand_keys = torch.empty_like(scores, dtype=torch.float32)
    effective_seed = seed if seed is not None else int(time.time())
    _fused_rand_mask_kernel[(batch_size,)](
        scores, rand_keys, effective_seed, N_COLS=num_candidates, BLOCK_SIZE=1024
    )
    _, random_idx = torch.topk(rand_keys, k=k, dim=1)
    return scores.gather(1, random_idx)


def fused_dslfm_training_loss_triton(h_re, h_im, cos, sin, t_re, t_im, gamma) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    n_batch, dim = h_re.shape
    B_BATCH, B_D = 128, _next_power_of_2(dim)
    grid = (triton.cdiv(n_batch, B_BATCH),)
    loss_out = torch.empty(grid[0], device=h_re.device, dtype=torch.float32)
    fused_training_loss_kernel[grid](
        h_re, h_im, cos, sin, t_re, t_im, loss_out, gamma, n_batch, dim, B_BATCH, B_D
    )
    return -loss_out.sum() / n_batch


def pc2_forward_triton(
    pos_probs, parents, root_probs, cond_probs, log_prior
) -> tuple[torch.Tensor, torch.Tensor]:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    batch_size, num_attrs = pos_probs.shape
    out_y0, out_y1 = (
        torch.empty(batch_size, device=pos_probs.device),
        torch.empty(batch_size, device=pos_probs.device),
    )
    _pc2_forward_kernel[(batch_size,)](
        pos_probs,
        parents,
        root_probs,
        cond_probs,
        out_y0,
        out_y1,
        float(log_prior[0].item()),
        float(log_prior[1].item()),
        num_attrs,
        BLOCK_SIZE=128,
    )
    return out_y0, out_y1


def pc2_matrix_forward_triton(
    heads_probs, tails_probs, parents, root_probs, cond_probs, log_prior_y1
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    num_heads, num_attrs = heads_probs.shape
    num_tails = tails_probs.size(0)
    output = torch.empty((num_heads, num_tails), device=heads_probs.device)
    _pc2_matrix_forward_kernel[(num_heads, num_tails)](
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


class TritonDotProductValidator:
    def __init__(self, entity_embeddings, device="cuda", block_n=1024):
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available")
        self.device, self.entity_embeddings = device, entity_embeddings.contiguous().to(device)
        self.num_entities, self.dim = entity_embeddings.shape
        self.block_n, self.block_d = block_n, _next_power_of_2(self.dim)

    def compute_ranks(self, queries, true_tail_indices):
        batch_size = queries.shape[0]
        ranks_out = torch.empty(batch_size, dtype=torch.int32, device=self.device)
        _dot_product_rank_kernel[(batch_size,)](
            queries.contiguous().to(self.device),
            self.entity_embeddings,
            true_tail_indices.contiguous().to(self.device),
            ranks_out,
            self.num_entities,
            self.dim,
            self.block_n,
            self.block_d,
        )
        return ranks_out


class TritonDSLFMValidator:
    def __init__(
        self,
        entity_re,
        entity_im,
        gamma,
        device="cuda",
        block_n=1024,
        autotune=True,
        bench_output_dir=None,
    ):
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available")
        self.device, self.gamma, self.block_n = device, gamma, block_n
        self.entity_re, self.entity_im = (
            entity_re.contiguous().to(device),
            entity_im.contiguous().to(device),
        )
        self.num_entities, self.dim = entity_re.shape
        self.block_d = _next_power_of_2(self.dim)
        if autotune:
            self.block_n = autotune_block_n(
                entity_re=self.entity_re,
                entity_im=self.entity_im,
                gamma=self.gamma,
                bench_output_dir=bench_output_dir,
            )

    def compute_ranks(self, query_re, query_im, true_tail_indices) -> torch.Tensor:
        batch_size = query_re.shape[0]
        ranks_out = torch.empty(batch_size, dtype=torch.int32, device=self.device)
        _dslfm_rank_kernel[(batch_size,)](
            query_re.contiguous().to(self.device),
            query_im.contiguous().to(self.device),
            self.entity_re,
            self.entity_im,
            true_tail_indices.contiguous().to(self.device),
            ranks_out,
            self.gamma,
            self.num_entities,
            self.dim,
            self.block_n,
            self.block_d,
        )
        return ranks_out


def compute_ranks_from_scores_triton(scores: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
    """Fallback functional rank calculation."""
    true_scores = scores.gather(1, tails.unsqueeze(1))
    return (scores > true_scores).sum(dim=1) + 1


def fused_logsigmoid(x: torch.Tensor, negate: bool = False) -> torch.Tensor:
    if not TRITON_AVAILABLE or not x.is_cuda:
        raise RuntimeError("Triton/CUDA required")
    x_flat = x.contiguous().view(-1)
    output = torch.empty_like(x_flat)
    n_elements, BLOCK_SIZE = x_flat.numel(), 1024
    _fused_logsigmoid_kernel[((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)](
        x_flat, output, n_elements, negate, BLOCK_SIZE
    )
    return output.view_as(x)


if TRITON_AVAILABLE:

    @triton.jit
    def _fused_logsigmoid_kernel(
        scores_ptr, output_ptr, N, negate: tl.constexpr, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(scores_ptr + offsets, mask=mask, other=0.0)
        if negate:
            x = -x
        result = tl.where(x >= 0, -tl.log(1.0 + tl.exp(-x)), x - tl.log(1.0 + tl.exp(x)))
        tl.store(output_ptr + offsets, result, mask=mask)


def expected_calibration_error_fast(probs, labels, n_bins: int = 15) -> float:
    try:
        import numpy as np
        from numba import njit, prange

        @njit(cache=True, fastmath=True, parallel=True)
        def _ece_numba_kernel(probs, labels, n_bins):
            n = len(probs)
            if n == 0:
                return 0.0
            bin_sums, label_sums, bin_counts = (
                np.zeros(n_bins),
                np.zeros(n_bins),
                np.zeros(n_bins, dtype=np.int64),
            )
            bin_width = 1.0 / n_bins
            for i in prange(n):
                p = max(0.0, min(1.0, probs[i]))
                b = min(int(p / bin_width), n_bins - 1)
                bin_sums[b] += p
                label_sums[b] += labels[i]
                bin_counts[b] += 1
            ece = 0.0
            for b in range(n_bins):
                if bin_counts[b] > 0:
                    ece += (
                        bin_counts[b]
                        / n
                        * abs(label_sums[b] / bin_counts[b] - bin_sums[b] / bin_counts[b])
                    )
            return ece

        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        return _ece_numba_kernel(
            np.asarray(probs, dtype=np.float64).ravel(),
            np.asarray(labels, dtype=np.float64).ravel(),
            n_bins,
        )
    except Exception:
        raise RuntimeError("Numba not available for ECE.")

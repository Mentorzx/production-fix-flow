"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/shared/acceleration/triton_kernels.py

"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from pff.shared.system.cuda import is_cuda_available
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger

triton: Any = None
tl: Any = None

try:
    import triton as _triton
    import triton.language as _tl

    triton = _triton
    tl = _tl

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
    """Execute benchmark block n.



    Args:

        entity_re: Input value used by this callable.

        entity_im: Input value used by this callable.

        gamma: Input value used by this callable.

        block_n: Input value used by this callable.

        runs: Optional input value.

        warmup: Optional input value.



    Returns:

        Return value produced by the callable.

    """

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

    block_d = _next_power_of_2(dim)

    def _launch_rank_kernel() -> None:
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
            BLOCK_D=block_d,
        )

    for _ in range(warmup):
        _launch_rank_kernel()
    torch.cuda.synchronize(device=device)

    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        _launch_rank_kernel()
        torch.cuda.synchronize(device=device)
        timings.append((time.perf_counter() - start) * 1000.0)

    return min(timings) if timings else float("inf")


def _rank_chunk_size(num_entities: int, batch_size: int, dtype: torch.dtype) -> int:
    """Choose a CUDA chunk size for rank computation that avoids OOM."""
    if batch_size <= 0 or num_entities <= 0:
        return 1
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    if not torch.cuda.is_available():
        return min(num_entities, 4096)
    try:
        free_mem, _ = torch.cuda.mem_get_info()
    except Exception:
        return min(num_entities, 4096)

    if batch_size >= 512:
        desired_bytes = 32 * 1024 * 1024
    elif batch_size >= 256:
        desired_bytes = 24 * 1024 * 1024
    elif batch_size >= 128:
        desired_bytes = 16 * 1024 * 1024
    else:
        desired_bytes = 8 * 1024 * 1024

    target_bytes = min(int(free_mem * 0.2), desired_bytes)
    denom = max(batch_size * bytes_per, 1)
    chunk = target_bytes // denom
    chunk = max(1024, min(chunk, 65536))
    return min(num_entities, int(chunk))


def _compute_dot_ranks_chunked_cuda(
    queries: torch.Tensor,
    entity_embeddings: torch.Tensor,
    true_tail_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute exact ranks using chunked GEMM on CUDA."""
    batch_size = int(queries.shape[0])
    num_entities = int(entity_embeddings.shape[0])
    tails = true_tail_indices.to(device=queries.device, dtype=torch.long).contiguous()
    true_entities = entity_embeddings.index_select(0, tails)
    true_scores = (queries * true_entities).sum(dim=1)

    ranks = torch.zeros(batch_size, dtype=torch.int64, device=queries.device)
    chunk = _rank_chunk_size(num_entities, batch_size, queries.dtype)
    for start in range(0, num_entities, chunk):
        end = min(start + chunk, num_entities)
        scores = queries @ entity_embeddings[start:end].T
        ranks += (scores > true_scores.unsqueeze(1)).sum(dim=1, dtype=torch.int64)
    return (ranks + 1).to(torch.int32)


def _compute_dslfm_ranks_chunked_cuda(
    query_re: torch.Tensor,
    query_im: torch.Tensor,
    entity_re: torch.Tensor,
    entity_im: torch.Tensor,
    true_tail_indices: torch.Tensor,
    gamma: float,
    entity_norm_sq: torch.Tensor,
) -> torch.Tensor:
    """Compute exact DSLFM ranks with chunked CUDA GEMM and norm trick."""
    batch_size = int(query_re.shape[0])
    num_entities = int(entity_re.shape[0])
    tails = true_tail_indices.to(device=query_re.device, dtype=torch.long).contiguous()

    q_norm_sq = (query_re.square() + query_im.square()).sum(dim=1)
    true_dot = (query_re * entity_re.index_select(0, tails)).sum(dim=1) + (
        query_im * entity_im.index_select(0, tails)
    ).sum(dim=1)
    true_dist_sq = (q_norm_sq + entity_norm_sq.index_select(0, tails) - 2.0 * true_dot).clamp_min(
        0.0
    )
    true_scores = gamma - torch.sqrt(true_dist_sq)

    ranks = torch.zeros(batch_size, dtype=torch.int64, device=query_re.device)
    chunk = _rank_chunk_size(num_entities, batch_size, query_re.dtype)
    for start in range(0, num_entities, chunk):
        end = min(start + chunk, num_entities)
        dot = query_re @ entity_re[start:end].T
        dot += query_im @ entity_im[start:end].T
        dist_sq = (
            q_norm_sq.unsqueeze(1) + entity_norm_sq[start:end].unsqueeze(0) - 2.0 * dot
        ).clamp_min(0.0)
        scores = gamma - torch.sqrt(dist_sq)
        ranks += (scores > true_scores.unsqueeze(1)).sum(dim=1, dtype=torch.int64)
    return (ranks + 1).to(torch.int32)


def autotune_block_n(
    *,
    entity_re: torch.Tensor,
    entity_im: torch.Tensor,
    gamma: float,
    candidates: list[int] | None = None,
    bench_output_dir: Path | None = None,
) -> int:
    """Execute autotune block n.



    Args:

        entity_re: Input value used by this callable.

        entity_im: Input value used by this callable.

        gamma: Input value used by this callable.

        candidates: Optional input value.

        bench_output_dir: Optional input value.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    key = (int(entity_re.shape[0]), int(entity_re.shape[1]))
    cached = _AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    num_entities, dim = key
    if num_entities <= 4096:
        heuristic_block = 256 if dim <= 256 else 512
        _AUTOTUNE_CACHE[key] = heuristic_block
        return heuristic_block

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


if TRITON_AVAILABLE and triton is not None and tl is not None:
    _RANK_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ]

    @triton.autotune(
        configs=_RANK_AUTOTUNE_CONFIGS,
        key=["NUM_ENTITIES", "DIM", "BLOCK_N"],
    )
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
        BLOCK_N: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        BLOCK_D: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        """Execute dslfm rank kernel.



        Args:

            Q_re_ptr: Input value used by this callable.

            Q_im_ptr: Input value used by this callable.

            E_re_ptr: Input value used by this callable.

            E_im_ptr: Input value used by this callable.

            T_idx_ptr: Input value used by this callable.

            Rank_out_ptr: Input value used by this callable.

            gamma: Input value used by this callable.

            NUM_ENTITIES: Input value used by this callable.

            DIM: Input value used by this callable.

            BLOCK_N: Input value used by this callable.

            BLOCK_D: Input value used by this callable.

        """

        pid = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        offs_d = tl.max_contiguous(tl.multiple_of(offs_d, 8), 8)
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
            offs_n = tl.max_contiguous(tl.multiple_of(offs_n, 8), 8)
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
            rank_acc += tl.sum(is_better.to(tl.int32))  # type: ignore[attr-defined]

        tl.store(Rank_out_ptr + pid, (rank_acc + 1).to(tl.int32))  # type: ignore[attr-defined]

    @triton.autotune(
        configs=_RANK_AUTOTUNE_CONFIGS,
        key=["NUM_ENTITIES", "DIM", "BLOCK_N"],
    )
    @triton.jit
    def _dot_product_rank_kernel(
        Q_ptr,
        E_ptr,
        T_idx_ptr,
        Rank_out_ptr,
        NUM_ENTITIES,
        DIM,
        BLOCK_N: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        BLOCK_D: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        """Execute dot product rank kernel.



        Args:

            Q_ptr: Input value used by this callable.

            E_ptr: Input value used by this callable.

            T_idx_ptr: Input value used by this callable.

            Rank_out_ptr: Input value used by this callable.

            NUM_ENTITIES: Input value used by this callable.

            DIM: Input value used by this callable.

            BLOCK_N: Input value used by this callable.

            BLOCK_D: Input value used by this callable.

        """

        pid = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        offs_d = tl.max_contiguous(tl.multiple_of(offs_d, 8), 8)
        mask_d = offs_d < DIM

        q_vec = tl.load(Q_ptr + pid * DIM + offs_d, mask=mask_d, other=0.0)
        t_idx = tl.load(T_idx_ptr + pid)
        t_vec = tl.load(E_ptr + t_idx * DIM + offs_d, mask=mask_d, other=0.0)
        score_target = tl.sum(q_vec * t_vec)

        rank_acc = 0
        for block_start in range(0, NUM_ENTITIES, BLOCK_N):
            offs_n = block_start + tl.arange(0, BLOCK_N)
            offs_n = tl.max_contiguous(tl.multiple_of(offs_n, 8), 8)
            mask_n = offs_n < NUM_ENTITIES

            e_ptrs = E_ptr + (offs_n[:, None] * DIM) + offs_d[None, :]
            mask_2d = mask_n[:, None] & mask_d[None, :]
            e_block = tl.load(e_ptrs, mask=mask_2d, other=0.0)

            scores = tl.sum(q_vec[None, :] * e_block, axis=1)
            is_better = (scores > score_target) & mask_n
            rank_acc += tl.sum(is_better.to(tl.int32))  # type: ignore[attr-defined]

        tl.store(Rank_out_ptr + pid, (rank_acc + 1).to(tl.int32))  # type: ignore[attr-defined]

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
        BLOCK_BATCH: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        BLOCK_D: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        """Execute fused training loss kernel.



        Args:

            H_re_ptr: Input value used by this callable.

            H_im_ptr: Input value used by this callable.

            Cos_ptr: Input value used by this callable.

            Sin_ptr: Input value used by this callable.

            T_re_ptr: Input value used by this callable.

            T_im_ptr: Input value used by this callable.

            Loss_out_ptr: Input value used by this callable.

            gamma: Input value used by this callable.

            N_BATCH: Input value used by this callable.

            DIM: Input value used by this callable.

            BLOCK_BATCH: Input value used by this callable.

            BLOCK_D: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

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
            Cos_ptr + o_b[:, None] * DIM + o_d[None, :],
            mask=m_b[:, None] & m_d[None, :],
            other=0.0,
        )
        sin = tl.load(
            Sin_ptr + o_b[:, None] * DIM + o_d[None, :],
            mask=m_b[:, None] & m_d[None, :],
            other=0.0,
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
        BLOCK_K: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        """Execute fused subsample kernel.



        Args:

            input_ptr: Input value used by this callable.

            output_ptr: Input value used by this callable.

            n_rows: Input value used by this callable.

            n_cols: Input value used by this callable.

            k: Input value used by this callable.

            seed: Input value used by this callable.

            stride_row: Input value used by this callable.

            stride_col: Input value used by this callable.

            stride_out_row: Input value used by this callable.

            stride_out_col: Input value used by this callable.

            BLOCK_K: Input value used by this callable.

        """

        row_idx = tl.program_id(0)
        if row_idx >= n_rows:
            return
        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < k
        rand_indices = tl.abs(tl.randint(seed + row_idx, offsets)) % n_cols
        input_offsets = row_idx * stride_row + rand_indices * stride_col
        vals = tl.load(input_ptr + input_offsets, mask=mask)
        tl.store(
            output_ptr + row_idx * stride_out_row + offsets * stride_out_col,
            vals,
            mask=mask,
        )

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
        BLOCK_SIZE: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        """Execute pc2 forward kernel.



        Args:

            pos_probs_ptr: Input value used by this callable.

            parents_ptr: Input value used by this callable.

            root_probs_ptr: Input value used by this callable.

            cond_probs_ptr: Input value used by this callable.

            output_y0_ptr: Input value used by this callable.

            output_y1_ptr: Input value used by this callable.

            prior_y0: Input value used by this callable.

            prior_y1: Input value used by this callable.

            NumAttrs: Input value used by this callable.

            BLOCK_SIZE: Input value used by this callable.

        """

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
        root_log_ptr,
        root_log_inv_ptr,
        cond_log_ptr,
        cond_log_inv_ptr,
        output_y1_ptr,
        prior_y1,
        NumHeads,
        NumTails,
        NumAttrs,
        BLOCK_A: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        """Execute pc2 matrix forward kernel.



        Args:

            heads_probs_ptr: Input value used by this callable.

            tails_probs_ptr: Input value used by this callable.

            parents_ptr: Input value used by this callable.

            root_probs_ptr: Input value used by this callable.

            cond_probs_ptr: Input value used by this callable.

            output_y1_ptr: Input value used by this callable.

            prior_y1: Input value used by this callable.

            NumHeads: Input value used by this callable.

            NumTails: Input value used by this callable.

            NumAttrs: Input value used by this callable.

        """

        row_h, row_t = tl.program_id(0), tl.program_id(1)
        if row_h >= NumHeads or row_t >= NumTails:
            return
        acc_y1 = prior_y1
        for a_start in range(0, NumAttrs, BLOCK_A):
            offs_a = a_start + tl.arange(0, BLOCK_A)
            mask_a = offs_a < NumAttrs

            head_vals = tl.load(
                heads_probs_ptr + row_h * NumAttrs + offs_a,
                mask=mask_a,
                other=0.0,
            )
            tail_vals = tl.load(
                tails_probs_ptr + row_t * NumAttrs + offs_a,
                mask=mask_a,
                other=0.0,
            )
            p_val = 0.5 * (head_vals + tail_vals)
            p_neg = 1.0 - p_val

            parent_idx = tl.load(parents_ptr + offs_a, mask=mask_a, other=-1)
            is_root = parent_idx == -1
            parent_idx_safe = tl.where(is_root, 0, parent_idx)

            r_y1 = tl.load(root_log_ptr + offs_a * 2 + 1, mask=mask_a, other=0.0)
            r_y1_inv = tl.load(root_log_inv_ptr + offs_a * 2 + 1, mask=mask_a, other=0.0)
            root_term = p_val * r_y1 + p_neg * r_y1_inv

            head_parent = tl.load(
                heads_probs_ptr + row_h * NumAttrs + parent_idx_safe,
                mask=mask_a,
                other=0.0,
            )
            tail_parent = tl.load(
                tails_probs_ptr + row_t * NumAttrs + parent_idx_safe,
                mask=mask_a,
                other=0.0,
            )
            p_parent = 0.5 * (head_parent + tail_parent)

            log_p1_y1 = p_val * tl.load(
                cond_log_ptr + offs_a * 4 + 1 * 2 + 1,
                mask=mask_a,
                other=0.0,
            ) + p_neg * tl.load(
                cond_log_inv_ptr + offs_a * 4 + 1 * 2 + 1,
                mask=mask_a,
                other=0.0,
            )
            log_p0_y1 = p_val * tl.load(
                cond_log_ptr + offs_a * 4 + 0 * 2 + 1,
                mask=mask_a,
                other=0.0,
            ) + p_neg * tl.load(
                cond_log_inv_ptr + offs_a * 4 + 0 * 2 + 1,
                mask=mask_a,
                other=0.0,
            )
            child_term = p_parent * log_p1_y1 + (1.0 - p_parent) * log_p0_y1

            term = tl.where(is_root, root_term, child_term)
            acc_y1 += tl.sum(tl.where(mask_a, term, 0.0))
        tl.store(output_y1_ptr + row_h * NumTails + row_t, acc_y1)

    @triton.jit
    def _fused_rand_mask_kernel(
        scores_ptr,
        output_ptr,
        seed,
        N_COLS: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        BLOCK_SIZE: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        """Execute fused rand mask kernel.



        Args:

            scores_ptr: Input value used by this callable.

            output_ptr: Input value used by this callable.

            seed: Input value used by this callable.

            N_COLS: Input value used by this callable.

            BLOCK_SIZE: Input value used by this callable.

        """

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
    """Execute is triton available.



    Returns:

        Return value produced by the callable.

    """

    return TRITON_AVAILABLE


def fused_random_subsample_triton(
    scores: torch.Tensor, k: int, *, seed: int | None = None
) -> torch.Tensor:
    """Execute fused random subsample triton.



    Args:

        scores: Input value used by this callable.

        k: Input value used by this callable.

        seed: Optional input value.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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
    """Execute fused dslfm training loss triton.



    Args:

        h_re: Input value used by this callable.

        h_im: Input value used by this callable.

        cos: Input value used by this callable.

        sin: Input value used by this callable.

        t_re: Input value used by this callable.

        t_im: Input value used by this callable.

        gamma: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Raises:

        Exception: Propagates domain-specific failures with context.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    n_batch, dim = h_re.shape
    B_BATCH, B_D = 128, _next_power_of_2(dim)
    grid = (triton.cdiv(n_batch, B_BATCH),)
    loss_out = torch.empty(grid[0], device=h_re.device, dtype=torch.float32)
    fused_training_loss_kernel[grid](
        h_re, h_im, cos, sin, t_re, t_im, loss_out, gamma, n_batch, dim, B_BATCH, B_D
    )
    return -loss_out.sum() / n_batch  # type: ignore[no-any-return]


def pc2_forward_triton(
    pos_probs, parents, root_probs, cond_probs, log_prior
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute pc2 forward triton.



    Args:

        pos_probs: Input value used by this callable.

        parents: Input value used by this callable.

        root_probs: Input value used by this callable.

        cond_probs: Input value used by this callable.

        log_prior: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Raises:

        Exception: Propagates domain-specific failures with context.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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
    return out_y0, out_y1  # type: ignore[return-value]


def pc2_matrix_forward_triton(
    heads_probs, tails_probs, parents, root_probs, cond_probs, log_prior_y1
) -> torch.Tensor:
    """Execute pc2 matrix forward triton.



    Args:

        heads_probs: Input value used by this callable.

        tails_probs: Input value used by this callable.

        parents: Input value used by this callable.

        root_probs: Input value used by this callable.

        cond_probs: Input value used by this callable.

        log_prior_y1: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Raises:

        Exception: Propagates domain-specific failures with context.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    num_heads, num_attrs = heads_probs.shape
    num_tails = tails_probs.size(0)
    root_log = torch.log(root_probs)
    root_log_inv = torch.log1p(-root_probs)
    cond_log = torch.log(cond_probs)
    cond_log_inv = torch.log1p(-cond_probs)
    output = torch.empty((num_heads, num_tails), device=heads_probs.device)
    _pc2_matrix_forward_kernel[(num_heads, num_tails)](
        heads_probs,
        tails_probs,
        parents,
        root_log,
        root_log_inv,
        cond_log,
        cond_log_inv,
        output,
        log_prior_y1,
        num_heads,
        num_tails,
        num_attrs,
        BLOCK_A=64,
    )
    return output


class TritonDotProductValidator:
    """Represent TritonDotProductValidator."""

    def __init__(self, entity_embeddings, device="cuda", block_n=1024):
        """Execute init.



        Args:

            entity_embeddings: Input value used by this callable.

            device: Optional input value.

            block_n: Optional input value.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available")
        self.device, self.entity_embeddings = device, entity_embeddings.contiguous().to(device)
        self.num_entities, self.dim = entity_embeddings.shape
        self.block_n, self.block_d = block_n, _next_power_of_2(self.dim)

    def compute_ranks(self, queries, true_tail_indices):
        """Execute compute ranks.



        Args:

            queries: Input value used by this callable.

            true_tail_indices: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return _compute_dot_ranks_chunked_cuda(
            queries=queries.contiguous().to(self.device),
            entity_embeddings=self.entity_embeddings,
            true_tail_indices=true_tail_indices.contiguous().to(self.device),
        )


class TritonDSLFMValidator:
    """Represent TritonDSLFMValidator."""

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
        """Execute init.



        Args:

            entity_re: Input value used by this callable.

            entity_im: Input value used by this callable.

            gamma: Input value used by this callable.

            device: Optional input value.

            block_n: Optional input value.

            autotune: Optional input value.

            bench_output_dir: Optional input value.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available")
        self.device, self.gamma, self.block_n = device, gamma, block_n
        self.entity_re, self.entity_im = (
            entity_re.contiguous().to(device),
            entity_im.contiguous().to(device),
        )
        self.num_entities, self.dim = entity_re.shape
        self.block_d = _next_power_of_2(self.dim)
        self.entity_norm_sq = (self.entity_re.square() + self.entity_im.square()).sum(dim=1)
        if autotune:
            self.block_n = autotune_block_n(
                entity_re=self.entity_re,
                entity_im=self.entity_im,
                gamma=self.gamma,
                bench_output_dir=bench_output_dir,
            )

    def compute_ranks(self, query_re, query_im, true_tail_indices) -> torch.Tensor:
        """Execute compute ranks.



        Args:

            query_re: Input value used by this callable.

            query_im: Input value used by this callable.

            true_tail_indices: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return _compute_dslfm_ranks_chunked_cuda(
            query_re=query_re.contiguous().to(self.device),
            query_im=query_im.contiguous().to(self.device),
            entity_re=self.entity_re,
            entity_im=self.entity_im,
            true_tail_indices=true_tail_indices.contiguous().to(self.device),
            gamma=self.gamma,
            entity_norm_sq=self.entity_norm_sq,
        )


def compute_ranks_from_scores_triton(scores: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
    """Fallback functional rank calculation."""
    true_scores = scores.gather(1, tails.unsqueeze(1))
    return (scores > true_scores).sum(dim=1) + 1


def fused_logsigmoid(x: torch.Tensor, negate: bool = False) -> torch.Tensor:
    """Execute fused logsigmoid.



    Args:

        x: Input value used by this callable.

        negate: Optional input value.



    Returns:

        Return value produced by the callable.



    Raises:

        Exception: Propagates domain-specific failures with context.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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
        scores_ptr,
        output_ptr,
        N,
        negate: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        BLOCK_SIZE: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        """Execute fused logsigmoid kernel.



        Args:

            scores_ptr: Input value used by this callable.

            output_ptr: Input value used by this callable.

            N: Input value used by this callable.

            negate: Input value used by this callable.

            BLOCK_SIZE: Input value used by this callable.

        """

        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(scores_ptr + offsets, mask=mask, other=0.0)
        if negate:
            x = -x
        result = tl.where(x >= 0, -tl.log(1.0 + tl.exp(-x)), x - tl.log(1.0 + tl.exp(x)))
        tl.store(output_ptr + offsets, result, mask=mask)


def expected_calibration_error_fast(probs, labels, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error using vectorized NumPy.

    Args:
        probs: Predicted probabilities (tensor or array).
        labels: Ground truth binary labels (tensor or array).
        n_bins: Number of calibration bins.

    Returns:
        ECE scalar value.
    """
    import numpy as np

    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()

    n = len(probs)
    if n == 0:
        return 0.0

    probs = np.clip(probs, 0.0, 1.0)
    bin_indices = np.minimum((probs * n_bins).astype(np.int64), n_bins - 1)

    bin_sums = np.bincount(bin_indices, weights=probs, minlength=n_bins)
    label_sums = np.bincount(bin_indices, weights=labels, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    mask = bin_counts > 0
    ece = np.sum(
        bin_counts[mask]
        / n
        * np.abs(label_sums[mask] / bin_counts[mask] - bin_sums[mask] / bin_counts[mask])
    )
    return float(ece)

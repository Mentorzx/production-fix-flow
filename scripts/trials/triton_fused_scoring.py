import torch
import triton
import triton.language as tl
import time


@triton.jit
def _fused_score_kernel(
    H_re_ptr,
    H_im_ptr,
    Cos_ptr,
    Sin_ptr,
    T_re_ptr,
    T_im_ptr,
    Scores_ptr,
    gamma,
    N_QUERIES,
    N_ENTITIES,
    DIM,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rd = tl.arange(0, BLOCK_D)

    # Load and rotate
    h_re = tl.load(
        H_re_ptr + rm[:, None] * DIM + rd[None, :],
        mask=(rm[:, None] < N_QUERIES) & (rd[None, :] < DIM),
        other=0.0,
    )
    h_im = tl.load(
        H_im_ptr + rm[:, None] * DIM + rd[None, :],
        mask=(rm[:, None] < N_QUERIES) & (rd[None, :] < DIM),
        other=0.0,
    )
    cos = tl.load(
        Cos_ptr + rm[:, None] * DIM + rd[None, :],
        mask=(rm[:, None] < N_QUERIES) & (rd[None, :] < DIM),
        other=0.0,
    )
    sin = tl.load(
        Sin_ptr + rm[:, None] * DIM + rd[None, :],
        mask=(rm[:, None] < N_QUERIES) & (rd[None, :] < DIM),
        other=0.0,
    )

    q_re = h_re * cos - h_im * sin
    q_im = h_re * sin + h_im * cos

    # ||q||^2
    q_norm_sq = tl.sum(q_re * q_re + q_im * q_im, axis=1)  # [BLOCK_M]

    # Load entities
    # Note: tl.dot requires second operand to be [K, N] if first is [M, K]
    # Here M=BLOCK_M, K=BLOCK_D, N=BLOCK_N
    t_re = tl.load(
        T_re_ptr + rn[None, :] * DIM + rd[:, None],
        mask=(rn[None, :] < N_ENTITIES) & (rd[:, None] < DIM),
        other=0.0,
    )
    t_im = tl.load(
        T_im_ptr + rn[None, :] * DIM + rd[:, None],
        mask=(rn[None, :] < N_ENTITIES) & (rd[:, None] < DIM),
        other=0.0,
    )

    # ||t||^2
    t_norm_sq = tl.sum(t_re * t_re + t_im * t_im, axis=0)  # [BLOCK_N]

    # dot = q @ t (where t is already loaded as [DIM, BLOCK_N])
    dot_re = tl.dot(q_re, t_re)
    dot_im = tl.dot(q_im, t_im)

    dist_sq = q_norm_sq[:, None] + t_norm_sq[None, :] - 2.0 * (dot_re + dot_im)
    dist_sq = tl.maximum(dist_sq, 0.0)
    scores = gamma - tl.sqrt(dist_sq)

    tl.store(
        Scores_ptr + rm[:, None] * N_ENTITIES + rn[None, :],
        scores,
        mask=(rm[:, None] < N_QUERIES) & (rn[None, :] < N_ENTITIES),
    )


def fused_score_triton(h_re, h_im, cos, sin, t_re, t_im, gamma):
    N_QUERIES, DIM = h_re.shape
    N_ENTITIES, _ = t_re.shape

    scores = torch.empty((N_QUERIES, N_ENTITIES), device=h_re.device, dtype=h_re.dtype)

    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_D = triton.next_power_of_2(DIM)
    if BLOCK_D < 16:
        BLOCK_D = 16

    grid = (triton.cdiv(N_QUERIES, BLOCK_M), triton.cdiv(N_ENTITIES, BLOCK_N))

    _fused_score_kernel[grid](
        h_re,
        h_im,
        cos,
        sin,
        t_re,
        t_im,
        scores,
        gamma,
        N_QUERIES,
        N_ENTITIES,
        DIM,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
    )
    return scores


if __name__ == "__main__":
    device = torch.device("cuda")
    batch_size = 512
    num_entities = 20000
    dim = 256
    gamma = 12.0

    h_re = torch.randn(batch_size, dim, device=device)
    h_im = torch.randn(batch_size, dim, device=device)
    t_re = torch.randn(num_entities, dim, device=device)
    t_im = torch.randn(num_entities, dim, device=device)
    cos = torch.randn(batch_size, dim, device=device)
    sin = torch.randn(batch_size, dim, device=device)

    # Warmup
    _ = fused_score_triton(h_re, h_im, cos, sin, t_re, t_im, gamma)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(20):
        _ = fused_score_triton(h_re, h_im, cos, sin, t_re, t_im, gamma)
    torch.cuda.synchronize()
    print(f"TRITON_FUSED_SCORING_MS: {(time.perf_counter() - start) * 50.0:.2f}")

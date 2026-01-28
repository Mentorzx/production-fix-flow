import torch
import triton
import triton.language as tl
import time


@triton.jit
def _fused_joint_minimal_kernel(
    H_re_ptr,
    H_im_ptr,
    Cos_ptr,
    Sin_ptr,
    E_re_ptr,
    E_im_ptr,
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

    mask_m = rm[:, None] < N_QUERIES
    mask_n = rn[None, :] < N_ENTITIES
    mask_d = rd < DIM

    # Minimal load to test memory pressure
    h_re = tl.load(
        H_re_ptr + rm[:, None] * DIM + rd[None, :], mask=mask_m & mask_d[None, :], other=0.0
    )
    e_re = tl.load(
        E_re_ptr + rn[None, :] * DIM + rd[:, None], mask=mask_n & mask_d[:, None], other=0.0
    )

    # Dot product is the core memory consumer in Triton
    dot = tl.dot(h_re, e_re)
    scores = gamma - dot

    tl.store(Scores_ptr + rm[:, None] * N_ENTITIES + rn[None, :], scores, mask=mask_m & mask_n)


def run_fused(h_re, h_im, cos, sin, t_re, t_im, gamma):
    N_QUERIES, DIM = h_re.shape
    N_ENTITIES, _ = t_re.shape
    scores = torch.empty((N_QUERIES, N_ENTITIES), device=h_re.device, dtype=h_re.dtype)
    # Even smaller blocks
    BLOCK_M, BLOCK_N, BLOCK_D = 16, 16, 32
    grid = (triton.cdiv(N_QUERIES, BLOCK_M), triton.cdiv(N_ENTITIES, BLOCK_N))
    _fused_joint_minimal_kernel[grid](
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
    batch_size, n_ent, dim = 512, 20000, 256
    gamma = 12.0
    h_re, h_im = (
        torch.randn(batch_size, dim, device=device),
        torch.randn(batch_size, dim, device=device),
    )
    t_re, t_im = torch.randn(n_ent, dim, device=device), torch.randn(n_ent, dim, device=device)
    cos, sin = (
        torch.randn(batch_size, dim, device=device),
        torch.randn(batch_size, dim, device=device),
    )

    _ = run_fused(h_re, h_im, cos, sin, t_re, t_im, gamma)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        _ = run_fused(h_re, h_im, cos, sin, t_re, t_im, gamma)
    torch.cuda.synchronize()
    print(f"FUSED_MINIMAL_MS: {(time.perf_counter() - start) * 20.0:.2f}")

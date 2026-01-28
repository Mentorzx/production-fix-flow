import torch
import triton
import triton.language as tl
import time


@triton.jit
def _fused_loss_kernel(
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
    offs_b = pid * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    offs_d = tl.arange(0, BLOCK_D)

    mask_b = offs_b < N_BATCH
    mask_d = offs_d < DIM

    # 1. Load data
    h_re = tl.load(
        H_re_ptr + offs_b[:, None] * DIM + offs_d[None, :],
        mask=mask_b[:, None] & mask_d[None, :],
        other=0.0,
    )
    h_im = tl.load(
        H_im_ptr + offs_b[:, None] * DIM + offs_d[None, :],
        mask=mask_b[:, None] & mask_d[None, :],
        other=0.0,
    )
    cos = tl.load(
        Cos_ptr + offs_b[:, None] * DIM + offs_d[None, :],
        mask=mask_b[:, None] & mask_d[None, :],
        other=0.0,
    )
    sin = tl.load(
        Sin_ptr + offs_b[:, None] * DIM + offs_d[None, :],
        mask=mask_b[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Rotation
    q_re = h_re * cos - h_im * sin
    q_im = h_re * sin + h_im * cos

    # Positive sample load
    t_re = tl.load(
        T_re_ptr + offs_b[:, None] * DIM + offs_d[None, :],
        mask=mask_b[:, None] & mask_d[None, :],
        other=0.0,
    )
    t_im = tl.load(
        T_im_ptr + offs_b[:, None] * DIM + offs_d[None, :],
        mask=mask_b[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Score = gamma - dist
    diff_re = q_re - t_re
    diff_im = q_im - t_im
    dist_sq = tl.sum(diff_re * diff_re + diff_im * diff_im, axis=1)
    score = gamma - tl.sqrt(dist_sq)

    # Log-Sigmoid reduction
    # log(sigmoid(x)) = -softplus(-x)
    loss = -tl.log(1.0 + tl.exp(-score))

    # Reduction per block
    tl.store(Loss_out_ptr + pid, tl.sum(loss))


def run_fused_loss(h_re, h_im, cos, sin, t_re, t_im, gamma):
    N_BATCH, DIM = h_re.shape
    BLOCK_BATCH = 128
    BLOCK_D = triton.next_power_of_2(DIM)

    grid = (triton.cdiv(N_BATCH, BLOCK_BATCH),)
    loss_out = torch.empty(grid[0], device=h_re.device, dtype=torch.float32)

    _fused_loss_kernel[grid](
        h_re, h_im, cos, sin, t_re, t_im, loss_out, gamma, N_BATCH, DIM, BLOCK_BATCH, BLOCK_D
    )
    return loss_out.sum()


if __name__ == "__main__":
    device = torch.device("cuda")
    batch_size, dim = 2048, 512
    gamma = 12.0
    h_re, h_im = (
        torch.randn(batch_size, dim, device=device),
        torch.randn(batch_size, dim, device=device),
    )
    t_re, t_im = (
        torch.randn(batch_size, dim, device=device),
        torch.randn(batch_size, dim, device=device),
    )
    cos, sin = (
        torch.randn(batch_size, dim, device=device),
        torch.randn(batch_size, dim, device=device),
    )

    _ = run_fused_loss(h_re, h_im, cos, sin, t_re, t_im, gamma)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = run_fused_loss(h_re, h_im, cos, sin, t_re, t_im, gamma)
    torch.cuda.synchronize()
    print(f"FUSED_LOSS_TRIAL_MS: {(time.perf_counter() - start) * 10.0:.2f}")

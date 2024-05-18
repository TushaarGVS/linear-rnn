from typing import Any

import torch
import triton
import triton.language as tl
from einops import rearrange

from linear_rnn.utils.utils import is_power_of_2


@triton.jit
def _sequential_scan_block_diag_a_fwd_kernel(
    x_ptr,
    a_ptr,
    out_ptr,
    stride_x_batch,
    stride_x_len,
    stride_x_block,
    stride_x_dim_m,
    stride_a_batch,
    stride_a_len,
    stride_a_block,
    stride_a_dim_m,
    stride_a_dim_n,
    stride_out_batch,
    stride_out_len,
    stride_out_block,
    stride_out_dim_m,
    block_dim: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Sequential scan (with block-diagonal A) forward kernel, process BLOCK_SIZE_DIM (x) elements."""
    # Grid: (batch, num_blocks, dim/BLOCK_SIZE_M)
    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)
    pid_dim_m = tl.program_id(2)

    # Move all ptrs to the correct batch, then the right block. We don't need to offset across seq_len dimension,
    # since we're running a sequential scan; we will offset by stride_*_len while looping over timesteps.
    x_ptr += pid_batch * stride_x_batch + pid_block * stride_x_block  # x: (batch, seq_len, num_blocks, m=block_dim)
    a_ptr += (
        pid_batch * stride_a_batch + pid_block * stride_a_block
    )  # a: (batch, seq_len, num_blocks, m=block_dim, n=block_dim)
    out_ptr += (
        pid_batch * stride_out_batch + pid_block * stride_out_block
    )  # out: (batch, seq_len, num_blocks, m=block_dim)

    # Create 1D ptrs for x and out, each of size: BLOCK_SIZE_M.
    offsets_dim_m = pid_dim_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    x_ptrs = x_ptr + offsets_dim_m * stride_x_dim_m
    out_ptrs = out_ptr + offsets_dim_m * stride_out_dim_m
    # Create 2D ptrs for a, of size (BLOCK_SIZE_M, BLOCK_SIZE_N). We move row-by-row, offset_dim_m should move ptrs to
    # the first BLOCK_SIZE_N in the row; we will offset iteratively to move across the row, one BLOCK_SIZE_N at a time.
    offsets_dim_n = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + offsets_dim_m[:, None] * stride_a_dim_m + offsets_dim_n[None, :] * stride_a_dim_n

    for t in range(seq_len):
        # Move through one BLOCK_SIZE_N at a time and accumulate the result (matrix-vector multiplication) in acc; see
        # https://github.com/openai/triton/issues/375#issuecomment-1441180533.
        acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
        for n in range(tl.cdiv(block_dim, BLOCK_SIZE_N)):
            a_t = tl.load(a_ptrs + n * BLOCK_SIZE_N * stride_a_dim_n).to(tl.float32)
            prev_ht = (
                tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
                if t == 0
                else tl.load(out_ptrs - stride_out_len + n * BLOCK_SIZE_N * stride_out_dim_m).to(tl.float32)
            )
            acc += tl.sum(a_t * prev_ht[:, None], axis=1, keep_dims=False)  # accumulate sum along axis-n

        x_t = tl.load(x_ptrs).to(tl.float32)
        acc += x_t
        tl.store(out_ptrs, acc.to(out_ptrs.dtype.element_ty))

        # Advance all ptrs to point to the next element in the sequence.
        x_ptrs += stride_x_len
        a_ptrs += stride_a_len
        out_ptrs += stride_out_len


@triton.jit
def _sequential_scan_block_diag_a_bwd_kernel():
    """Sequential scan (with block-diagonal A) backward kernel, process BLOCK_SIZE_DIM elements."""
    pass


class SequentialScanBlockDiagA(torch.autograd.Function):
    """Sequential scan (with block-diagonal A), parallelized across "dim" dimension."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx: Any, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        num_blocks, block_dim = (
            a.shape[-3],
            a.shape[-1],
        )  # a: (batch, seq_len, num_blocks, block_dim, block_dim)

        BLOCK_SIZE = 512
        num_warps = min(max(BLOCK_SIZE // 32, 1), 16)
        print(f"x_elements_per_block={BLOCK_SIZE}, threads_per_block={num_warps * 32}")

        assert dim % BLOCK_SIZE == 0
        assert dim == num_blocks * block_dim, f"{(num_blocks * block_dim)=} mismatches with {dim=}"
        assert is_power_of_2(BLOCK_SIZE), f"{BLOCK_SIZE=} must be a power of two"

        x = rearrange(x, "... (n d) -> ... n d", n=num_blocks)  # (batch, seq_len, num_blocks, block_dim)
        out = torch.empty_like(x)  # (batch, seq_len, num_blocks, block_dim)

        # a_t@h_t + x_t: (BLOCK_SIZE_M, BLOCK_SIZE_N)@BLOCK_SIZE_N + BLOCK_SIZE_M.
        grid = lambda META: (
            batch,
            num_blocks,
            triton.cdiv(dim, META["BLOCK_SIZE_M"]),
        )
        _sequential_scan_block_diag_a_fwd_kernel[grid](
            x_ptr=x,
            a_ptr=a,
            out_ptr=out,
            stride_x_batch=x.stride(0),
            stride_x_len=x.stride(1),
            stride_x_block=x.stride(2),
            stride_x_dim_m=x.stride(3),
            stride_a_batch=a.stride(0),
            stride_a_len=a.stride(1),
            stride_a_block=a.stride(2),
            stride_a_dim_m=a.stride(3),
            stride_a_dim_n=a.stride(4),
            stride_out_batch=out.stride(0),
            stride_out_len=out.stride(1),
            stride_out_block=out.stride(2),
            stride_out_dim_m=out.stride(3),
            block_dim=block_dim,
            seq_len=seq_len,
            BLOCK_SIZE_M=BLOCK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE,
            num_warps=num_warps,
        )
        out = rearrange(out, "... n d -> ... (n d)", n=num_blocks)
        return out[:, -1]

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass


if __name__ == "__main__":
    from linear_rnn.reference.scan import scan_block_diag_a_ref

    _batch, _seq_len, _dim = 64, 2048, 256 * 50
    _num_blocks = 50

    test_x = torch.randn(_batch, _seq_len, _dim, dtype=torch.float32).cuda()
    test_a = torch.randn(
        _batch, _seq_len, _num_blocks, _dim // _num_blocks, _dim // _num_blocks, dtype=torch.float32
    ).cuda()

    test_ref_out = scan_block_diag_a_ref(test_x, test_a)
    test_out = SequentialScanBlockDiagA.apply(test_x, test_a)
    print(f"max_abs_err={torch.max(torch.abs(test_ref_out - test_out))}")
    assert torch.allclose(test_ref_out, test_out, atol=0.125, rtol=0)

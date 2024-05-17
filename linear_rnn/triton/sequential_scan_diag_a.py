from typing import Any

import torch
import triton
import triton.language as tl

from linear_rnn.utils.utils import is_power_of_2


@triton.jit
def _sequential_scan_diag_a_bwd_kernel():
    """Sequential scan (with diagonal A) backward kernel, processes BLOCK_SIZE elements."""
    pass


@triton.jit
def _sequential_scan_diag_a_fwd_kernel(
    x_ptr,
    a_ptr,
    out_ptr,
    stride_x_batch,
    stride_x_len,
    stride_x_dim,
    stride_a_batch,
    stride_a_len,
    stride_a_dim,
    stride_out_batch,
    stride_out_len,
    stride_out_dim,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sequential scan (with diagonal A) forward kernel, processes BLOCK_SIZE elements."""
    # Grid: (batch, dim/BLOCK_SIZE) blocks.
    pid_batch = tl.program_id(0)
    pid_dim = tl.program_id(1)

    # Move all ptrs to the right batch being operated on.
    x_ptr += pid_batch * stride_x_batch
    a_ptr += pid_batch * stride_a_batch
    out_ptr += pid_batch * stride_out_batch

    # Within the batch, move to the right block start. Note: all the ptrs currently
    # point to the first element in the sequence.
    offsets = pid_dim * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + offsets * stride_x_dim
    a_ptrs = a_ptr + offsets * stride_a_dim
    out_ptrs = out_ptr + offsets * stride_out_dim

    h_t = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for t in range(seq_len):
        # Load x and a from HBM to SRAM.
        x_t = tl.load(x_ptrs).to(tl.float32)
        a_t = tl.load(a_ptrs).to(tl.float32)

        h_t = a_t * h_t + x_t
        tl.store(out_ptrs, h_t.to(out_ptrs.dtype.element_ty))  # SRAM to HBM

        # Advance all ptrs to the next element in the sequence.
        if t < seq_len - 1:
            x_ptrs += stride_x_len
            a_ptrs += stride_a_len
            out_ptrs += stride_out_len


class SequentialScanDiagA(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx: Any, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Sequential scan (with diagonal A), parallelized across "dim" dimension."""
        batch, seq_len, dim = x.shape

        # Set num_warps based on BLOCK_SIZE: (BLOCK_SIZE // 32) indicates the number of
        # warps to process the entire block in one go.
        BLOCK_SIZE = 512
        num_warps = min(max(BLOCK_SIZE // 32, 1), 16)

        assert dim % BLOCK_SIZE == 0, f"{dim=} must be a multiple of {BLOCK_SIZE=}"
        assert x.shape == a.shape
        assert is_power_of_2(BLOCK_SIZE), f"{BLOCK_SIZE=} must be a power of two"

        # Do not initialize h_0 here, directly create it on SRAM.
        out = torch.empty_like(x)

        # Grid: (batch, dim/BLOCK_SIZE) blocks, with BLOCK_SIZE elements per block.
        grid = lambda META: (batch, triton.cdiv(dim, META["BLOCK_SIZE"]))
        _sequential_scan_diag_a_fwd_kernel[grid](
            x_ptr=x,
            a_ptr=a,
            out_ptr=out,
            stride_x_batch=x.stride(0),
            stride_x_len=x.stride(1),
            stride_x_dim=x.stride(2),
            stride_a_batch=a.stride(0),
            stride_a_len=a.stride(1),
            stride_a_dim=a.stride(2),
            stride_out_batch=out.stride(0),
            stride_out_len=out.stride(1),
            stride_out_dim=out.stride(2),
            seq_len=seq_len,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass


if __name__ == "__main__":
    from linear_rnn.reference.scan import scan_diag_a_ref

    _batch, _seq_len, _dim = 32, 1024, 256 * 20

    test_x = torch.randn(_batch, _seq_len, _dim, dtype=torch.float32).cuda()
    test_a = torch.randn(_batch, _seq_len, _dim, dtype=torch.float32).cuda()

    test_ref_out = scan_diag_a_ref(test_x, test_a)
    test_out = SequentialScanDiagA.apply(test_x, test_a)
    assert torch.allclose(test_ref_out, test_out, atol=0.125, rtol=0)

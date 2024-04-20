import math
import torch
import triton
import triton.language as tl

from linear_rnn.triton.naive_associative_rnn_scan import _associative_scan_op
from linear_rnn.triton.sequential_rnn_scan import rnn_scan_ref


@triton.jit
def _associative_rnn_scan_fwd_kernel(
    x_ptr,
    a_ptr,
    cum_a_ptr,
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
    stride_cum_a_batch,
    stride_cum_a_len,
    stride_cum_a_dim,
    BLOCK_SIZE_LEN: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    # Grid: (batch, seq_len/BLOCK_SIZE_LEN, dim/BLOCK_SIZE_DIM)
    pid_batch = tl.program_id(0)
    pid_len = tl.program_id(1)
    pid_dim = tl.program_id(2)

    x_ptr += pid_batch * stride_x_batch
    a_ptr += pid_batch * stride_a_batch
    if cum_a_ptr is not None:
        cum_a_ptr += pid_batch * stride_cum_a_batch
    out_ptr += pid_batch * stride_out_batch

    offsets_dim = pid_dim * BLOCK_SIZE_DIM + tl.arange(0, BLOCK_SIZE_DIM)
    offsets_len = pid_len * BLOCK_SIZE_LEN + tl.arange(0, BLOCK_SIZE_LEN)
    x_ptrs = x_ptr + offsets_dim[None, :] * stride_x_dim + offsets_len[:, None] * stride_x_len
    a_ptrs = a_ptr + offsets_dim[None, :] * stride_a_dim + offsets_len[:, None] * stride_a_len

    out_ptrs = out_ptr + offsets_dim[None, :] * stride_out_dim + offsets_len[:, None] * stride_out_len
    if cum_a_ptr is not None:
        cum_a_ptrs = cum_a_ptr + offsets_dim[None, :] * stride_cum_a_dim + offsets_len[:, None] * stride_cum_a_len

    x = tl.load(x_ptrs).to(tl.float32)
    a = tl.load(a_ptrs).to(tl.float32)
    cum_a, all_hiddens = tl.associative_scan(input=(a, x), axis=0, combine_fn=_associative_scan_op)

    mask = (offsets_len == ((pid_len + 1) * BLOCK_SIZE_LEN - 1))[:, None]
    if cum_a_ptr is not None:
        tl.store(cum_a_ptrs, cum_a.to(cum_a_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptrs, all_hiddens.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _associative_rnn_scan_bwd_kernel():
    pass


def associative_rnn_scan(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Parallelize across seq_len and dim: https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf#page=6.39."""
    BLOCK_SIZE_LEN, BLOCK_SIZE_DIM, num_warps = 32, 256, 16  # TODO: change later to autotune

    batch, seq_len, dim = x.shape
    assert seq_len % BLOCK_SIZE_LEN == 0, f"{seq_len=} is not a multiple of {BLOCK_SIZE_LEN=}"
    assert dim % BLOCK_SIZE_DIM == 0, f"{dim=} is not a multiple of {BLOCK_SIZE_DIM=}"
    assert x.shape == a.shape

    out = torch.zeros_like(x)
    cum_a = torch.zeros_like(x)

    num_iters = int(triton.cdiv(math.log2(out.shape[1]), math.log2(BLOCK_SIZE_LEN)))
    for iter_id in range(num_iters):
        grid = lambda META: (
            batch,
            max(1, triton.cdiv(out.shape[1], META["BLOCK_SIZE_LEN"])),
            triton.cdiv(dim, META["BLOCK_SIZE_DIM"]),
        )
        _associative_rnn_scan_fwd_kernel[grid](
            x_ptr=x,
            a_ptr=a,
            cum_a_ptr=cum_a if iter_id != num_iters - 1 else None,
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
            stride_cum_a_batch=cum_a.stride(0) if iter_id != num_iters - 1 else None,
            stride_cum_a_len=cum_a.stride(1) if iter_id != num_iters - 1 else None,
            stride_cum_a_dim=cum_a.stride(2) if iter_id != num_iters - 1 else None,
            BLOCK_SIZE_LEN=BLOCK_SIZE_LEN if iter_id != num_iters - 1 else out.shape[1],
            BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
            num_warps=num_warps,
        )
        if iter_id != num_iters - 1:
            out = out[:, BLOCK_SIZE_LEN - 1 :: BLOCK_SIZE_LEN]
            cum_a = cum_a[:, BLOCK_SIZE_LEN - 1 :: BLOCK_SIZE_LEN]
            x, a = out, cum_a
    return out[:, -1]


if __name__ == "__main__":
    _batch, _seq_len, _dim = 100, 1024, 256 * 100

    test_x = torch.randn(_batch, _seq_len, _dim).cuda()
    test_a = torch.randn(_batch, _seq_len, _dim).cuda()

    assert torch.allclose(associative_rnn_scan(test_x, test_a), rnn_scan_ref(test_x, test_a), atol=0.125, rtol=0)

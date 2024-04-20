import torch
import triton
import triton.language as tl

from linear_rnn.triton.sequential_rnn_scan import rnn_scan_ref


@triton.jit
def _associative_scan_op(a_l, x_l, a_r, x_r):
    return a_r * a_l, a_r * x_l + x_r


@triton.jit
def _naive_associative_rnn_scan_fwd_kernel(
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
    # Grid: (batch, dim/BLOCK_SIZE)
    pid_batch = tl.program_id(0)
    pid_dim = tl.program_id(1)

    x_ptr += pid_batch * stride_x_batch
    a_ptr += pid_batch * stride_a_batch
    out_ptr += pid_batch * stride_out_batch

    offsets_dim = pid_dim * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_len = tl.arange(0, seq_len)
    x_ptrs = x_ptr + offsets_dim[None, :] * stride_x_dim + offsets_len[:, None] * stride_x_len
    a_ptrs = a_ptr + offsets_dim[None, :] * stride_a_dim + offsets_len[:, None] * stride_a_len

    out_ptrs = out_ptr + offsets_dim[None, :] * stride_out_dim + offsets_len[:, None] * stride_out_len

    x = tl.load(x_ptrs).to(tl.float32)
    a = tl.load(a_ptrs).to(tl.float32)
    _, all_hiddens = tl.associative_scan(input=(a, x), axis=0, combine_fn=_associative_scan_op)
    tl.store(out_ptrs, all_hiddens.to(out_ptr.dtype.element_ty), mask=(offsets_len == seq_len - 1)[:, None])


@triton.jit
def _naive_associative_rnn_scan_bwd_kernel():
    pass


def naive_associative_rnn_scan(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Process entire sequence at once; doesn't work for long sequences."""
    BLOCK_SIZE, num_warps = 256, 16  # TODO: change later to autotune

    batch, seq_len, dim = x.shape
    assert dim % BLOCK_SIZE == 0, f"{dim=} is not a multiple of {BLOCK_SIZE=}"
    assert x.shape == a.shape

    out = torch.empty_like(x)

    grid = lambda META: (batch, triton.cdiv(dim, META["BLOCK_SIZE"]))
    _naive_associative_rnn_scan_fwd_kernel[grid](
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
    return out[:, -1]


if __name__ == "__main__":
    _batch, _seq_len, _dim = 100, 256, 256 * 20

    test_x = torch.randn(_batch, _seq_len, _dim).cuda()
    test_a = torch.randn(_batch, _seq_len, _dim).cuda()

    assert torch.allclose(naive_associative_rnn_scan(test_x, test_a), rnn_scan_ref(test_x, test_a), atol=0.125, rtol=0)

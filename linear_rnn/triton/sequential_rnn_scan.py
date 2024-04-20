import math
import torch
import triton
import triton.language as tl


@triton.jit
def _sequential_rnn_scan_fwd_kernel(
    x_ptr,
    a_ptr,
    h0_ptr,
    out_ptr,
    stride_x_batch,
    stride_x_len,
    stride_x_dim,
    stride_a_batch,
    stride_a_len,
    stride_a_dim,
    stride_h0_batch,
    stride_h0_dim,
    stride_out_batch,
    stride_out_dim,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: (batch, dim/BLOCK_SIZE)
    pid_batch = tl.program_id(0)
    pid_dim = tl.program_id(1)

    x_ptr += pid_batch * stride_x_batch
    a_ptr += pid_batch * stride_a_batch
    ht_ptr = h0_ptr + pid_batch * stride_h0_batch
    out_ptr += pid_batch * stride_out_batch

    offsets = pid_dim * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + offsets * stride_x_dim
    a_ptrs = a_ptr + offsets * stride_a_dim

    ht_ptrs = ht_ptr + offsets * stride_h0_dim
    out_ptrs = out_ptr + offsets * stride_out_dim

    h_t = tl.load(ht_ptrs).to(tl.float32)  # load to SRAM
    for t in range(seq_len):
        x_t = tl.load(x_ptrs).to(tl.float32)
        a_t = tl.load(a_ptrs).to(tl.float32)

        h_t = a_t * h_t + x_t

        if t < seq_len - 1:
            x_ptrs += stride_x_len
            a_ptrs += stride_a_len

    tl.store(out_ptrs, h_t.to(out_ptr.dtype.element_ty))  # write back to HBM/DRAM


@triton.jit
def _sequential_rnn_scan_bwd_kernel():
    pass


def sequential_rnn_scan(x: torch.Tensor, a: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
    BLOCK_SIZE, num_warps = 256, 16  # TODO: change later to autotune

    batch, seq_len, dim = x.shape
    assert math.floor(math.log2(batch)) == math.ceil(math.log2(batch)), f"{batch=} must be a power of 2"
    assert dim % BLOCK_SIZE == 0, f"{dim=} is not a multiple of {BLOCK_SIZE=}"
    assert x.shape == a.shape
    assert h0 is None or h0.shape == (batch, dim)

    if h0 is None:
        h0 = torch.zeros_like(x[:, 0], device=x.device, dtype=x.dtype)
    out = torch.empty_like(h0)

    grid = lambda META: (batch, triton.cdiv(dim, META["BLOCK_SIZE"]))
    _sequential_rnn_scan_fwd_kernel[grid](
        x_ptr=x,
        a_ptr=a,
        h0_ptr=h0,
        out_ptr=out,
        stride_x_batch=x.stride(0),
        stride_x_len=x.stride(1),
        stride_x_dim=x.stride(2),
        stride_a_batch=a.stride(0),
        stride_a_len=a.stride(1),
        stride_a_dim=a.stride(2),
        stride_h0_batch=h0.stride(0),
        stride_h0_dim=h0.stride(1),
        stride_out_batch=out.stride(0),
        stride_out_dim=out.stride(1),
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


def rnn_scan_ref(x: torch.Tensor, a: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
    h_t = torch.zeros_like(x[:, 0], device=x.device, dtype=x.dtype) if h0 is None else h0
    for t in range(x.shape[1]):
        h_t = a[:, t] * h_t + x[:, t]
    return h_t


if __name__ == "__main__":
    _batch, _seq_len, _dim = 100, 1024, 256 * 20

    test_x = torch.randn(_batch, _seq_len, _dim, dtype=torch.float32).cuda()
    test_a = torch.randn(_batch, _seq_len, _dim, dtype=torch.float32).cuda()

    assert torch.allclose(sequential_rnn_scan(test_x, test_a), rnn_scan_ref(test_x, test_a), atol=0.125, rtol=0)

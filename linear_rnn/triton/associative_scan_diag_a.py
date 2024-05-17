from typing import Any

import math
import torch
import triton
import triton.language as tl

from linear_rnn.utils.utils import is_power_of_2


@triton.jit
def _recurrence_binary_op(a_l, x_l, a_r, x_r):
    return a_r * a_l, a_r * x_l + x_r


@triton.jit
def _associative_scan_diag_a_fwd_kernel(
    x_ptr,
    a_ptr,
    acc_a_ptr,
    out_ptr,
    stride_x_batch,
    stride_x_len,
    stride_x_dim,
    stride_a_batch,
    stride_a_len,
    stride_a_dim,
    stride_acc_a_batch,
    stride_acc_a_len,
    stride_acc_a_dim,
    stride_out_batch,
    stride_out_len,
    stride_out_dim,
    BLOCK_SIZE_LEN: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    """
    Associative scan (with diagonal A) forward kernel, process BLOCK_SIZE_LEN * BLOCK_SIZE_DIM elements.

    NOTE: This kernel only focuses on extracting the final hidden state. To that end, intermediate (unnecessary)
    hidden states are discarded. If you need all hidden states, refer to the algorithm depicted in Figure 5 of
    https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf#page=6.39.
    """
    # Grid: (batch, seq_len/BLOCK_SIZE_LEN, dim/BLOCK_SIZE_DIM)
    pid_batch = tl.program_id(0)
    pid_len = tl.program_id(1)
    pid_dim = tl.program_id(2)

    # Move all ptrs to the right batch.
    x_ptr += pid_batch * stride_x_batch
    a_ptr += pid_batch * stride_a_batch
    if acc_a_ptr is not None:
        acc_a_ptr += pid_batch * stride_acc_a_batch
    out_ptr += pid_batch * stride_out_batch

    # Move all ptrs to point to the right position in seq and the right dim. For instance, if we want to access the
    # second "chunk" of elements and the first dim block in them, we move BLOCK_SIZE_LEN elements to get to the second
    # chunk (no need to move further to get to the right dim).
    offsets_len = pid_len * BLOCK_SIZE_LEN + tl.arange(0, BLOCK_SIZE_LEN)
    offsets_dim = pid_dim * BLOCK_SIZE_DIM + tl.arange(0, BLOCK_SIZE_DIM)
    # 2D ptr blocks of size (BLOCK_DIM_LEN, BLOCK_DIM_DIM) with dim-1 as the "length" dim, dim-2 as the "dim" dimension.
    x_ptrs = x_ptr + offsets_len[:, None] * stride_x_len + offsets_dim[None, :] * stride_x_dim
    a_ptrs = a_ptr + offsets_len[:, None] * stride_a_len + offsets_dim[None, :] * stride_a_dim
    if acc_a_ptr is not None:
        acc_a_ptrs = acc_a_ptr + offsets_len[:, None] * stride_acc_a_len + offsets_dim[None, :] * stride_acc_a_dim
    out_ptrs = out_ptr + offsets_len[:, None] * stride_out_len + offsets_dim[None, :] * stride_out_dim

    x = tl.load(x_ptrs).to(tl.float32)
    a = tl.load(a_ptrs).to(tl.float32)
    # Run associative scan along the length dim.
    acc_a, all_h = tl.associative_scan(input=(a, x), axis=0, combine_fn=_recurrence_binary_op)

    # Store only the last elements of each scan.
    mask = (offsets_len == (pid_len + 1) * BLOCK_SIZE_LEN - 1)[:, None]
    if acc_a_ptr is not None:
        tl.store(acc_a_ptrs, acc_a.to(acc_a_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptrs, all_h.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _associative_scan_diag_a_bwd_kernel():
    """Associative scan (with diagonal A) backward kernel, process BLOCK_SIZE_LEN * BLOCK_SIZE_DIM elements."""
    pass


class AssociativeScanDiagA(torch.autograd.Function):
    """
    Associative scan (with diagonal A), parallelize across "seq len" and "dim" dimensions.
    Ref: https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf#page=6.39 (see Figure 5).
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx: Any, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape

        BLOCK_SIZE_LEN, BLOCK_SIZE_DIM = 64, 128
        num_warps = min(max((BLOCK_SIZE_LEN * BLOCK_SIZE_DIM) // 32, 1), 32)
        print(f"elements_per_block={BLOCK_SIZE_LEN * BLOCK_SIZE_DIM}, threads_per_block={num_warps * 32}")

        assert seq_len % BLOCK_SIZE_LEN == 0, f"{seq_len=} must be a multiple of {BLOCK_SIZE_LEN=}"
        assert dim % BLOCK_SIZE_DIM == 0, f"{dim=} must be a multiple of {BLOCK_SIZE_DIM=}"
        assert x.shape == a.shape
        assert is_power_of_2(
            BLOCK_SIZE_LEN * BLOCK_SIZE_DIM
        ), f"number of elements={BLOCK_SIZE_LEN * BLOCK_SIZE_DIM}, not power of two"

        acc_a = torch.empty_like(x)
        out = torch.empty_like(x)

        num_iters = math.ceil(math.log2(out.shape[1]) / math.log2(BLOCK_SIZE_LEN))
        for iter_idx in range(num_iters):
            # Grid: (batch, seq_len/BLOCK_SIZE_LEN, dim/BLOCK_SIZE_DIM)
            grid = lambda META: (batch, max(1, triton.cdiv(seq_len, BLOCK_SIZE_LEN)), triton.cdiv(dim, BLOCK_SIZE_DIM))
            _associative_scan_diag_a_fwd_kernel[grid](
                x_ptr=x,
                a_ptr=a,
                acc_a_ptr=acc_a if iter_idx != num_iters - 1 else None,
                out_ptr=out,
                stride_x_batch=x.stride(0),
                stride_x_len=x.stride(1),
                stride_x_dim=x.stride(2),
                stride_a_batch=a.stride(0),
                stride_a_len=a.stride(1),
                stride_a_dim=a.stride(2),
                stride_acc_a_batch=acc_a.stride(0) if iter_idx != num_iters - 1 else None,
                stride_acc_a_len=acc_a.stride(1) if iter_idx != num_iters - 1 else None,
                stride_acc_a_dim=acc_a.stride(2) if iter_idx != num_iters - 1 else None,
                stride_out_batch=out.stride(0),
                stride_out_len=out.stride(1),
                stride_out_dim=out.stride(2),
                BLOCK_SIZE_LEN=BLOCK_SIZE_LEN if iter_idx != num_iters - 1 else out.shape[1],
                BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
                num_warps=num_warps,
            )
            if iter_idx != num_iters - 1:
                acc_a = acc_a[:, BLOCK_SIZE_LEN - 1 :: BLOCK_SIZE_LEN]
                out = out[:, BLOCK_SIZE_LEN - 1 :: BLOCK_SIZE_LEN]
                a, x = acc_a, out

        # Only return the last hidden state (note: the hidden states at other positions in out would be incorrect).
        return out[:, -1]

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass


if __name__ == "__main__":
    from linear_rnn.reference.scan import scan_diag_a_ref

    _batch, _seq_len, _dim = 64, 1024, 256 * 100

    test_x = torch.randn(_batch, _seq_len, _dim, dtype=torch.float32).cuda()
    test_a = torch.randn(_batch, _seq_len, _dim, dtype=torch.float32).cuda()

    test_ref_out = scan_diag_a_ref(test_x, test_a)
    test_out = AssociativeScanDiagA.apply(test_x, test_a)
    assert torch.allclose(test_ref_out, test_out, atol=0.125, rtol=0)

import torch

import triton
import triton.language as tl


@triton.jit
def _softmax_fwd_kernel(output_ptr, output_row_stride, input_ptr, input_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)  # parallelize across rows

    input_row_start_ptr = input_ptr + (row_idx * input_row_stride)  # go to the location of the row
    col_offsets = tl.arange(0, BLOCK_SIZE)  # fitting one row per block

    input_ptrs = input_row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=float("-inf"))  # move row from HBM to SRAM

    normalized_row = row - tl.max(row, axis=0)
    numerator = tl.exp(normalized_row)  # note: exponentiation is faster but approximate in Triton
    softmax_out = numerator / tl.sum(numerator, axis=0)

    output_row_start_ptr = output_ptr + (row_idx * input_row_stride)
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Triton kernel for row-wise parallelization; softmax is applied row-wise in parallel."""

    assert len(x.shape) == 2, f"softmax only supports 2D inputs (for now)"
    n_rows, n_cols = x.shape

    BLOCK_SIZE = tl.next_power_of_2(n_cols)  # num elements each program processes; must be a power of 2

    num_warps = 4  # thread density per row (* 32 threads)
    if BLOCK_SIZE > 2047:
        num_warps = 8
    elif BLOCK_SIZE > 4095:
        num_warps = 16

    y = torch.empty_like(x)  # allocate memory for output buffer

    grid = (n_rows,)  # one kernel instance per row of the input
    _softmax_fwd_kernel[grid](y,    # output buffer
        y.stride(0),                # num elements to get to the next row in input
        x,                          # input buffer
        x.stride(0),                # num elements to get to the next row in output
        n_cols,                     # num cols to mask out the "pad" entries
        BLOCK_SIZE=BLOCK_SIZE,      # block size, must be a power of 2
        num_warps=num_warps,        # thread density (* 32 threads)
    )
    return y


if __name__ == "__main__":
    test_tensor = torch.randn(10000, 768, device="cuda")
    assert torch.allclose(torch.nn.functional.softmax(test_tensor, dim=1), softmax(test_tensor))

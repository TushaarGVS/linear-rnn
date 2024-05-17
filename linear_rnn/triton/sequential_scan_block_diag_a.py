from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _sequential_scan_block_diag_a_fwd_kernel(
    x_ptr,
    a_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sequential scan (with block-diagonal A) forward kernel, process BLOCK_SIZE_DIM elements."""
    pass


@triton.jit
def _sequential_scan_block_diag_a_bwd_kernel():
    """Sequential scan (with block-diagonal A) backward kernel, process BLOCK_SIZE_DIM elements."""
    pass


class SequentialScanBlockDiagA(torch.autograd.Function):
    """Sequential scan (with block-diagonal A), parallelized across "num_blocks" and "dim" dimensions."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx: Any, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass

import torch
from einops import rearrange


def scan_diag_a_ref(x: torch.Tensor, a: torch.Tensor):
    """(Sequential) scan reference with diagonal A."""
    # x, a: (batch, seq_len, dim)
    # h_t: (batch, dim)
    h_t = torch.zeros_like(x[:, 0])
    for t in range(x.shape[1]):
        h_t = a[:, t] * h_t + x[:, t]
    return h_t


def scan_block_diag_a_ref(x: torch.Tensor, a: torch.Tensor):
    """(Sequential) scan reference with block-diagonal A."""
    # x: (batch, seq_len, dim)
    # a: (batch, seq_len, num_blocks, block_dim, block_dim)
    num_blocks, block_dim = a.shape[-3], a.shape[-1]
    assert (
        num_blocks * block_dim == x.shape[-1]
    ), f"({a.shape[-3]=} * {a.shape[-1]=})={num_blocks * block_dim} in a must match {x.shape[-1]=}"
    x = rearrange(x, "... (n d) -> ... n d", n=num_blocks)  # (batch, seq_len, num_blocks, block_dim)

    h_t = torch.zeros_like(x[:, 0])  # h_t: (batch, num_blocks, block_dim)
    for t in range(x.shape[1]):
        h_t = torch.einsum("... d k, ... k -> ... d", a[:, t], h_t) + x[:, t]
    h_t = rearrange(h_t, "... n d -> ... (n d)", n=num_blocks)

    return h_t

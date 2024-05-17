import torch


def scan_diag_a_ref(x: torch.Tensor, a: torch.Tensor):
    """(Sequential) scan reference."""
    # x, a: (batch, seq_len, dim).
    # h_t: (batch, dim)
    h_t = torch.zeros_like(x[:, 0])
    for t in range(1, x.shape[1]):
        h_t = a[:, t] * h_t + x[:, t]
    return h_t

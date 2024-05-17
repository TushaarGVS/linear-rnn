import torch


def scan_diag_a_ref(x: torch.Tensor, a: torch.Tensor):
    """(Sequential) scan reference."""
    # x, a, h: (batch, seq_len, dim).
    h = torch.zeros_like(x)

    h[:, 0] = x[:, 0]  # h_0 = x_0
    for t in range(1, x.shape[1]):
        h[:, t] = a[:, t] * h[:, t - 1] + x[:, t]
    return h

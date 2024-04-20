from typing import Tuple, Callable, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from linear_rnn.triton.associative_rnn_scan import associative_rnn_scan
from linear_rnn.triton.sequential_rnn_scan import sequential_rnn_scan, rnn_scan_ref


def _randn_x_a(
    batch: int = 4, seq_len: int = 1024, dim: int = 5120, dtype: torch.dtype = torch.float16, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(batch, seq_len, dim, dtype=dtype, device="cuda")
    a = torch.randn(batch, seq_len, dim, dtype=dtype, device="cuda")
    return x, a


def _get_max_abs_err(fn_out: torch.Tensor, ref_out: torch.Tensor) -> float:
    return (fn_out - ref_out).abs().max().item()


@torch.inference_mode()
def benchmark(
    benchmark_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], seq_len: int, dtype: torch.dtype, seed: int
) -> Dict[str, Any]:
    x, a = _randn_x_a(batch=4, seq_len=seq_len, dim=5120, dtype=dtype, seed=seed)
    fn_out = benchmark_fn(x, a)
    ref_out = rnn_scan_ref(x, a)
    return {"seq_len": seq_len, "dtype": dtype, "max_abs_err": _get_max_abs_err(fn_out=fn_out, ref_out=ref_out)}


if __name__ == "__main__":
    seeds = [3407, 4, 42, 57]  # https://arxiv.org/abs/2109.08203
    seq_lengths = [2**i for i in range(5, 18, 1)]
    benchmark_fns = [sequential_rnn_scan, associative_rnn_scan]
    dtypes = [torch.float32, torch.bfloat16, torch.float16]

    fn_marker_styles = ["+", "3"]
    dtype_colors = ["blue", "lime", "green"]

    for dtype_idx, _dtype in enumerate(dtypes):
        for fn_idx, _benchmark_fn in enumerate(benchmark_fns):
            seq_len_to_err_mapping = {}
            for _seed in seeds:
                for _seq_len in seq_lengths:
                    err = benchmark(benchmark_fn=_benchmark_fn, seq_len=_seq_len, dtype=_dtype, seed=_seed)[
                        "max_abs_err"
                    ]
                    seq_len_to_err_mapping[_seq_len] = seq_len_to_err_mapping.get(_seq_len, 0) + err
            plt.scatter(
                x=seq_lengths,
                y=np.array(list(seq_len_to_err_mapping.values())) / len(seeds),
                label=f"{_benchmark_fn.__name__} ({_dtype})",
                color=dtype_colors[dtype_idx],
                marker=fn_marker_styles[fn_idx],
            )

    plt.xlabel("seq_len")
    plt.ylabel("max_abs_err")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    plt.savefig("linear_rnn/benchmarking/results/rnn_scan/rnn_scan_precision.png")

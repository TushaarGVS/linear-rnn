import torch
import triton

from linear_rnn.triton.associative_rnn_scan import associative_rnn_scan
from linear_rnn.triton.sequential_rnn_scan import sequential_rnn_scan, rnn_scan_ref


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[int(2**i) for i in range(5, 18, 1)],
            x_log=True,
            y_log=True,
            line_arg="benchmark_fn",
            line_vals=["rnn_scan_ref", "sequential_rnn_scan", "associative_rnn_scan"],
            line_names=["sequential_scan_torch", "sequential_scan_triton", "associative_scan_triton"],
            styles=[("blue", "-"), ("green", "--"), ("green", "-")],
            ylabel="ms",
            plot_name="rnn_scan_performance",
            args={"batch": 4, "dim": 5120},
        )
    ]
)
def benchmark(batch: int, seq_len: int, dim: int, benchmark_fn: str):
    x = torch.randn(batch, seq_len, dim, device="cuda", dtype=torch.float16)
    a = torch.randn(batch, seq_len, dim, device="cuda", dtype=torch.float16)

    quantiles = [0.5, 0.2, 0.8]
    if benchmark_fn == "rnn_scan_ref":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rnn_scan_ref(x, a), quantiles=quantiles)
    elif benchmark_fn == "sequential_rnn_scan":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sequential_rnn_scan(x, a), quantiles=quantiles)
    elif benchmark_fn == "associative_rnn_scan":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: associative_rnn_scan(x, a), quantiles=quantiles)
    return ms, min_ms, max_ms


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True, save_path="linear_rnn/benchmarking/results/rnn_scan/")

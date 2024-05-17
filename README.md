# Linear RNN (Triton)

## Installation

The repository uses [Poetry](https://python-poetry.org/docs/) to manage dependencies. To install dependencies for the
entire package, run:

```shell
cd $HOME; git clone https://github.com/TushaarGVS/linear-rnn.git
cd $HOME/linear-rnn

poetry install
```

To run the package in an editable mode, run:

```shell
pip install -e .
```

__Note.__ The pacakge uses `triton-3.0.0` which needs to be installed from source (and is also not compatible with
`torch=2.2.2`).

---

## Profiling (and running)

Sample tests are included within the source code files (please follow those to view how to use the provided modules).
To profile, simply run:

```shell
ncu_path python3 linear_rnn/triton/sequential_scan_diag_a.py
```

On Linux, the `ncu_path` is defaulted to: `/usr/local/cuda-<version>/nsight-compute-<version>/ncu`; for other platforms,
please refer to the [installation doc](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#quickstart).

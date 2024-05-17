# Linear RNN (Triton)

## Installation

The repository uses [Poetry](https://python-poetry.org/docs/) to manage dependencies. To install dependencies for the
entire package, run:

```shell
conda create -n linear-rnn-env python=3.10
conda activate linear-rnn-env
# conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit

cd $HOME; git clone https://github.com/TushaarGVS/linear-rnn.git
cd $HOME/linear-rnn
```

To install and run the package in an editable mode, run:

```shell
pip install -e .
```

---

## Profiling (and running)

Sample tests are included within the source code files (please follow those to view how to use the provided modules).
To profile, simply run (all options [here](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#profile)):

```shell
ncu_path -f -o ~/profile_log.txt python3 linear_rnn/triton/sequential_scan_diag_a.py
```

On Linux, the `ncu_path` is defaulted to: `/usr/local/cuda-<version>/nsight-compute-<version>/ncu`; for other platforms,
please refer to the [installation doc](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#quickstart).

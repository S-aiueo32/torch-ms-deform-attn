# GPU benchmarks

[Back to README](../README.md) · [Runpod setup and cleanup](gpu-runner.md)

The **CUDA benchmark** workflow (`.github/workflows/cuda-benchmark.yml`) runs
manually on a temporary Runpod GPU. Configure the `RUNPOD_API_KEY` Actions secret
as described in the Runpod setup guide, then choose **Actions → CUDA benchmark →
Run workflow**. The defaults are one NVIDIA L4, a $0.50/hour compute-price limit,
and a 45-minute controller deadline. `operation=check` checks credentials and
pricing without creating a Pod.

```bash
gh workflow run cuda-benchmark.yml --ref main \
  --field operation=benchmark --field gpu='NVIDIA L4' \
  --field max_hourly_usd=0.50 --field timeout_minutes=45
```

The workflow builds and installs a CUDA wheel with Python 3.11 and PyTorch 2.5.1
(CUDA 12.4), runs the correctness suite, then executes
`benchmarks/benchmark_cuda.py` outside the checkout. It compares the eager CUDA
extension with the PyTorch `grid_sample` reference using float32 inputs for the
small, decoder, and batched cases. Each case measures forward and
forward-plus-backward execution, after checking outputs and gradients.

Each backend receives five warmup calls before `torch.utils.benchmark.Timer`
measures execution with CUDA synchronization. The reference uses static Python
shape tuples to avoid device-to-host metadata transfers. Timings include Python
dispatch and GPU execution; they are not isolated kernel timings. Compilation
and mixed precision are not measured.

Download the `cuda-benchmark-<run-id>-<attempt>` artifact (retained for seven days):

- `artifacts/benchmark-cuda.json`: median and IQR in milliseconds, speedup
  (`pytorch_ms / cuda_ms`), GPU and software metadata.
- `artifacts/nvidia-smi.txt`: GPU and driver information.
- `controller.log`, `artifacts/cuda-checks.log`, and built distributions.

A correctness or benchmark failure fails the job. The workflow shares the CUDA
correctness workflow's concurrency group, price checks, timeout, and Pod cleanup.
The optional recovery workflow also handles completed benchmark runs once the
updated workflows are on the default branch and `RUNPOD_CLEANUP_ENABLED=true`.
See the setup guide for billing and recovery limitations.

To benchmark an already installed CUDA wheel locally:

```bash
python benchmarks/benchmark_cuda.py --min-run-time 1.0 > benchmark-cuda.json
```

# Benchmarks

[Back to README](../README.md)

Compare the installed extension with the PyTorch reference on synthetic CPU or
CUDA workloads. Timings are median milliseconds; speedup is reference latency
divided by extension latency. These measurements do not predict whole-model
latency or peak memory usage.

## CPU benchmarks

### OpenMP measurement

Measured on macOS 26.6.1 arm64 with Python 3.11.16,
PyTorch 2.5.1, eager float32, and the optimized OpenMP extension. One-thread and
four-thread runs were sequential, with one-second minimum measurement windows.
Outputs and all three gradients were checked before timing.

The CPU model was not recorded. Source hashes, build settings, and IQRs are
available in the `parallel_optimized` section of
the [measurement report](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/9386e99/benchmarks/results/cpu-optimization.json).

| Threads | Case | Mode | C++ extension (ms) | PyTorch reference (ms) | Speedup |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | small | forward | 0.0589 | 0.3292 | 5.59× |
| 1 | small | forward + backward | 0.2165 | 0.8170 | 3.77× |
| 1 | decoder | forward | 1.0462 | 6.0300 | 5.76× |
| 1 | decoder | forward + backward | 4.3460 | 17.0029 | 3.91× |
| 1 | batched | forward | 4.4434 | 21.2837 | 4.79× |
| 1 | batched | forward + backward | 20.4952 | 48.0750 | 2.35× |
| 4 | small | forward | 0.0238 | 0.2952 | 12.40× |
| 4 | small | forward + backward | 0.0917 | 0.7786 | 8.49× |
| 4 | decoder | forward | 0.2879 | 2.8453 | 9.88× |
| 4 | decoder | forward + backward | 1.1886 | 7.4185 | 6.24× |
| 4 | batched | forward | 1.2405 | 8.1830 | 6.60× |
| 4 | batched | forward + backward | 5.4480 | 16.2660 | 2.99× |

### Run CPU benchmarks

After [installing the package](installation.md), run from the repository root:

```bash
python benchmarks/benchmark_cpu.py --threads 1 --min-run-time 1 --json
python benchmarks/benchmark_cpu.py --threads 4 --min-run-time 1 --json
```

The harness checks outputs and all three gradients before timing. Check the
[CPU parallel backend](installation.md#configure-cpu-parallelism) when comparing
thread counts: `--threads` does not parallelize a serial extension.

To compare eager and Inductor execution:

```bash
python benchmarks/benchmark_compile.py --threads 1
```

Compilation and warmup precede timing. The compile harness uses static shape
tuples for the reference, so its baseline differs from `benchmark_cpu.py`.
Compare runs with the same PyTorch version, build settings, shapes, and thread
count.

## GPU benchmarks

### L4 measurement

Measured at commit `d229c2c`: NVIDIA L4,
Python 3.11.10, PyTorch 2.5.1+cu124, eager float32, one CPU thread, five warmup
calls, and one-second minimum measurement windows. This is one recorded run,
not a benchmark of every subsequent revision.
[Workflow record](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/34706657808).

| Case | Mode | CUDA extension (ms) | PyTorch reference (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| small | forward | 0.0522 | 0.1721 | 3.30× |
| small | forward + backward | 0.4265 | 0.8757 | 2.05× |
| decoder | forward | 0.0514 | 0.2320 | 4.51× |
| decoder | forward + backward | 0.5026 | 1.2420 | 2.47× |
| batched | forward | 0.0783 | 0.8129 | 10.39× |
| batched | forward + backward | 0.3555 | 2.1476 | 6.04× |

### Run GPU benchmarks

Requires a visible CUDA GPU and a [CUDA-enabled extension](installation.md#select-cpu-or-cuda):

```bash
python benchmarks/benchmark_cuda.py --min-run-time 1.0 > benchmark-cuda.json
```

By default the harness measures eager float32 forward, backward-only, and
forward+backward execution, with five warmup calls and three repetitions.
Use `--execution compiled` and `--dtypes float16 bfloat16` to select Inductor and
AMP. The reference uses static shape tuples to avoid metadata transfers.

Outputs and all three gradients are checked against the eager reference before
timing. Location gradients are compared in feature-pixel units to account for
spatial resolution. JSON records each backend's median latency, IQR, CUDA-event
intervals, allocated memory, configuration, and provenance. Compute speedups
from rows with matching configuration and repetition. Float64 algorithm checks
remain in the CUDA regression suite. Exact workloads and tolerances are defined
in [the script](../benchmarks/benchmark_cuda.py).

### Run on Runpod

Configure the account and `RUNPOD_API_KEY` using the
[GPU runner guide](gpu-runner.md), including its cleanup and cost controls. Then:

```bash
gh workflow run cuda-benchmark.yml --repo S-aiueo32/torch-ms-deform-attn --ref main \
  --field operation=benchmark --field gpu='NVIDIA L4' \
  --field max_hourly_usd=0.50 --field timeout_minutes=45
```

Use `operation=check` to check credentials and pricing without creating a Pod.
The workflow builds and tests an installed CUDA wheel, runs the benchmark, and
deletes the Pod. A correctness or benchmark failure fails the job.

Download `cuda-benchmark-<run-id>-<attempt>` from the workflow run within seven
days. It contains `artifacts/benchmark-cuda.json`, GPU information, logs, and
built distributions.

## P2 measurement coverage and regression policy

`benchmark_compile.py --strict` emits diagnostics in JSON and exits nonzero on
any compile/correctness/timing failure. Failed rows never contain latency values.
The CUDA harness always exits nonzero on failure and discards partial timings
for the failed configuration. Its compiled paths use Inductor with fullgraph.

```bash
python benchmarks/benchmark_cuda.py --cases small decoder encoder \
  --batch 1 4 --channels 16 32 --steps 2 64 \
  --dtypes float32 float16 bfloat16 --execution eager compiled \
  --repeats 3 --warmup 5 --profile-kernels > cuda.json
python benchmarks/benchmark_compile.py --strict > compile.json
```

The complete Cartesian benchmark is intentionally opt-in through its dtype and
execution flags and can be expensive. Narrow dimensions for smoke checks.
Encoder uses Q=5440. Every timed implementation first passes forward and all
three gradient comparisons against the eager reference. Location gradients are
compared in feature-pixel units. AMP comparisons allow low-precision rounding.
CUDA benchmark coordinates stay
at least a quarter pixel away from interpolation knots, where a tiny rounding
change can select a different one-sided coordinate derivative. Dedicated operator
tests cover boundaries; the timing workload compares smooth sampling points.

Each repetition records synchronized host wall time (including Python dispatch),
its IQR, CUDA-event stream intervals and their IQR, and peak allocated memory.
Event intervals can include host-induced GPU idle gaps: they are **not pure kernel
time**. Optional profiler CUDA activity sums report kernel/copy activity durations
separately and are excluded from latency measurement; summed durations can exceed
elapsed time if activities overlap. Backward-only timing retains a graph constructed
before measurement, and its memory baseline includes that graph. Allocated memory
excludes allocator reservations, driver memory, and other processes.

Reports include CPU model, GPU/driver, PyTorch/CUDA, source SHA and working-tree
status when available, runtime build environment, seed, warmup, and repetitions.
Build environment variables describe the invocation, not reconstructed compiler
commands; unknown values are labeled unknown. Archive build logs beside results.

Compare only passing rows on identical hardware, software, build flags and case
configuration. Use at least three independent repetitions. Flag a latency regression
when the increase in median exceeds **both 5% and twice the larger IQR**; repeat the
entire comparison in a fresh process to confirm it before treating it as a regression.
Memory changes require matching baselines and a repeatable increase. This is an
investigation threshold, not a cross-machine performance guarantee. No operator
speedup here establishes a whole-model speedup.

# torch-deform-attn

Standalone **CPU/CUDA** multi-scale deformable attention for PyTorch, extracted from
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
Includes a C++ CPU kernel, the upstream CUDA kernels, first-order autograd, and
an independent `grid_sample` reference implementation. CPU builds need neither
the CUDA toolkit nor torchvision.

This is an initial source release, not yet published to PyPI. It implements the attention sampling/reduction operator;
projection layers and the detection model are outside the package.

## Install

Requires Python 3.10+, PyTorch 2.5+ (below 3), and a C++17 compiler. Tested locally
with Python 3.11 / PyTorch 2.5.1 on macOS arm64. Linux/macOS CI is included;
Windows, other PyTorch versions, and prebuilt wheels are not yet validated.
CUDA support is built automatically when CUDA-enabled PyTorch, the CUDA toolkit,
and a visible GPU are available. Otherwise the extension builds for CPU. Both
backends remain available in CUDA builds. MPS is unsupported.

The CUDA sampling and reduction algorithms are retained from upstream. Changes
are limited to packaging, current PyTorch APIs, launch error handling, input
checks, device/stream handling, partial batch chunks, 64-bit memory offsets,
determinism checks, and removal of redundant forward-output zero initialization.
Hosted CI builds CPU and CUDA wheels; CUDA runtime tests require a GPU runner
and have not yet been verified on GPU hardware.

From this repository:

```bash
python -m pip install torch 'setuptools>=77' 'packaging>=24.2' wheel ninja
python -m pip install --no-build-isolation .
```

Build against the PyTorch installed in the target environment. Rebuild after
changing PyTorch versions; locally built wheels are not guaranteed to work
across PyTorch versions. `--no-build-isolation` avoids building against a
separate, potentially different PyTorch installation.

To explicitly select a backend at build time:

```bash
FORCE_CPU=1 python -m pip install --no-build-isolation .
# Requires CUDA-enabled PyTorch and a matching CUDA toolkit:
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0;8.6" python -m pip install --no-build-isolation .
```

For builds without a visible GPU, set `TORCH_CUDA_ARCH_LIST` to the compute
capabilities of your deployment GPUs (the values above are examples).
The extension recompiles its sources when rebuilding, including when switching
CPU/CUDA or OpenMP settings.

CPU builds automatically enable OpenMP when the compiler can use the installed
PyTorch's OpenMP runtime. macOS wheels that bundle `omp.h` and `libomp.dylib`
need no additional OpenMP installation. Builds reuse PyTorch's runtime to avoid
loading a second OpenMP library. If headers are missing, install your compiler's
OpenMP development files or set `OMP_PREFIX` to a prefix containing `include/omp.h`.
If detection fails, the build reports the reason and uses a serial CPU kernel.
PyTorch builds with the native thread pool use that backend directly.
On macOS, extensions and wheels target the running Python/PyTorch architecture,
including with a universal2 Python. Cross-architecture `ARCHFLAGS` are rejected.

```bash
FORCE_OPENMP=1 python -m pip install --no-build-isolation .  # Require OpenMP; fail if unavailable.
FORCE_OPENMP=0 python -m pip install --no-build-isolation .  # Disable OpenMP (native stays native).
python -c 'from torch_deform_attn import _C; print(_C.cpu_parallel_backend)'
```

`torch.set_num_threads(...)` controls enabled CPU parallelism. The build diagnostic
reports `openmp`, `native`, or `serial`; PyTorch's own configuration alone does
not establish whether an extension was compiled with OpenMP.

## Use

```python
import torch
from torch_deform_attn import ms_deform_attn

shapes = torch.tensor([[4, 4], [2, 2]], dtype=torch.long)
starts = torch.tensor([0, 16], dtype=torch.long)
value = torch.randn(1, 20, 2, 8, requires_grad=True)
locations = torch.rand(1, 3, 2, 2, 4, 2, requires_grad=True)
weights = torch.full((1, 3, 2, 2, 4), 1 / 8, requires_grad=True)
output = ms_deform_attn(value, shapes, starts, locations, weights)
assert output.shape == (1, 3, 16)
output.square().mean().backward()
```

| Argument | Shape | Meaning |
| --- | --- | --- |
| `value` | `[N, S, M, D]` | Flattened feature levels; M heads, D channels per head |
| `spatial_shapes` | `[L, 2]` | Height and width per level, int64 |
| `level_start_index` | `[L]` | Each level's starting offset in S, int64 |
| `sampling_locations` | `[N, Q, M, L, P, 2]` | Normalized x/y sampling coordinates |
| `attention_weights` | `[N, Q, M, L, P]` | Weights used directly, without softmax |
| output | `[N, Q, M * D]` | Weighted sum of sampled features |

All inputs must be on the same CPU or CUDA device. Floating tensors must share
float32 or float64 dtype. For CUDA, move every tensor in the example to CUDA,
including `shapes` and `starts`.
Noncontiguous tensors are accepted. Sampling is bilinear, with zero padding and
`align_corners=False`; finite coordinates outside [0, 1] are allowed.
Nonfinite coordinates are not part of the supported input contract.
Gradients are provided for value, locations, and weights. Higher-order gradients
and native float16/bfloat16 kernel arithmetic are unsupported. AMP accepts
low-precision inputs by computing this operator in float32; see below.

`MSDeformAttnFunction.apply(value, shapes, starts, locations, weights, im2col_step)`
is also exported for code using the upstream autograd API. The positive
`im2col_step` argument is accepted for compatibility; the CPU kernel partitions
forward by batch/query/head and backward by batch/head, and does not use that
chunk size. Actual CPU parallel execution depends on build support. CUDA processes at most
`im2col_step` batch elements per launch, including a smaller final chunk; any
positive batch size fitting int32 is accepted. Empty CUDA dimensions remain
unsupported. CUDA memory offsets use int64; individual dimensions must fit int32,
and unsupported products or launch sizes are rejected before allocation/launch.
Reduce `im2col_step` if a batch chunk exceeds the CUDA grid limit.

CUDA backward uses atomic additions and is nondeterministic. With
`torch.use_deterministic_algorithms(True)`, backward raises an error;
`warn_only=True` emits a warning and permits execution.
Spatial shapes and level offsets must describe valid ranges in `value`. CUDA
validates these ranges on the device before accessing features, without a host
synchronization. Invalid metadata triggers a device assertion, which may surface
at the next CUDA synchronization. A device assertion invalidates the CUDA context;
restart the process after such an error.

## AMP and torch.compile

The public function and compatibility `.apply` entry point both support autocast
on CPU and CUDA. Inside autocast, float16/bfloat16 values, locations, and weights
are promoted to float32. Float32 inputs stay float32; float64 inputs are preserved.
Output is float32 for the low-precision path, and autograd casts gradients back
to the original input dtypes. Outside autocast, low-precision inputs are rejected.

```python
# All input tensors must already be on CUDA.
with torch.autocast("cuda", dtype=torch.float16):
    output = ms_deform_attn(value, shapes, starts, locations, weights)
# Backward can run outside the autocast context, including with GradScaler.
output.sum().backward()

compiled_attention = torch.compile(ms_deform_attn, fullgraph=True, dynamic=True)
output = compiled_attention(value.float(), shapes, starts,
                            locations.float(), weights.float())
```

Forward and backward are registered custom operators with FakeTensor and
autograd registrations. CPU tests cover full-graph compilation with AOT eager
and Inductor, dynamic batch/query sizes, compiled AMP, and `torch.library.opcheck`.
CUDA tests cover compiled forward/backward and AMP when a GPU is available.
Compilation treats the attention operator as opaque; it does not fuse its CUDA
kernel internals. Export/ONNX and higher-order differentiation are not claimed.

## CI

`.github/workflows/ci.yml` builds and tests installed CPU wheels on Linux with
OpenMP enabled and disabled, and on macOS with OpenMP enabled. Worker-count tests
check actual extension parallelism, and collision tests compare all gradients
across thread counts.

`.github/workflows/cuda-build.yml` runs on ordinary hosted Linux runners using a
pinned PyTorch/CUDA development image. It builds a CUDA wheel from the sdist and
runs the CPU, indexing, and integration tests against the installed wheel outside
the checkout. Indexing tests exercise offsets above 2^31 and overflow rejection
without allocating huge tensors. This workflow verifies CUDA compilation and
host-side integration; it has no GPU and skips CUDA runtime tests.

`.github/workflows/cuda.yml` provisions one Runpod GPU on manual dispatch, builds
and tests the installed CUDA wheel, collects logs, and deletes the Pod. An ordinary
GitHub-hosted job controls the GPU over SSH using the `RUNPOD_API_KEY` secret.
The default is an RTX A5000, a $0.50/hour compute-price limit, and a 45-minute
deadline. Its optional `sanitizer` input runs reduction and batch-chunk tests under
Compute Sanitizer's `memcheck`, `racecheck`, or `synccheck` tool. Use `operation=check`
to validate API access and pricing without renting a GPU.
See [Runpod setup and cleanup](docs/gpu-runner.md) for account configuration,
recovery after cancellation, and billing limits. No GPU CI run has been verified yet.

## Verify and benchmark

After installing the package:

```bash
python -m unittest discover -s tests -v
# CUDA tests run when the installed extension has CUDA support and a GPU is available.
python benchmarks/benchmark_cpu.py --threads 1
python benchmarks/benchmark_cpu.py --threads 4 --json
```

Tests compare forward values and all three gradients against the PyTorch
reference, run finite-difference gradcheck, and cover noncontiguous inputs,
boundary/out-of-bounds sampling, overlapping level offsets, colliding samples,
empty CPU dimensions, and invalid inputs.
CUDA tests also cover upstream channel-reduction paths, noncontiguous inputs,
nondefault streams, partial batch chunks, AMP, compilation, and multiple devices
when available. CPU-only runs skip them.
Benchmarks report median CPU forward and forward+backward latency for three
synthetic shapes. A speedup above 1 means C++ is faster. They do not measure
whole-model latency or peak memory. No general performance advantage is claimed.

### CPU kernel optimization

The CPU kernel computes the four neighbor offsets, boundary masks, and bilinear
coefficients once per sampling point, then reuses them across channels. Backward
accumulates coordinate and attention-weight gradients in local scalars before
writing them once. Workers own separate batch/head pairs, so repeated samples
and overlapping levels need no atomic additions.

Measured against a separate build of commit `733c73a`, on macOS 26.6.1 arm64,
Python 3.11.16 / PyTorch 2.5.1. The baseline and first optimized build used `-O3`
without OpenMP; the final build also enables OpenMP using PyTorch's bundled runtime.
All builds used a benchmark harness that checks outputs and all three gradients
against the reference before timing. Runs were sequential, eager float32, with
one-second minimum measurement windows. Times below are median milliseconds for
**forward + backward**; speedup compares the baseline with the final OpenMP build.

| Threads | Case | Before | Optimized serial | Optimized OpenMP | Speedup | Reference |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | small | 0.647 | 0.180 | 0.216 | 2.99x | 0.817 |
| 1 | decoder | 17.205 | 3.409 | 4.346 | 3.96x | 17.003 |
| 1 | batched | 69.506 | 15.499 | 20.495 | 3.39x | 48.075 |
| 4 | small | 0.654 | 0.183 | 0.092 | 7.13x | 0.779 |
| 4 | decoder | 17.786 | 3.663 | 1.189 | 14.96x | 7.419 |
| 4 | batched | 72.961 | 16.170 | 5.448 | 13.39x | 16.266 |

The OpenMP build scales forward + backward by 2.36–3.76x from one to four threads
on these cases. Its single-thread backward is slower than the optimized serial
build on this compiler; `FORCE_OPENMP=0` remains available for single-thread
deployments. At four threads, the final operator is 2.99–8.49x faster than the
reference. These are synthetic operator results on one machine; model and GPU
performance remain unmeasured.

[Raw before/after results](benchmarks/results/cpu-optimization.json) include
forward timings, IQRs, commands, and source hashes. Reproduce with
`python benchmarks/benchmark_cpu.py --threads 1 --min-run-time 1 --json` and
`--threads 4` after building and installing each version separately.

### Earlier local sample result

Measured source commit `5678212` on macOS 26.6.1 arm64, Python 3.11.16,
PyTorch 2.5.1, eager float32. Each backend/mode used a one-second minimum
measurement window; times are medians. Thread configurations were run sequentially.
The extension includes the dispatcher/AMP changes; autocast and compilation were
not enabled for these measurements. No CUDA GPU was available.

| Threads | Case | Mode | C++ (ms) | PyTorch (ms) | Speedup |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | small | Forward | 0.105 | 0.345 | 3.29x |
| 1 | small | Forward + backward | 0.664 | 0.849 | 1.28x |
| 1 | decoder | Forward | 2.545 | 6.386 | 2.51x |
| 1 | decoder | Forward + backward | 17.612 | 18.142 | 1.03x |
| 1 | batched | Forward | 10.958 | 22.472 | 2.05x |
| 1 | batched | Forward + backward | 73.970 | 50.306 | 0.68x |
| 4 | small | Forward | 0.104 | 0.293 | 2.81x |
| 4 | small | Forward + backward | 0.646 | 0.771 | 1.19x |
| 4 | decoder | Forward | 2.491 | 2.818 | 1.13x |
| 4 | decoder | Forward + backward | 17.241 | 7.352 | 0.43x |
| 4 | batched | Forward | 10.381 | 8.132 | 0.78x |
| 4 | batched | Forward + backward | 72.964 | 16.544 | 0.23x |

The C++ path wins the three single-thread forward cases. At four threads the
reference scales better: batched forward+backward takes 16.544 ms versus 72.964 ms
for C++, making C++ about 4.4x slower in that case.

That build does not enable OpenMP for the extension, although the installed
PyTorch uses OpenMP. In these headers `at::parallel_for` falls back to serial
execution without OpenMP compiler support. Thus setting four PyTorch threads does
not parallelize the C++ CPU kernel in that build. Also, the kernel at that commit
partitions by batch only, so batch-one workloads have no kernel-level parallelism.
These results describe this build, not a tuned multithreaded CPU implementation.

Raw results and reproduction parameters:
[1 thread](benchmarks/results/cpu-threads-1.json),
[4 threads](benchmarks/results/cpu-threads-4.json).
Exact shapes are defined in `benchmarks/benchmark_cpu.py`. These are synthetic
workloads on one machine, not whole-model or GPU results.

### Earlier torch.compile comparison

`benchmarks/benchmark_compile.py` compares eager and Inductor (`fullgraph=True`)
for both implementations. Forward and backward compilation and warmup happen
before timing. All timed paths first pass output or gradient comparisons against
the eager reference. Shape tuples are static in both eager and compiled reference
paths to avoid tensor-to-Python graph breaks, so their baseline differs from
`benchmark_cpu.py`. These are operator-only measurements, not a compiled model.

```bash
python benchmarks/benchmark_compile.py --threads 1 > compile-1.json
python benchmarks/benchmark_compile.py --threads 4 > compile-4.json
```

Same machine and PyTorch version as above; float32, median milliseconds,
one-second minimum measurement windows. No GPU was available.

| Threads | Case | Mode | C++ eager | C++ compiled | Reference eager | Reference compiled |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | small | Forward | 0.098 | 0.105 | 0.317 | 0.278 |
| 1 | small | Forward + backward | 0.627 | 0.639 | 0.795 | 0.603 |
| 1 | decoder | Forward | 2.369 | 2.393 | 5.940 | 8.388 |
| 1 | decoder | Forward + backward | 16.142 | 16.035 | 17.068 | 15.993 |
| 1 | batched | Forward | 9.953 | 10.065 | 21.134 | 47.352 |
| 1 | batched | Forward + backward | 66.823 | 67.215 | 48.470 | 72.788 |
| 4 | small | Forward | 0.105 | 0.111 | 0.287 | 0.289 |
| 4 | small | Forward + backward | 0.632 | 0.643 | 0.768 | 0.628 |
| 4 | decoder | Forward | 2.488 | 2.501 | 2.819 | 8.613 |
| 4 | decoder | Forward + backward | 17.097 | 17.096 | 7.402 | 10.948 |
| 4 | batched | Forward | 10.349 | 10.419 | 9.169 | 48.825 |
| 4 | batched | Forward + backward | 71.555 | 71.499 | 16.266 | 57.069 |

Compilation gives no meaningful improvement to the C++ operator in these runs.
It remains an opaque call, and compilation does not fix the extension's missing
OpenMP support. The reference improves for small forward+backward workloads,
but its larger workloads regress with this Inductor/PyTorch/platform combination.
Compilation is not a guaranteed speedup; full-model behavior may differ.

Raw results, variability (IQR), and measurement parameters:
[1 thread](benchmarks/results/compile-cpu-threads-1.json),
[4 threads](benchmarks/results/compile-cpu-threads-4.json).

## Build distributions

```bash
python -m pip install build
python -m build --no-isolation
```

CI builds an sdist, builds a wheel from that sdist, and runs the tests against
the installed wheel outside the checkout.

## Attribution

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Derived files retain the
original SenseTime copyright notices. This project adds the CPU implementation,
standalone packaging, tests, and benchmarks; it is not an official upstream release.

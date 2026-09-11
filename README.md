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
checks, device/stream handling, partial batch chunks, and removal of redundant
forward-output zero initialization. CUDA compilation and runtime tests have not
yet been verified on GPU hardware; the included hosted CI tests CPU builds.

From this repository:

```bash
python -m pip install torch setuptools wheel ninja
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
Use a clean checkout when switching CPU/CUDA builds to avoid stale object files.

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
`im2col_step` argument is accepted for compatibility; the CPU kernel parallelizes
over batch elements and does not use that chunk size. CUDA processes at most
`im2col_step` batch elements per launch, including a smaller final chunk; any
positive batch size is accepted. Empty CUDA dimensions remain unsupported. CUDA backward uses atomic additions and is nondeterministic.
Spatial shapes and level offsets must describe valid ranges in `value`; CUDA
checks metadata shapes/dtypes but does not copy metadata to the host to validate
those ranges on each call.

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

## GPU CI

`.github/workflows/cuda.yml` is a manually dispatched correctness workflow for a
self-hosted Linux x64 runner labeled `gpu`, with an NVIDIA GPU and a CUDA toolkit
compatible with the pinned PyTorch 2.5.1 installation. It builds the CUDA wheel
from the sdist and tests both CPU and CUDA, without running benchmarks.
The runner must be provisioned separately; adding this workflow alone does not
provide GPU capacity. No GPU CI run has been verified yet.

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
boundary/out-of-bounds sampling, level offsets, empty CPU inputs, and invalid inputs.
CUDA tests also cover upstream channel-reduction paths, noncontiguous inputs,
nondefault streams, partial batch chunks, AMP, compilation, and multiple devices
when available. CPU-only runs skip them.
Benchmarks report median CPU forward and forward+backward latency for three
synthetic shapes. A speedup above 1 means C++ is faster. They do not measure
whole-model latency or peak memory. No general performance advantage is claimed.

### Local sample result

Historical result from commit `dac69a8`, before dispatcher/AMP integration.
Not rerun for the current version. One run on macOS 26.6.1 arm64, Python 3.11.16, PyTorch 2.5.1, float32,
one CPU thread, 0.3-second minimum measurement windows:

| Case | Mode | C++ (ms) | PyTorch (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| Small | Forward | 0.102 | 0.332 | 3.27x |
| Small | Forward + backward | 0.601 | 0.818 | 1.36x |
| Decoder | Forward | 2.449 | 6.108 | 2.49x |
| Decoder | Forward + backward | 16.527 | 17.399 | 1.05x |
| Batched | Forward | 10.593 | 21.618 | 2.04x |
| Batched | Forward + backward | 69.809 | 49.726 | 0.71x |

The batched forward+backward case is slower in C++. These are synthetic workloads
on one machine, not a promise of faster training. Exact shapes are defined in
`benchmarks/benchmark_cpu.py`; rerun on your target hardware and thread count.

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

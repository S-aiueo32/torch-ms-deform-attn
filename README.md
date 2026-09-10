# torch-deform-attn

Standalone **CPU** multi-scale deformable attention for PyTorch, extracted from
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
Includes a C++ forward/backward kernel, first-order autograd, and an independent
`grid_sample` reference implementation. No CUDA toolkit or torchvision required.

This is an initial source release, not yet published to PyPI. It implements the attention sampling/reduction operator;
projection layers and the detection model are outside the package.

## Install

Requires Python 3.10+, PyTorch 2.5+ (below 3), and a C++17 compiler. Tested locally
with Python 3.11 / PyTorch 2.5.1 on macOS arm64. Linux/macOS CI is included;
Windows, other PyTorch versions, and prebuilt wheels are not yet validated.
The first release supports CPU tensors only, including when PyTorch itself has
CUDA support. It does not contain a CUDA or MPS backend.

From this repository:

```bash
python -m pip install torch setuptools wheel ninja
python -m pip install --no-build-isolation .
```

Build against the PyTorch installed in the target environment. Rebuild after
changing PyTorch versions; locally built wheels are not guaranteed to work
across PyTorch versions. `--no-build-isolation` avoids building against a
separate, potentially different PyTorch installation.

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

All inputs must be on CPU. Floating tensors must share float32 or float64 dtype.
Noncontiguous tensors are accepted. Sampling is bilinear, with zero padding and
`align_corners=False`; finite coordinates outside [0, 1] are allowed.
Nonfinite coordinates are not part of the supported input contract.
Gradients are provided for value, locations, and weights. Higher-order gradients,
float16/bfloat16, and torch.compile/export integration are unsupported.

`MSDeformAttnFunction.apply(value, shapes, starts, locations, weights, im2col_step)`
is also exported for code using the upstream autograd API. The positive
`im2col_step` argument is accepted for compatibility; the CPU kernel parallelizes
over batch elements and does not use that chunk size.

## Verify and benchmark

After installing the package:

```bash
python -m unittest discover -s tests -v
python benchmarks/benchmark_cpu.py --threads 1
python benchmarks/benchmark_cpu.py --threads 4 --json
```

Tests compare forward values and all three gradients against the PyTorch
reference, run finite-difference gradcheck, and cover noncontiguous inputs,
boundary/out-of-bounds sampling, level offsets, empty inputs, and invalid inputs.
Benchmarks report median CPU forward and forward+backward latency for three
synthetic shapes. A speedup above 1 means C++ is faster. They do not measure
whole-model latency or peak memory. No general performance advantage is claimed.

### Local sample result

One run on macOS 26.6.1 arm64, Python 3.11.16, PyTorch 2.5.1, float32,
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

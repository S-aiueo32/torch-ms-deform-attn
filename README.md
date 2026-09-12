# torch-deform-attn

Standalone CPU/CUDA multi-scale deformable attention for PyTorch, extracted from
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
Includes C++/CUDA kernels, first-order autograd, AMP, `torch.compile` support,
and an independent PyTorch reference. This package implements the sampling and
weighted reduction operator; projection layers and detection models are outside
its scope.

## Install

Requires Python 3.10+, PyTorch >=2.5,<3, and a C++17 compiler.
From the repository root:

```bash
python -m pip install 'torch>=2.5,<3' 'setuptools>=77' 'packaging>=24.2' wheel ninja
python -m pip install --no-build-isolation .
```

CPU builds need neither the CUDA toolkit nor torchvision. CUDA is built
automatically when CUDA-enabled PyTorch, the CUDA toolkit, and a visible GPU
are available; CUDA builds also support CPU. Rebuild after changing PyTorch.
See [installation and build options](docs/installation.md) for forced CPU/CUDA
builds, GPU architecture selection, and OpenMP configuration.

## Use

```python
import torch
from torch_deform_attn import ms_deform_attn

# Two feature levels: 4×4 and 2×2, flattened into 20 positions.
shapes = torch.tensor([[4, 4], [2, 2]], dtype=torch.long)
starts = torch.tensor([0, 16], dtype=torch.long)
value = torch.randn(1, 20, 2, 8, requires_grad=True)  # 2 heads, 8 channels each
locations = torch.rand(1, 3, 2, 2, 4, 2, requires_grad=True)
weights = torch.full((1, 3, 2, 2, 4), 1 / 8, requires_grad=True)
output = ms_deform_attn(value, shapes, starts, locations, weights)
assert output.shape == (1, 3, 16)
output.square().mean().backward()
```

All inputs must be on the same CPU or CUDA device, including `shapes` and
`starts`. Floating inputs must share float32 or float64 dtype.
Weights are used directly, without softmax. Sampling is bilinear with zero
padding and `align_corners=False`.

AMP computes float16/bfloat16 inputs in float32. Native low-precision kernels,
MPS, and higher-order gradients are unsupported. CUDA backward is
nondeterministic.

## Validation and documentation

CPU CI covers Linux and macOS with Python 3.11 / PyTorch 2.5.1. CUDA correctness
and Compute Sanitizer memcheck have been recorded on an NVIDIA L4 with CUDA 12.4;
see the [GPU validation record](docs/gpu-runner.md#run-tests).
The declared dependency range is broader than this tested configuration.

- [Installation](docs/installation.md): requirements, backend selection, CPU parallelism.
- [API](docs/api.md): arguments, reference implementation, AMP, compilation, limitations.
- [Development and CI](docs/development.md): tests, distributions, workflow coverage.
- [Benchmarks](docs/benchmarks.md): measure CPU latency and compare eager/compiled execution.
- [GPU runner](docs/gpu-runner.md): Runpod setup, CUDA checks, cleanup, and costs.

## Attribution

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Derived files retain the
original SenseTime copyright notices. This project adds the CPU implementation,
standalone packaging, tests, and benchmarks; it is not an official upstream release.

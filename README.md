# torch-ms-deform-attn

Multi-scale deformable attention for PyTorch on CPU and CUDA, extracted from
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
Use the sampling and weighted reduction operator in your model with first-order
autograd, AMP, and `torch.compile`. Projection layers and detection models are
outside the package's scope.

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

## Use

```python
import torch
from torch_ms_deform_attn import ms_deform_attn

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
Weights are used directly, without softmax.

AMP computes float16/bfloat16 inputs in float32. Native low-precision kernels,
MPS, and higher-order gradients are unsupported. CUDA backward is
nondeterministic.

## Validation and documentation

For development, use uv to create `.venv`, install locked dependencies, and build
the editable extension:

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

CPU CI targets Linux with the Python/PyTorch combinations in the
[support matrix](docs/installation.md#prerequisites). CUDA correctness
and Compute Sanitizer memcheck have been recorded on an NVIDIA L4 with CUDA 12.4;
see the [GPU validation record](docs/gpu-runner.md#run-tests).
The declared dependency range is broader than this tested configuration.

- [Installation](docs/installation.md): requirements, backend selection, CPU parallelism.
- [API](docs/api.md): arguments, reference implementation, AMP, compilation, limitations.
- [Upstream compatibility](docs/compatibility.md): pinned Deformable-DETR module and training-step checks.
- [Development and CI](docs/development.md): tests, uv builds and PyPI publishing, workflow coverage.
- [Benchmarks](docs/benchmarks.md): CPU/CUDA results and measurement commands.
- [GPU runner](docs/gpu-runner.md): Runpod setup, CUDA checks, cleanup, and costs.

## Attribution

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Derived files retain the
original SenseTime copyright notices. This project adds the CPU implementation,
standalone packaging, tests, and benchmarks; it is not an official upstream release.

# torch-ms-deform-attn

Multi-scale deformable attention for PyTorch on CPU and CUDA, extracted from
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
Use the sampling and weighted reduction operator in your model with first-order
autograd, AMP, and `torch.compile`. Projection layers and detection models are
outside the package's scope.

## Install

Install the published release candidate from
[PyPI](https://pypi.org/project/torch-ms-deform-attn/0.1.0rc2/).
The current source requires Python 3.10+, PyTorch >=2.4.0,<3, and a C++ compiler
compatible with PyTorch (C++17 through 2.12; C++20 from 2.13). The package is
distributed as source and compiles against the PyTorch in your environment:

```bash
python -m pip install 'torch>=2.5,<3' 'setuptools>=77' 'packaging>=24.2' wheel ninja
python -m pip install --no-build-isolation torch-ms-deform-attn==0.1.0rc2
```

The published `0.1.0rc2` still requires PyTorch >=2.5,<3. To use 2.4.x,
install from the current source as described in [installation](docs/installation.md).

CPU builds need neither the CUDA toolkit nor torchvision. CUDA is built
automatically when CUDA-enabled PyTorch, the CUDA toolkit, and a visible GPU
are available; CUDA builds also support CPU. Install your chosen CPU/CUDA PyTorch
build first. The explicit RC version does not require `--pre`.
See [installation and backend selection](docs/installation.md) for CPU/CUDA
examples, verification, and rebuilding after changing PyTorch.

## Support matrix

Tested combinations on Linux x86_64. **✅ Verified · ➖ Unverified**

### CPU

| PyTorch | Tested Python | Status |
| --- | --- | :---: |
| 2.4.0 | 3.10, 3.11 | ✅ |
| 2.5.0 | 3.10 | ✅ |
| 2.5.1 | 3.11 | ✅ |
| 2.7.1 | 3.11, 3.12 | ✅ |
| 2.8.0 | 3.11, 3.12 | ✅ |
| 2.9.1 | 3.12 | ✅ |
| 2.10.0 | 3.12 | ✅ |
| 2.11.0 | 3.12 | ✅ |
| 2.12.1 | 3.12 | ✅ |
| 2.13.0 | 3.12 | ✅ |
| 2.14.0 | 3.12 | ✅ |

### GPU (CUDA)

| PyTorch | Tested Python | Status |
| --- | --- | :---: |
| 2.4.0 | 3.11 | ✅ |
| 2.5.1 | 3.11 | ✅ |
| 2.7.1 | 3.11 | ✅ |
| 2.8.0 | 3.11 | ✅ |
| 2.14.0 | 3.12 | ✅ |

➖ Other GPU combinations, including PyTorch 2.9–2.13, are unverified.

CUDA validation selects a CUDA build from
[PyTorch's official version-specific builds](https://pytorch.org/get-started/previous-versions/)
and uses a matching toolkit. Each ✅ covers that tested pair; exact CUDA versions
are recorded in the [detailed matrix](docs/installation.md#prerequisites).

### Scope

Other Python/PyTorch combinations, CUDA toolkits, macOS and Windows are best
effort. The dependency range permits more versions than this tested matrix.
MPS is unsupported.

The 2.4.0 and 2.8–2.14 entries describe current-source validation; the published
`0.1.0rc2` requires PyTorch >=2.5. Results are tied to specific source revisions:
see the [2.4.0 evidence](docs/validation/2026-09-17-pytorch240/README.md),
[2.8–2.14 evidence](docs/validation/2026-09-17-newer-pytorch/README.md), and
[earlier validation records](docs/installation.md#prerequisites).

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
`starts`. Outside autocast, floating inputs must share float16, bfloat16,
float32, or float64 dtype.
Weights are used directly, without softmax.

Explicit `.half()` / `.bfloat16()` inputs compute in float32 and return the input
dtype. AMP computes low-precision inputs in float32 and returns float32.
Native low-precision kernels,
MPS, and higher-order gradients are unsupported. CUDA backward is
nondeterministic.

## Validation and documentation

For development, use uv to create `.venv`, install locked dependencies, and build
the editable extension:

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

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

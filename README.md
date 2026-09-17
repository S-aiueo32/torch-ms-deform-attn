# torch-ms-deform-attn

Multi-scale deformable attention for PyTorch on CPU, CUDA, and Apple Silicon MPS, extracted from
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
Use the sampling and weighted reduction operator in your model with first-order
autograd, AMP, and `torch.compile`. Projection layers and detection models are
outside the package's scope.

## Install

Install `0.1.0` from PyPI. It requires Python 3.10+,
PyTorch >=2.4,<3, and a compatible C++ compiler. Installation builds the extension
from source against your installed PyTorch:

```bash
python -m pip install 'torch>=2.4,<3' 'setuptools>=77' 'packaging>=24.2' wheel ninja
python -m pip install --no-build-isolation torch-ms-deform-attn==0.1.0
```

CPU builds need neither the CUDA toolkit nor torchvision. CUDA is built
automatically when CUDA-enabled PyTorch, the CUDA toolkit, and a visible GPU
are available; CUDA builds also support CPU. Install your chosen CPU/CUDA PyTorch
build first.
See [installation and backend selection](docs/installation.md) for CPU/CUDA
examples, verification, and rebuilding after changing PyTorch.

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

All inputs must be on the same device, including `shapes` and
`starts`. Outside autocast, floating inputs must share float16, bfloat16,
float32, or float64 dtype.
Weights are used directly, without softmax.

Explicit `.half()` / `.bfloat16()` inputs compute in float32 and return the input
dtype. AMP computes low-precision inputs in float32 and returns float32.
Native low-precision arithmetic and higher-order gradients are unsupported. CUDA backward is
nondeterministic.

Apple Silicon MPS uses dedicated Metal forward
and backward kernels (macOS 13.3+, PyTorch 2.4+). Install from source using the
[MPS installation instructions](docs/installation.md#apple-silicon-mps).
All five inputs must be on MPS. float32 and float16 inputs are supported; bfloat16
requires macOS 14+. float64, empty dimensions, and MPS `torch.compile` support are
excluded. MPS backward is nondeterministic.

## Support matrix

See the [support matrix](docs/support.md) for tested Python/PyTorch
pairs, CUDA toolkits, and known `torch.compile` limitations. The maintained
CPU/CUDA matrix covers Linux x86_64 with standard CPython. Apple Silicon MPS
has a separate [validation record](docs/validation/mps/README.md); other macOS
and Windows configurations are best effort.
Results apply to the source revisions recorded in each validation report.

## Documentation

| Task | Guide |
| --- | --- |
| Install, choose a backend, or rebuild | [Installation](docs/installation.md) |
| Look up tensor shapes, dtypes, and errors | [API reference](docs/api.md) |
| Replace the upstream operator | [Deformable-DETR integration](docs/compatibility.md) |
| Choose a validated environment | [Support matrix](docs/support.md) |
| Make a change and run checks | [Contributing](CONTRIBUTING.md) |
| Build packages or publish a release | [Development and CI](docs/development.md) |
| Measure operator performance | [Benchmarks](docs/benchmarks.md) |
| Run GPU validation on Runpod | [GPU runner](docs/gpu-runner.md) |
| Inspect recorded test results | [Validation evidence](docs/validation/README.md) |

## Attribution

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Derived files retain the
original SenseTime copyright notices. This project adds the CPU implementation,
standalone packaging, tests, and benchmarks; it is not an official upstream release.

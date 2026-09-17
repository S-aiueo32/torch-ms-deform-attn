# Installation and build options

[Back to README](../README.md)

The release candidate `0.1.0rc2` is available on
[PyPI](https://pypi.org/project/torch-ms-deform-attn/0.1.0rc2/).
It is distributed as a source archive (sdist), so installation compiles the
extension against your environment's PyTorch, with CPU or CPU/CUDA support.

For repository development, follow the [uv setup](development.md#set-up-with-uv):
`uv sync --locked` creates `.venv` and installs an editable build with the locked
development dependencies. The instructions below cover installation into an
existing environment with your chosen PyTorch version.

## Prerequisites

| Requirement | Published `0.1.0rc2` | Current checkout |
| --- | --- | --- |
| Python | 3.10+ | 3.10+ |
| PyTorch | >=2.5,<3 | >=2.4.0,<3 |
| C++ compiler | Compatible with installed PyTorch | Compatible with installed PyTorch |
| CUDA execution | CUDA-enabled PyTorch, matching CUDA toolkit, NVIDIA GPU | Same |

The compiler needs C++17 through PyTorch 2.12 and C++20 from 2.13;
`BuildExtension` selects the standard. CPU builds need neither the CUDA toolkit
nor torchvision. MPS is unsupported.

Check the [support matrix](support.md) before choosing versions: the dependency
range is broader than the tested combinations. PyTorch 2.4.0 requires the current
checkout; see [Install from a checkout](#install-from-a-checkout). For repository
work, the development lock uses PyTorch 2.5.1.

## Install from PyPI

Run these commands in your target Python environment; no repository checkout
is needed. Install your chosen PyTorch build first, then the build tools and
this package. If PyTorch is already installed, keep that build as long as it
satisfies `>=2.5,<3`.

```bash
python -m pip install 'torch>=2.5,<3'
python -m pip install 'setuptools>=77' 'packaging>=24.2' wheel ninja
python -m pip install --no-build-isolation torch-ms-deform-attn==0.1.0rc2
```

`--no-build-isolation` uses the installed PyTorch and build tools instead of
creating a separate build environment with a potentially different PyTorch.
They must be installed before running the last command.
The exact `==0.1.0rc2` pin selects the prerelease without `--pre`; see
[pip's prerelease handling](https://pip.pypa.io/en/stable/cli/pip_install/#pre-release-versions).

### Linux CPU example

This uses the validated Python 3.11 / PyTorch 2.5.1 combination:

```bash
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'setuptools>=77' 'packaging>=24.2' wheel ninja
FORCE_CPU=1 python -m pip install --no-build-isolation torch-ms-deform-attn==0.1.0rc2
```

### Linux CUDA example

For Python 3.11 with the CUDA 12.4 toolkit (`nvcc`) installed and a visible GPU:

```bash
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu124
python -m pip install 'setuptools>=77' 'packaging>=24.2' wheel ninja
FORCE_CUDA=1 python -m pip install --no-build-isolation torch-ms-deform-attn==0.1.0rc2
```

The PyTorch wheel alone does not supply the CUDA compiler. `FORCE_CUDA=1` makes
missing CUDA build prerequisites an error. For machines without a visible GPU,
also set the target architectures as shown below.

## Install from a checkout

From the repository root, install your chosen PyTorch version and build tools
first. This example uses the current source's minimum PyTorch version:

```bash
python -m pip install 'torch==2.4.0' 'setuptools>=77' 'packaging>=24.2' wheel ninja
python -m pip install --no-build-isolation .
```

This installs the checked-out source, which may contain changes newer than the
published RC. For an editable development install, use
[uv setup](development.md#set-up-with-uv).

## Verify the installation

```bash
python - <<'PY'
from importlib.metadata import version
import torch
from torch_ms_deform_attn import _C

print("Package:", version("torch-ms-deform-attn"))
print("PyTorch:", torch.__version__)
print("Extension CUDA support:", _C.with_cuda)
print("CUDA device available:", torch.cuda.is_available())
print("CPU parallel backend:", _C.cpu_parallel_backend)
PY
```

CUDA execution needs both extension CUDA support and an available CUDA device.
The [README example](../README.md#use) also exercises forward and backward on CPU.

## Select CPU or CUDA

CUDA support is built automatically when CUDA-enabled PyTorch, the CUDA toolkit,
and a visible GPU are available. Otherwise the extension builds for CPU.
CUDA builds also support CPU. To select a backend explicitly:

```bash
FORCE_CPU=1 python -m pip install --no-build-isolation torch-ms-deform-attn==0.1.0rc2
# Requires CUDA-enabled PyTorch and a matching CUDA toolkit:
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0;8.6" python -m pip install --no-build-isolation torch-ms-deform-attn==0.1.0rc2
```

For builds without a visible GPU, set `TORCH_CUDA_ARCH_LIST` to the compute
capabilities of your deployment GPUs (the values above are examples).
Replace the package requirement with `.` to build a local checkout. If the
package is already installed or you previously built it with other settings,
use the rebuild command below so pip does not reuse the installed or cached wheel.

## Rebuild after changing PyTorch or build options

Rebuild after changing PyTorch, CPU/CUDA selection, GPU architectures, or OpenMP
settings. Locally compiled wheels are not guaranteed to work across PyTorch
versions, and pip can reuse a cached wheel built with previous settings.
After installing your chosen PyTorch version, force a fresh source build:

```bash
python -m pip install --no-build-isolation --no-deps --force-reinstall \
  --no-cache-dir --no-binary=torch-ms-deform-attn torch-ms-deform-attn==0.1.0rc2
```

Prefix this command with the desired build variables, such as `FORCE_CUDA=1`.
`--no-deps` preserves the PyTorch installation you selected. The cache and binary
options ensure the extension is rebuilt from source; see
[pip's wheel cache behavior](https://pip.pypa.io/en/stable/topics/caching/).

## Configure CPU parallelism

Builds automatically enable OpenMP when the compiler can use the installed
PyTorch's OpenMP runtime. macOS wheels that bundle `omp.h` and `libomp.dylib`
need no additional OpenMP installation. Builds reuse PyTorch's runtime to avoid
loading a second OpenMP library. If headers are missing, install your compiler's
OpenMP development files or set `OMP_PREFIX` to a prefix containing `include/omp.h`.
If detection fails, the build reports the reason and uses a serial CPU kernel.
PyTorch builds with the native thread pool use that backend directly.

```bash
# Require OpenMP; fail if unavailable:
FORCE_OPENMP=1 python -m pip install --no-build-isolation torch-ms-deform-attn==0.1.0rc2
# Disable OpenMP (native stays native):
FORCE_OPENMP=0 python -m pip install --no-build-isolation torch-ms-deform-attn==0.1.0rc2
python -c 'from torch_ms_deform_attn import _C; print(_C.cpu_parallel_backend)'
```

For an existing installation, apply these variables to the
[rebuild command](#rebuild-after-changing-pytorch-or-build-options).
`torch.set_num_threads(...)` controls enabled CPU parallelism. The build diagnostic
reports `openmp`, `native`, or `serial`; PyTorch's own configuration alone does
not establish whether an extension was compiled with OpenMP.

## macOS architecture

Extensions and wheels target the running Python/PyTorch architecture, including
with a universal2 Python. Cross-architecture `ARCHFLAGS` are rejected.

## Other build options

`USE_NINJA=0` selects the setuptools compiler path; the default `USE_NINJA=1`
uses Ninja when available (PyTorch falls back if Ninja is absent). This option
and `ARCHFLAGS` participate in uv's extension-build cache key.
An explicit `OMP_PREFIX` without `include/omp.h` makes the OpenMP probe fail:
auto mode falls back to serial, while `FORCE_OPENMP=1` fails with an explanation.

Native thread-pool PyTorch remains best effort until a real native-backend build
is recorded. For backend transition checks, see [development tests](development.md#run-tests)
and the [CPU build evidence](validation/execution-and-benchmarks/README.md).

## Install from TestPyPI

TestPyPI is used to test the publishing process. For normal use, install from
PyPI as above. To check the TestPyPI archive, install prerequisites from PyPI
first, then select only this package from TestPyPI:

```bash
python -m pip install 'torch>=2.5,<3' 'setuptools>=77' 'packaging>=24.2' wheel ninja
python -m pip install --no-build-isolation --no-deps --no-cache-dir \
  --index-url https://test.pypi.org/simple/ torch-ms-deform-attn==0.1.0rc1
```

Use a fresh environment, or add `--force-reinstall` if the same version is
already installed. Publisher configuration is in the
[development guide](development.md#testpypi-setup-and-installation).

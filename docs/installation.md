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

## Current source compatibility

The current checkout permits `torch>=2.4.0,<3` with Python 3.10+.
PyTorch 2.4 introduced `torch.library.custom_op`,
which this package requires; earlier versions cannot import the package.
Local validation on macOS arm64 / Python 3.11 / PyTorch 2.4.0 passed the CPU
OpenMP build, all 40 runnable tests (including AMP, FakeTensor, and dynamic
`torch.compile` forward/backward), and 7 build-policy tests. The 26 GPU-dependent
tests were skipped. Linux CPU serial (Python 3.10) and OpenMP (Python 3.11)
also passed the installed-wheel suite and build-policy tests. CUDA builds
require the current source's assertion compatibility fix for PyTorch 2.4.
With that fix, the full CUDA 12.4 suite and all four Compute Sanitizer tools
passed on an NVIDIA L4; see the
[2.4.0 validation record](validation/2026-09-17-pytorch240/README.md).

For a source install with PyTorch 2.4.0, run from the repository root:

```bash
python -m pip install 'torch==2.4.0' 'setuptools>=77' 'packaging>=24.2' wheel ninja
python -m pip install --no-build-isolation .
```

The release installation commands and historical validation records below still
refer to the published releases. The development lock remains on PyTorch 2.5.1.

## Prerequisites

The published `0.1.0rc2` requires Python 3.10+ and PyTorch >=2.5,<3.
The compiler must support the C++ standard required by the installed PyTorch:
C++17 through 2.12, C++20 from 2.13. `BuildExtension` selects that standard.
CUDA builds also
require CUDA-enabled PyTorch and a matching CUDA toolkit. MPS is unsupported.
The dependency range permits installation; it is not a tested Cartesian product.
The maintained support and CI matrix is Linux-only. The validation targets
below use published wheels from the
[official PyTorch version table](https://pytorch.org/get-started/previous-versions/).
Other Python/PyTorch versions in the dependency range, macOS, Windows, and other CUDA
pairs are best effort. The full, manually dispatched CPU matrix includes each
minor series from 2.7 to 2.14, selecting one patch release per series; it does
not test every patch. Normal PR checks use three minimum/latest configurations;
see [Actions usage](development.md#actions-usage).

| Platform | Python | PyTorch | Backend/toolkit | Build / test evidence |
| --- | --- | --- | --- | --- |
| Linux | 3.10 | 2.4.0 | CPU serial (current source lower bounds) | [Installed-wheel tests passed](validation/2026-09-17-pytorch240/README.md) |
| Linux | 3.11 | 2.4.0 | CPU OpenMP | [Installed-wheel tests passed](validation/2026-09-17-pytorch240/README.md) |
| Linux | 3.11 | 2.4.0 | CUDA 12.4 | [L4 full suite and all four sanitizers passed](validation/2026-09-17-pytorch240/README.md) |
| Linux | 3.10 | 2.5.0 | CPU serial (published RC lower bounds) | RC CPU package job passed |
| Linux | 3.11 | 2.5.1 | CPU OpenMP | RC CPU package job passed |
| Linux | 3.12 | 2.7.1 | CPU OpenMP | RC CPU package job passed |
| Linux | 3.12 | 2.8.0 | CPU OpenMP | [Installed-wheel tests passed](validation/2026-09-17-newer-pytorch/README.md) |
| Linux | 3.12 | 2.9.1 | CPU OpenMP | [Installed-wheel tests passed](validation/2026-09-17-newer-pytorch/README.md) |
| Linux | 3.12 | 2.10.0 | CPU OpenMP | [Installed-wheel tests passed](validation/2026-09-17-newer-pytorch/README.md) |
| Linux | 3.12 | 2.11.0 | CPU OpenMP | [Installed-wheel tests passed](validation/2026-09-17-newer-pytorch/README.md) |
| Linux | 3.12 | 2.12.1 | CPU OpenMP | [Installed-wheel tests passed](validation/2026-09-17-newer-pytorch/README.md) |
| Linux | 3.12 | 2.13.0 | CPU OpenMP | [Installed-wheel tests passed](validation/2026-09-17-newer-pytorch/README.md) |
| Linux | 3.12 | 2.14.0 | CPU OpenMP | [Installed-wheel tests passed](validation/2026-09-17-newer-pytorch/README.md) |
| Linux | 3.11 | 2.5.1 | CUDA 12.4 | RC L4 full suite and all four sanitizers passed |
| Linux | 3.11 | 2.7.1 | CUDA 12.6 | Local L4 sdist-to-wheel build, full suite and all four sanitizers passed |
| Linux | 3.11 | 2.8.0 | CUDA 12.6 | [L4 full suite and all four sanitizers passed](validation/2026-09-17-newer-pytorch/README.md) |
| Linux | 3.12 | 2.14.0 | CUDA 12.6 | [L4 full suite and all four sanitizers passed](validation/2026-09-17-newer-pytorch/README.md) |

The rows above preserve the earlier per-version validation records. The initial
2.8–2.14 validation selected 2.8.0 and 2.14.0 for CUDA runtime checks; its exact
source identities and sanitizer reports remain in the
[newer-PyTorch validation record](validation/2026-09-17-newer-pytorch/README.md).

The [Python-by-PyTorch tables](../README.md#support-matrix) also include the
[expanded Python matrix](validation/2026-09-17-support-matrix/README.md), which
tests the intermediate releases on CUDA and adds separate CPU-only wheel runs.
Those runs use CUDA 12.4 for 2.4.0/2.5.0/2.5.1 and CUDA 12.6 for the listed
2.7–2.14 releases. Each record identifies its Python version and source SHA.
The local Docker records use CPU PyTorch wheels; the GPU-host CPU-only extension
records use the matching CUDA-enabled PyTorch installation. Both execute the
CPU suite, and neither CPU result alone establishes CUDA support.

PyTorch 2.5.0/2.5.1 with Python 3.13 and 2.9.1 with Python 3.14 reject
`torch.compile` upstream. Their builds and eager tests do not establish full
feature support; the matrices mark the failed full suites with ⚠️. These results
describe current-source validation, not a new package release.

The CUDA 12.6 local result above refers only to source
`c29ca3640b8a7a0cf4fc9907d40ccde4d42f9045`; see the
[full logs and environment records](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/fedcad0064f3cea79146be9ca181c97f53aeeda8/docs/validation/2026-09-14-p1/README.md).
Historical macOS runs remain in the evidence archive as supplemental results;
they do not add macOS to the support matrix. The RC results are linked below.

Each CPU job writes its SHA and result to the Actions summary. The manual full-matrix `rebuild` job
builds and tests 2.5.0 then 2.7.1 in the same checkout and Python 3.11 environment.
A build-only CUDA success does not establish GPU runtime support. Dispatch CUDA
correctness once per pair using `torch_version`; actual Python/toolkit versions
appear in the logs. Until successful build and runtime evidence exists, new rows
are validation targets with best-effort status, not certified combinations.

### Published RC validation

For `0.1.0rc2`, the exact source revision, CPU/GPU workflow results, and
CUDA 12.4/12.6 test and sanitizer reports are recorded in the
[release notes and attached evidence](https://github.com/S-aiueo32/torch-ms-deform-attn/releases/tag/v0.1.0rc2).
This candidate adds explicit `.half()` / `.bfloat16()` inputs with float32
computation and output in the input dtype.

For `0.1.0rc1` (source `a05428cbbaf9ba8352fbf3e798d78203f30d6a21`), the
[Linux CPU matrix](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/34912110258)
passed for Python/PyTorch 3.10/2.5.0 (serial), 3.11/2.5.1 (OpenMP), and
3.12/2.7.1 (OpenMP), including the rebuild job.
[GPU release validation](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/34912110099)
passed on L4 with Python 3.11, PyTorch 2.5.1 and CUDA 12.4: 54 tests,
2 expected skips, and all four Compute Sanitizer tools passed. The reports are
attached to the [release](https://github.com/S-aiueo32/torch-ms-deform-attn/releases/tag/v0.1.0rc1).
These results apply to that RC; the CUDA 12.6 row records historical validation.

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

From the repository root:

```bash
python -m pip install 'torch>=2.5,<3' 'setuptools>=77' 'packaging>=24.2' wheel ninja
python -m pip install --no-build-isolation .
```

This installs the checked-out source, which may contain changes newer than the
published RC. For an editable development install, use
[uv setup](development.md#set-up-with-uv).

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


### Build selection validation

`USE_NINJA=0` selects the setuptools compiler path; the default `USE_NINJA=1`
uses Ninja when available (PyTorch falls back if Ninja is absent). This option
and `ARCHFLAGS` participate in uv's extension-build cache key.
An explicit `OMP_PREFIX` without `include/omp.h` makes the OpenMP probe fail:
auto mode falls back to serial, while `FORCE_OPENMP=1` fails with an explanation.

`python scripts/validate_cpu_builds.py --output /tmp/msda-build-results` builds
and loads real extensions through auto/Ninja, serial/setuptools, invalid-prefix
fallback, forced failure, and OpenMP restoration in the same temporary checkout.
It requires an OpenMP PyTorch build and compiler support. Mocked policy tests in
`build_tests` also cover native/serial PyTorch backends, absent CUDA toolkit,
conflicting selection flags, and architecture mismatch. Native thread-pool
PyTorch remains best effort until a real native-backend build is recorded; mocked
selection coverage does not establish binary compatibility.

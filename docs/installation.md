# Installation and build options

[Back to README](../README.md)

Build and install the extension against your environment's PyTorch, with CPU
or CPU/CUDA support.

For repository development, follow the [uv setup](development.md#set-up-with-uv):
`uv sync --locked` creates `.venv` and installs an editable build with the locked
development dependencies. The instructions below cover installation into an
existing environment with your chosen PyTorch version.

## Prerequisites

Requires Python 3.10+, PyTorch >=2.5,<3, and a C++17 compiler. CUDA builds also
require CUDA-enabled PyTorch and a matching CUDA toolkit. MPS is unsupported.
The dependency range permits installation; it is not a tested Cartesian product.
The maintained support and CI matrix is Linux-only. The validation targets
below use published wheels from the
[official PyTorch version table](https://pytorch.org/get-started/previous-versions/).
Other Python/PyTorch versions in the dependency range, macOS, Windows, and other CUDA
pairs are best effort. 2.7.1 is the selected newer regression series, not a claim
that it is the newest available release.

| Platform | Python | PyTorch | Backend/toolkit | Build / test evidence |
| --- | --- | --- | --- | --- |
| Linux | 3.10 | 2.5.0 | CPU serial (declared lower bounds) | CPU package job; not run (Actions unavailable) |
| Linux | 3.11 | 2.5.1 | CPU OpenMP | CPU tests passed in the CUDA wheel; CPU-only wheel job unverified |
| Linux | 3.12 | 2.7.1 | CPU OpenMP | CPU package job; not run (Actions unavailable) |
| Linux | 3.11 | 2.5.1 | CUDA 12.4 | Local L4 sdist-to-wheel build, full suite and all four sanitizers passed |
| Linux | 3.11 | 2.7.1 | CUDA 12.6 | Local L4 sdist-to-wheel build, full suite and all four sanitizers passed |

The successful local results above refer only to source
`c29ca3640b8a7a0cf4fc9907d40ccde4d42f9045`; see the
[full logs and environment records](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/fedcad0064f3cea79146be9ca181c97f53aeeda8/docs/validation/2026-09-14-p1/README.md).
Historical macOS runs remain in the evidence archive as supplemental results;
they do not add macOS to the support matrix. Unexecuted Linux CPU-only rows
remain best effort.

Each CPU job writes its SHA and result to the Actions summary. The `rebuild` job
builds and tests 2.5.0 then 2.7.1 in the same checkout and Python 3.11 environment.
A build-only CUDA success does not establish GPU runtime support. Dispatch CUDA
correctness once per pair using `torch_version`; actual Python/toolkit versions
appear in the logs. Until successful build and runtime evidence exists, new rows
are validation targets with best-effort status, not certified combinations.

## Install from source

From the repository root:

```bash
python -m pip install 'torch>=2.5,<3' 'setuptools>=77' 'packaging>=24.2' wheel ninja
python -m pip install --no-build-isolation .
```

Build against the PyTorch installed in the target environment. Rebuild after
changing PyTorch versions; locally built wheels are not guaranteed to work
across PyTorch versions. `--no-build-isolation` avoids building against a
separate, potentially different PyTorch installation.

## Select CPU or CUDA

CUDA support is built automatically when CUDA-enabled PyTorch, the CUDA toolkit,
and a visible GPU are available. Otherwise the extension builds for CPU.
CUDA builds also support CPU. To select a backend explicitly:

```bash
FORCE_CPU=1 python -m pip install --no-build-isolation .
# Requires CUDA-enabled PyTorch and a matching CUDA toolkit:
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0;8.6" python -m pip install --no-build-isolation .
```

For builds without a visible GPU, set `TORCH_CUDA_ARCH_LIST` to the compute
capabilities of your deployment GPUs (the values above are examples).
The extension recompiles its sources when rebuilding, including when switching
CPU/CUDA or OpenMP settings.

## Configure CPU parallelism

Builds automatically enable OpenMP when the compiler can use the installed
PyTorch's OpenMP runtime. macOS wheels that bundle `omp.h` and `libomp.dylib`
need no additional OpenMP installation. Builds reuse PyTorch's runtime to avoid
loading a second OpenMP library. If headers are missing, install your compiler's
OpenMP development files or set `OMP_PREFIX` to a prefix containing `include/omp.h`.
If detection fails, the build reports the reason and uses a serial CPU kernel.
PyTorch builds with the native thread pool use that backend directly.

```bash
FORCE_OPENMP=1 python -m pip install --no-build-isolation .  # Require OpenMP; fail if unavailable.
FORCE_OPENMP=0 python -m pip install --no-build-isolation .  # Disable OpenMP (native stays native).
python -c 'from torch_ms_deform_attn import _C; print(_C.cpu_parallel_backend)'
```

`torch.set_num_threads(...)` controls enabled CPU parallelism. The build diagnostic
reports `openmp`, `native`, or `serial`; PyTorch's own configuration alone does
not establish whether an extension was compiled with OpenMP.

## macOS architecture

Extensions and wheels target the running Python/PyTorch architecture, including
with a universal2 Python. Cross-architecture `ARCHFLAGS` are rejected.

# Installation and build options

[Back to README](../README.md)

Build and install the extension against your environment's PyTorch, with CPU
or CPU/CUDA support.

## Prerequisites

Requires Python 3.10+, PyTorch >=2.5,<3, and a C++17 compiler. CUDA builds also
require CUDA-enabled PyTorch and a matching CUDA toolkit. MPS is unsupported.
CI covers Python 3.11 / PyTorch 2.5.1 on Linux and macOS; Windows and other
PyTorch versions are not covered by the checked-in CI matrix.

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
python -c 'from torch_deform_attn import _C; print(_C.cpu_parallel_backend)'
```

`torch.set_num_threads(...)` controls enabled CPU parallelism. The build diagnostic
reports `openmp`, `native`, or `serial`; PyTorch's own configuration alone does
not establish whether an extension was compiled with OpenMP.

## macOS architecture

Extensions and wheels target the running Python/PyTorch architecture, including
with a universal2 Python. Cross-architecture `ARCHFLAGS` are rejected.

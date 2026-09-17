# Support matrix

[Back to README](../README.md)

Use these tables to choose a validated Python/PyTorch combination. Results apply
to the recorded source revision and environment, not every release of this package.

Compatibility on Linux x86_64 with standard CPython (GIL enabled).

- ✅ Verified with this library.
- ⚠️ Tested, but the full suite failed; see the validation record for limitations.
- ➖ Within PyTorch's Python compatibility range, but unverified with this library.
- ❌ Outside PyTorch's Python compatibility range.

An asterisk (*) marks experimental Python support in PyTorch. The icon records
this library's validation status independently. Upstream compatibility follows
[PyTorch's release matrix](https://github.com/pytorch/pytorch/blob/v2.14.0/RELEASE.md#release-compatibility-matrix)
and [2.14.0 package metadata](https://pypi.org/project/torch/2.14.0/).

## CPU

| PyTorch | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 | Python 3.14 |
| --- | :---: | :---: | :---: | :---: | :---: |
| 2.4.0 | ✅ | ✅ | ✅ | ❌ | ❌ |
| 2.5.0 | ✅ | ✅ | ✅ | ⚠️* | ❌ |
| 2.5.1 | ✅ | ✅ | ✅ | ⚠️* | ❌ |
| 2.7.1 | ✅ | ✅ | ✅ | ✅ | ❌ |
| 2.8.0 | ✅ | ✅ | ✅ | ✅ | ❌ |
| 2.9.1 | ✅ | ✅ | ✅ | ✅ | ⚠️* |
| 2.10.0 | ✅ | ✅ | ✅ | ✅ | ✅* |
| 2.11.0 | ✅ | ✅ | ✅ | ✅ | ✅* |
| 2.12.1 | ✅ | ✅ | ✅ | ✅ | ✅* |
| 2.13.0 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 2.14.0 | ✅ | ✅ | ✅ | ✅ | ✅ |

## GPU (CUDA)

| PyTorch | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 | Python 3.14 |
| --- | :---: | :---: | :---: | :---: | :---: |
| 2.4.0 | ✅ | ✅ | ✅ | ❌ | ❌ |
| 2.5.0 | ✅ | ✅ | ✅ | ⚠️* | ❌ |
| 2.5.1 | ✅ | ✅ | ✅ | ⚠️* | ❌ |
| 2.7.1 | ✅ | ✅ | ✅ | ✅ | ❌ |
| 2.8.0 | ✅ | ✅ | ✅ | ✅ | ❌ |
| 2.9.1 | ✅ | ✅ | ✅ | ✅ | ⚠️* |
| 2.10.0 | ✅ | ✅ | ✅ | ✅ | ✅* |
| 2.11.0 | ✅ | ✅ | ✅ | ✅ | ✅* |
| 2.12.1 | ✅ | ✅ | ✅ | ✅ | ✅* |
| 2.13.0 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 2.14.0 | ✅ | ✅ | ✅ | ✅ | ✅ |

CUDA validation selects a CUDA build from
[PyTorch's official version-specific builds](https://pytorch.org/get-started/previous-versions/)
and uses a matching toolkit. Each ✅ covers that tested pair. The expanded Python
matrix uses CUDA 12.4 for 2.4.0/2.5.0/2.5.1 and CUDA 12.6 for the listed
2.7–2.14 releases.

PyTorch 2.5.0/2.5.1 on Python 3.13 and 2.9.1 on Python 3.14 reject
`torch.compile` upstream. Their full suites fail; build and test results are in the
[Python-version validation](validation/python-matrix/README.md).

## Scope

Unverified combinations within PyTorch's compatibility range, other CUDA
toolkits, macOS and Windows are best effort. Combinations marked ❌ are excluded.
This library requires Python >=3.10 even where PyTorch supports older Python.
The dependency range permits more versions than this tested matrix.
Release candidate `0.1.0rc3` adds an Apple Silicon MPS backend. The existing Linux tables
do not imply MPS coverage: see the separate [MPS validation record](validation/mps/README.md).
MPS requires macOS 13.3+ (14+ for bfloat16); float64, empty dimensions, and MPS
`torch.compile` support are excluded.

The tables combine current-source validation and historical release records.
The published `0.1.0rc3` requires PyTorch >=2.4. Results are tied to source revisions:
see the [Python-matrix evidence](validation/python-matrix/README.md),
[2.4.0 evidence](validation/minimum-pytorch/README.md),
[2.8–2.14 evidence](validation/pytorch-versions/README.md), and
[earlier validation records](#historical-validation).

## Historical validation

These records preserve the environments used before the expanded Python matrix.
For test counts, source identities, and logs, follow the evidence links.

| Platform | Python | PyTorch | Backend/toolkit | Build / test evidence |
| --- | --- | --- | --- | --- |
| Linux | 3.10 | 2.4.0 | CPU serial (current source lower bounds) | [Installed-wheel tests passed](validation/minimum-pytorch/README.md) |
| Linux | 3.11 | 2.4.0 | CPU OpenMP | [Installed-wheel tests passed](validation/minimum-pytorch/README.md) |
| Linux | 3.11 | 2.4.0 | CUDA 12.4 | [L4 full suite and all four sanitizers passed](validation/minimum-pytorch/README.md) |
| Linux | 3.10 | 2.5.0 | CPU serial (published RC lower bounds) | RC CPU package job passed |
| Linux | 3.11 | 2.5.1 | CPU OpenMP | RC CPU package job passed |
| Linux | 3.12 | 2.7.1 | CPU OpenMP | RC CPU package job passed |
| Linux | 3.12 | 2.8.0 | CPU OpenMP | [Installed-wheel tests passed](validation/pytorch-versions/README.md) |
| Linux | 3.12 | 2.9.1 | CPU OpenMP | [Installed-wheel tests passed](validation/pytorch-versions/README.md) |
| Linux | 3.12 | 2.10.0 | CPU OpenMP | [Installed-wheel tests passed](validation/pytorch-versions/README.md) |
| Linux | 3.12 | 2.11.0 | CPU OpenMP | [Installed-wheel tests passed](validation/pytorch-versions/README.md) |
| Linux | 3.12 | 2.12.1 | CPU OpenMP | [Installed-wheel tests passed](validation/pytorch-versions/README.md) |
| Linux | 3.12 | 2.13.0 | CPU OpenMP | [Installed-wheel tests passed](validation/pytorch-versions/README.md) |
| Linux | 3.12 | 2.14.0 | CPU OpenMP | [Installed-wheel tests passed](validation/pytorch-versions/README.md) |
| Linux | 3.11 | 2.5.1 | CUDA 12.4 | RC L4 full suite and all four sanitizers passed |
| Linux | 3.11 | 2.7.1 | CUDA 12.6 | Local L4 sdist-to-wheel build, full suite and all four sanitizers passed |
| Linux | 3.11 | 2.8.0 | CUDA 12.6 | [L4 full suite and all four sanitizers passed](validation/pytorch-versions/README.md) |
| Linux | 3.12 | 2.14.0 | CUDA 12.6 | [L4 full suite and all four sanitizers passed](validation/pytorch-versions/README.md) |

The rows above preserve the earlier per-version validation records. The initial
2.8–2.14 validation selected 2.8.0 and 2.14.0 for CUDA runtime checks; its exact
source identities and sanitizer reports remain in the
[newer-PyTorch validation record](validation/pytorch-versions/README.md).

The CUDA 12.6 local result above refers only to source
`c29ca3640b8a7a0cf4fc9907d40ccde4d42f9045`; see the
[full logs and environment records](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/fedcad0064f3cea79146be9ca181c97f53aeeda8/docs/validation/2026-09-14-p1/README.md).
Historical macOS runs remain in the evidence archive as supplemental results;
they do not add macOS to the support matrix. The RC results are linked below.

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

# Kernel correctness and sanitizers

## Objective and conditions

Validate installed-wheel forward/backward correctness, reduction coverage and
CUDA memory/synchronization checks on PyTorch 2.5.1 / CUDA 12.4 and
PyTorch 2.7.1 / CUDA 12.6, with macOS CPU coverage across Python 3.10–3.12.
Each GPU/toolchain pair uses one sdist-to-wheel build for the full suite and
all four Compute Sanitizer tools, controlled locally.

Tested source: [`c29ca3640b8a7a0cf4fc9907d40ccde4d42f9045`](https://github.com/S-aiueo32/torch-ms-deform-attn/tree/c29ca3640b8a7a0cf4fc9907d40ccde4d42f9045),
retained by tag `validation/p1-2026-09-14`. Execution date: 2026-09-14.

## Results

| Environment | Build / tests |
| --- | --- |
| macOS arm64, Python 3.10.21, PyTorch 2.5.0 | CPU OpenMP; 54 tests, 35 passed, 19 GPU skips |
| macOS arm64, Python 3.10.21, PyTorch 2.5.0 → 2.7.1 | Same-environment extension rebuild; 54 tests, 35 passed, 19 GPU skips |
| macOS arm64, Python 3.11.16, PyTorch 2.5.1 | CPU OpenMP; 54 tests, 35 passed, 19 GPU skips |
| macOS arm64, Python 3.12.14, PyTorch 2.7.1 | sdist-to-wheel build, installed-wheel tests outside checkout; 54 tests, 35 passed, 19 GPU skips |
| Linux L4, PyTorch 2.5.1 / CUDA 12.4 | 54 tests, 52 passed, 2 expected skips; all four sanitizers passed |
| Linux L4, PyTorch 2.7.1 / CUDA 12.6 | 54 tests, 52 passed, 2 expected skips; all four sanitizers passed |

The CUDA skips are the CPU-only-wheel-on-GPU and two-GPU tests. Each sanitizer
runs four positive test methods, including reduction dtype/channel combinations
and padding, offset and collision cases. Deliberate device assertions execute
in subprocesses in the normal suite. Sanitizer logs report zero errors;
racecheck also reports zero warnings/hazards.

## Evidence and scope

- CUDA 12.4: [report](cu124/cuda-tests.json), [log](cu124/cuda-checks.log), [cleanup](cu124/cleanup.json), [checksums](cu124/sha256.json).
- CUDA 12.6: [report](cu126/cuda-tests.json), [log](cu126/cuda-checks.log), [cleanup](cu126/cleanup.json), [checksums](cu126/sha256.json).
- Root CPU logs record lower-bound environments and the cross-version rebuild.
  Controller unit tests: 39 passed; Ruff and ty passed.

These runs cover the stated source and test methods. See
[sanitizer scope](../../gpu-runner.md#sanitizer-evidence-and-scope) and the
separate [two-GPU validation](../execution-and-benchmarks/README.md).
The manifests include checksums for uncommitted wheels and source archives.
Both L4 Pods were verified deleted; each was quoted at $0.49/hour.

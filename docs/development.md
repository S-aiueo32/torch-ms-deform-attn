# Development and CI

[Back to README](../README.md)

Run correctness tests against an installed package and build distributions for
validation in CI. Install the [build prerequisites](installation.md) first.

## Run tests

After installing the package, run from the repository root:

```bash
python -m unittest discover -s tests -v
```

Tests compare forward values and all three gradients against the PyTorch
reference, run finite-difference gradcheck, and cover noncontiguous inputs,
boundary/out-of-bounds sampling, overlapping level offsets, colliding samples,
empty CPU dimensions, and invalid inputs. Integration tests cover AMP,
FakeTensor, `torch.library.opcheck`, and full-graph compilation.

CUDA tests run only when the installed extension has CUDA support and a GPU is
available. They cover channel-reduction paths, nondefault streams, partial batch
chunks, AMP, compilation, and multiple devices when available. A CPU-only test
run does not validate CUDA execution.

## Workflow coverage

| Workflow | Coverage |
| --- | --- |
| [CPU package](../.github/workflows/ci.yml) | Linux serial/OpenMP and macOS OpenMP; Python 3.11 / PyTorch 2.5.1 |
| [CUDA package build](../.github/workflows/cuda-build.yml) | Pinned PyTorch 2.5.1 / CUDA 12.4 image; compilation and host-side tests without a GPU |
| [CUDA correctness](../.github/workflows/cuda.yml) | Manual Runpod GPU run; installed-wheel tests and optional Compute Sanitizer |
| [Runpod controller checks](../.github/workflows/runpod-checks.yml) | Controller unit tests, shell syntax, and bootstrap checks in a CPU container |
| [Runpod cleanup](../.github/workflows/runpod-cleanup.yml) | Opt-in recovery after CUDA runs and hourly cleanup |

Package workflows build an sdist, build a wheel from it, and test the installed
wheel outside the checkout. CPU worker-count tests check actual extension
parallelism; collision tests compare gradients across thread counts. Indexing
tests check offsets above 2^31 and overflow rejection without huge allocations.

See [GPU runner setup and the recorded L4 validation](gpu-runner.md) for manual
dispatch, sanitizer options, artifacts, and cleanup configuration. GPU runtime
checks are separate from the GPU-free CUDA package build.

The controller tests can also run locally without renting a GPU:

```bash
python3 -m unittest discover -s scripts/tests -v
bash -n scripts/runpod_bootstrap.sh scripts/run_cuda_checks.sh scripts/test_runpod_bootstrap.sh
# Requires Docker:
bash scripts/test_runpod_bootstrap.sh
```

## Build distributions

Install the [build prerequisites](installation.md), then:

```bash
python -m pip install build
python -m build --no-isolation
```

Build against the target environment's PyTorch. Locally built wheels are not
guaranteed to work across PyTorch versions.

For performance measurements, see [benchmarks](benchmarks.md).

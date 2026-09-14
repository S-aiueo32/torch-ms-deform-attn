# Development and CI

[Back to README](../README.md)

## Set up with uv

Install [uv](https://docs.astral.sh/uv/) and a C++17 compiler, then run from the
repository root:

```bash
uv sync --locked
```

This creates `.venv`, installs the development dependencies from `uv.lock`, and
builds an editable installation. No separate `uv venv` or shell activation is
needed. `.python-version` selects Python 3.11; the development group pins PyTorch
2.5.1 to match the validated configuration and supports Python 3.10–3.12.
The published package retains its broader Python/PyTorch requirements.

uv installs PyTorch and the build tools before compiling the extension in the
same environment. CUDA builds additionally require a matching CUDA toolkit;
see [backend selection](installation.md#select-cpu-or-cuda).

Python edits are available immediately. After C++/CUDA edits or a build-option
change, run `uv sync --locked` again. Source files and backend environment
variables are included in uv's build cache keys. To force a rebuild (for example,
after changing compilers or the CUDA toolkit):

```bash
uv sync --locked --reinstall-package torch-ms-deform-attn
```

To update dependencies, run `uv lock --upgrade`, then `uv sync --locked` and the
tests below. Commit `uv.lock` with dependency changes. Change the development
group's PyTorch pin explicitly when validating a newer version.

## Lint, format, and type checks

`uv sync --locked` includes Ruff and ty through the `lint` dependency group.
Run these checks before committing:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
```

Ruff checks Python errors, unused imports, and import ordering across the
repository. Its formatter uses a 100-character line length. Apply automatic
fixes and formatting with:

```bash
uv run --locked ruff check --fix .
uv run --locked ruff format .
```

ty checks `src/` using the installed PyTorch types and the native extension's
`_C.pyi` stub. Update that stub when changing bindings in `csrc/vision.cpp`.
The lint workflow runs all three checks with CPU PyTorch and does not compile
the native extension. C++/CUDA validation runs in the package workflows below;
shell syntax checks run in the Runpod controller workflow.

## Run tests

After installing the package, run from the repository root:

```bash
uv run --locked python -m unittest discover -s tests -v
```

Tests compare outputs and all three gradients against the PyTorch reference,
run finite-difference gradcheck, and check input validation. Integration tests
cover AMP, `torch.library.opcheck`, and full-graph compilation.

CUDA tests run only when the installed extension has CUDA support and a GPU is
available. A CPU-only test run does not validate CUDA execution.

## Workflow coverage

| Workflow | Coverage |
| --- | --- |
| [Lint](../.github/workflows/lint.yml) | Ruff lint/format across Python files; ty checks for `src/` |
| [CPU package](../.github/workflows/ci.yml) | Linux serial/OpenMP and macOS OpenMP; Python 3.11 / PyTorch 2.5.1 |
| [CUDA package build](../.github/workflows/cuda-build.yml) | Pinned PyTorch 2.5.1 / CUDA 12.4 image; compilation and host-side tests without a GPU |
| [CUDA correctness](../.github/workflows/cuda.yml) | Manual Runpod GPU run; installed-wheel tests and optional Compute Sanitizer |
| [CUDA benchmark](../.github/workflows/cuda-benchmark.yml) | Manual Runpod GPU run; installed-wheel tests and eager CUDA latency measurements |
| [Runpod controller checks](../.github/workflows/runpod-checks.yml) | Controller unit tests, shell syntax, and bootstrap checks in a CPU container |
| [Runpod cleanup](../.github/workflows/runpod-cleanup.yml) | Opt-in recovery after CUDA runs and hourly cleanup |

Package workflows build an sdist, build a wheel from it, and test the installed
wheel outside the checkout.

See [GPU runner setup and the recorded L4 validation](gpu-runner.md) for manual
dispatch, sanitizer options, artifacts, and cleanup configuration. GPU runtime
checks are separate from the GPU-free CUDA package build.

The controller tests can also run locally without renting a GPU:

```bash
uv run --locked python -m unittest discover -s scripts/tests -v
bash -n scripts/runpod_bootstrap.sh scripts/run_cuda_checks.sh scripts/test_runpod_bootstrap.sh
# Requires Docker:
bash scripts/test_runpod_bootstrap.sh
```

## Build distributions

With [uv](https://docs.astral.sh/uv/guides/package/) installed, build a source
distribution for PyPI:

```bash
uv build --sdist --no-sources --out-dir dist/pypi
```

uv installs the build dependencies declared in `pyproject.toml` in an isolated
environment. The archive includes the CPU/CUDA sources, tests, documentation,
and license files.

For a local wheel, use the development environment's PyTorch:

```bash
uv sync --locked
uv build --wheel --no-build-isolation --python .venv/bin/python
```

Locally built wheels are not guaranteed to work across PyTorch versions. Publish
the source distribution so users can compile against their installed PyTorch
and select CPU or CUDA support.

## Publish to PyPI with uv

The current release version is `0.1.0` in `pyproject.toml`. For later releases,
update that version before building and use the matching filename below.

Validate the archive without uploading it:

```bash
uv publish --dry-run dist/pypi/torch_ms_deform_attn-0.1.0.tar.gz
```

Set your PyPI API token in the `UV_PUBLISH_TOKEN` environment variable, then run:

```bash
uv publish dist/pypi/torch_ms_deform_attn-0.1.0.tar.gz
```

uv publishes to PyPI by default; no Twine or extra index configuration is needed.
The explicit archive path excludes local wheels and artifacts from older releases.
PyPI release filenames cannot be replaced with changed contents.

After publishing, install into an environment with the build prerequisites:

```bash
uv pip install --no-build-isolation torch-ms-deform-attn==0.1.0
```

For performance measurements, see [benchmarks](benchmarks.md).

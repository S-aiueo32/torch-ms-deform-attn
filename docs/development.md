# Development and CI

[Back to README](../README.md)

To use the published package, follow [Install from PyPI](installation.md#install-from-pypi).
Use this guide to build the repository, run validation, and publish releases.
For the pull-request checklist, start with [Contributing](../CONTRIBUTING.md).

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
For native changes, also run the [C++/CUDA checks](native-checks.md).
The lint workflow runs all three checks with CPU PyTorch and does not compile
the native extension. C++/CUDA validation runs in the package workflows below;
shell syntax checks run in the Runpod controller workflow.

## Run tests

After installing the package, run from the repository root:

```bash
uv run --locked pytest -v
```

Tests compare outputs and all three gradients against the PyTorch reference,
run finite-difference gradcheck, and check input validation. Integration tests
cover AMP, `torch.library.opcheck`, and full-graph compilation.

Pinned upstream module fixtures are included in normal test discovery; see
[module compatibility](compatibility.md). Build-policy and benchmark failure-path
tests run separately, from an installed development environment:

```bash
uv run --locked pytest build_tests -v
uv run --locked python scripts/validate_cpu_builds.py --output /tmp/msda-build-results
```

The second command requires an OpenMP PyTorch build and performs real compiler,
linker, load, and CPU checks across backend changes in a temporary checkout.

CUDA tests run only when the installed extension has CUDA support and a GPU is
available. A CPU-only test run does not validate CUDA execution.

### Linux CPU validation with local Docker

`scripts/run_cpu_checks.sh OUTPUT_DIR` builds an sdist and CPU-only OpenMP wheel,
runs the build-policy tests, and tests the installed wheel outside the checkout.
It requires Python, CPU PyTorch and a C++ compiler. For example:

```bash
mkdir -p build/cpu-validation
git archive HEAD | docker run --rm -i --platform linux/amd64 \
  -v "$PWD/build/cpu-validation:/results" \
  -e CUDA_CHECKS_SOURCE_SHA="$(git rev-parse HEAD)" \
  python:3.12-bookworm bash -c '
    mkdir /tmp/source && tar -xf - -C /tmp/source && cd /tmp/source &&
    python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu &&
    bash scripts/run_cpu_checks.sh /results
  '
```

Change the image's Python version and the PyTorch pin together to test another
cell. The JSON records the actual versions, source SHA, architecture, test
counts and skip reasons; the log includes build-policy results. A nonzero exit
status must not be recorded as a verified cell.

On Apple Silicon, `linux/amd64` requires x86_64 translation. The local validation
used [Colima](https://colima.run/docs/installation/) with its VZ/Rosetta option.
Record this as emulated Linux x86_64 correctness evidence, not native speed or
CUDA evidence. CUDA execution still requires an NVIDIA GPU; the
[local Runpod controller](gpu-runner.md#validate-several-python-versions-on-one-pod)
can test several pairs on one disposable host without GitHub Actions.

## Workflow coverage

| Workflow | Coverage |
| --- | --- |
| [Lint](../.github/workflows/lint.yml) | Ruff lint/format across Python files; ty checks for `src/` |
| [Publish to PyPI / TestPyPI](../.github/workflows/publish.yml) | Release tag and GPU evidence checks; sdist build and Trusted Publishing; manual build-only validation or TestPyPI upload |
| [CPU package](../.github/workflows/ci.yml) | PRs: three minimum/latest CPU configurations; manual `full_matrix=true`: all 12 configurations and rebuild checks |
| [CUDA package build](../.github/workflows/cuda-build.yml) | Manual only: 2.4/12.4 and 2.14/12.6 by default; `full_matrix=true` adds 2.5/12.4, 2.7/12.6 and 2.8/12.6 |
| [CUDA GPU](../.github/workflows/cuda.yml) | Manual Runpod GPU run; access check, installed-wheel tests, optional Compute Sanitizer, benchmarks, or Kernel Hub validation |
| [Runpod controller checks](../.github/workflows/runpod-checks.yml) | Controller unit tests, shell syntax, and bootstrap checks in a CPU container |
| [Runpod cleanup](../.github/workflows/runpod-cleanup.yml) | Opt-in recovery after CUDA runs and hourly cleanup |

Package workflows build an sdist, build a wheel from it, and test the installed
wheel outside the checkout. `scripts/cuda_matrix.py` is the shared source for
the CUDA build containers and the Runpod controller's PyTorch/toolkit mapping.
The CUDA entry workflow delegates GPU jobs to `reusable-cuda-gpu.yml`; local
composite actions own the Runpod lifecycle and containerized package build.

### Actions usage

Automatic validation runs on relevant pull requests, with no duplicate push
run. CPU PR checks cover Python/PyTorch 3.10/2.4.0 (serial), 3.11/2.4.0
(OpenMP), and 3.12/2.14.0 (OpenMP). Lint and controller checks also use path
filters. New commits cancel obsolete runs of these workflows.

Documentation-only PRs do not trigger these checks. GitHub evaluates PR path
filters against the full PR diff, so a documentation update within an existing
code PR can still trigger the reduced checks.

Run the broader compatibility and rebuild checks when preparing a release or
changing supported versions:

```bash
gh workflow run ci.yml --ref BRANCH --field full_matrix=true
gh workflow run cuda-build.yml --ref BRANCH --field full_matrix=true
```

Without `full_matrix=true`, a manual CPU run uses the three PR configurations,
and a manual CUDA build uses the minimum and latest configurations only.
CUDA runtime and benchmark tasks are also manual. GPU cleanup retains its
completion-triggered and hourly recovery schedule.

See [GPU runner setup and the recorded L4 validation](gpu-runner.md) for manual
dispatch, sanitizer options, artifacts, and cleanup configuration. GPU runtime
checks are separate from the GPU-free CUDA package build.

The controller tests can also run locally without renting a GPU:

```bash
uv run --locked pytest scripts/tests -v
bash -n scripts/runpod_bootstrap.sh scripts/run_cuda_checks.sh scripts/test_runpod_bootstrap.sh
# Requires Docker:
bash scripts/test_runpod_bootstrap.sh
```

## Build distributions

With [uv](https://docs.astral.sh/uv/guides/package/) installed, build a source
distribution for PyPI:

```bash
uv build --sdist --no-sources --no-config --out-dir dist/pypi
```

`--no-config` bypasses the project's editable-build isolation override, so uv
installs the build dependencies declared in `pyproject.toml` in an isolated
environment. The CI publisher instead installs CPU PyTorch and the build tools
explicitly, pins the README's repository links to the checked-out commit with
`scripts/prepare_pypi_readme.py`, and builds with `--no-build-isolation`.
The archive includes the CPU/CUDA/Metal sources, tests, documentation,
and license files.

For a local wheel, use the development environment's PyTorch:

```bash
uv sync --locked
uv build --wheel --no-build-isolation --python .venv/bin/python
```

Locally built wheels are not guaranteed to work across PyTorch versions. Publish
the source distribution so users can compile against their installed PyTorch
and select CPU or CUDA support.

## Publish to PyPI / TestPyPI with GitHub Actions

The [publish workflow](../.github/workflows/publish.yml) runs when a GitHub release
is published, including prereleases. It requires a tag of `vVERSION` matching
`project.version` in `pyproject.toml`, verifies the attached GPU evidence against
the checked-out release commit, builds and validates an sdist, and publishes it.
The publish job downloads that archive from the build job and uses
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/);
no API token secret is needed. Only that job receives `id-token: write`.

One-time setup:

1. Create the GitHub Actions environment `pypi`. Configure its deployment rules
   for release tags and any required reviewers.
2. On PyPI, add a Trusted Publisher for `torch-ms-deform-attn` with this GitHub
   repository's owner/name, workflow filename `publish.yml`, and environment
   name `pypi`. For the first release, create a pending publisher.

For each release, update the package version and commit the changes. Complete
the [required GPU release validation](gpu-runner.md#required-release-validation)
for that exact commit. Create the matching `vVERSION` tag and a draft GitHub
release, attach the four required evidence files and add the validation details
to its notes, then publish the release. Missing or mismatched evidence fails the
workflow before upload. The evidence verifier checks contents; maintainers must
still ensure the assets came from a trusted validation run.

Increment the version for each candidate and mark its GitHub release as a
prerelease. Use a version without an RC suffix for a final release. Keep `uv.lock` in sync with
`uv lock`; uploaded filenames cannot be reused for changed archives.

Use **Actions → Publish to PyPI / TestPyPI → Run workflow** with a target:

| Trigger / target | Result |
| --- | --- |
| Manual / `build-only` (default) | Build and validate; no upload |
| Manual / `testpypi` | Build, validate and upload to TestPyPI |
| Published GitHub release, including RCs | Verify release tag/GPU evidence and upload to PyPI |

Manual runs do not require a release tag or GPU evidence and save the archive
as the `pypi-sdist` artifact. They do not validate GPU behavior. TestPyPI is for
packaging tests; PyPI releases still require the GPU evidence above.

### TestPyPI setup and installation

Create the GitHub environment `testpypi`, allowing the `main` branch and `v*`
tags. On [TestPyPI](https://test.pypi.org/manage/account/publishing/), register a
separate Trusted Publisher (or pending publisher for a new project) with the
same project, owner and repository, workflow `publish.yml`, and environment
`testpypi`. PyPI's publisher registration does not configure TestPyPI.
The workflow uses `https://test.pypi.org/legacy/` for uploads.

After registering the publisher, run:

```bash
gh workflow run publish.yml --ref main --field target=testpypi
```

Follow [Install from TestPyPI](installation.md#install-from-testpypi) to test
the uploaded archive in a fresh environment with your chosen PyTorch build.

Uploaded filenames cannot be reused for changed archives. Increment the RC
version before uploading another candidate to the same index.

## Publish to PyPI manually with uv

Before publishing, complete the [required GPU release validation](gpu-runner.md#required-release-validation) for the exact source SHA and archive the evidence with the GitHub release. A CPU or CUDA build-only CI success is insufficient.

The current release version is `0.1.0` in `pyproject.toml`. For later releases,
update that version before building and use the matching filename below.

For a manual upload, build from a clean release checkout and pin the packaged
README links to that checkout before running the sdist build command above:

```bash
python scripts/prepare_pypi_readme.py \
  --repository S-aiueo32/torch-ms-deform-attn --revision "$(git rev-parse HEAD)"
uv build --sdist --no-sources --no-config --out-dir dist/pypi
git restore README.md
```

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

After publishing, follow [Install from PyPI](installation.md#install-from-pypi)
and [verify the installation](installation.md#verify-the-installation).

For performance measurements, see [benchmarks](benchmarks.md).

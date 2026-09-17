# Contributing

Use this checklist to prepare changes to Python or C++/CUDA code for review.
Run commands from the repository root. You need uv, Python 3.10–3.12, and a C++17
compiler for the locked PyTorch 2.5.1 environment. CUDA checks also need a matching
toolkit and an NVIDIA GPU.

## Set up and build

```bash
uv sync --locked
```

This creates `.venv` and builds the editable extension. After C++/CUDA edits,
run the command again to rebuild. See [development setup](docs/development.md#set-up-with-uv)
for forced rebuilds and [backend selection](docs/installation.md#select-cpu-or-cuda)
for CPU/CUDA options.

## Check your change

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
uv run --locked pytest -v
```

Run the additional checks relevant to the change:

| Changed area | Checks |
| --- | --- |
| C++/CUDA/Metal sources | [Native formatting and static checks](docs/native-checks.md), plus the relevant backend tests |
| Build policy or benchmark harness | `uv run --locked pytest build_tests -v` |
| Runpod controller or evidence handling | `uv run --locked pytest scripts/tests -v` |
| CPU backend selection | [Real build validation](docs/development.md#run-tests) |
| CUDA behavior | [Installed-wheel GPU checks](docs/gpu-runner.md) |

A CPU-only run skips GPU tests and does not validate CUDA execution. See
[workflow coverage](docs/development.md#workflow-coverage) for CI behavior.

For Metal changes, build on Apple Silicon with `FORCE_MPS=1` and run
`uv run --locked pytest tests/test_mps.py -v` with an accessible GPU.
Also run the full suite to cover CPU and FakeTensor regressions. A build-only
result or skipped MPS tests do not validate Metal execution. Record PyTorch/macOS
versions and benchmark results using `benchmarks/benchmark_mps.py`.

## Open a pull request

Describe the problem, resulting behavior, and checks you ran. State any untested
platform or skipped GPU checks that affect the change. Update the API documentation
and `src/torch_ms_deform_attn/_C.pyi` when changing native bindings; commit `uv.lock`
when changing dependencies.

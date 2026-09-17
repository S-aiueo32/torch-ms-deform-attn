# Python/PyTorch matrix validation — 2026-09-17

Source: `12cd60774870b2849628cd24e9705ef68348d91f`.
The expansion resolves all 79 previously unverified cells in the README.
Combining these runs with the earlier records, each backend has 46 verified
pairs and three pairs with known limitations among 49 in-range targets.
The six out-of-range cells per backend remain excluded. All seven Pods were
deleted and their deletion was verified by the controller.

## Local Docker CPU

| PyTorch | Python | Result | Evidence |
| --- | --- | --- | --- |
| 2.4.0+cpu | 3.12.14 | 40 passed, 26 expected GPU skips; 7 build-policy tests passed | [report](local-cpu-2.4.0-python-3.12/cpu-tests.json), [log](local-cpu-2.4.0-python-3.12/cpu-checks.log) |

The installed CPU-only OpenMP wheel was built from an sdist and tested outside
the source tree. The container is Linux x86_64, running through Rosetta on an
Apple Silicon Mac using Colima (`msda-validation`, 4 vCPUs, 8 GiB RAM).
Image: `python:3.12-bookworm` at
`sha256:581429e3df12d76e6af4be5ab7d0e7fc2013eb57dc23d2de691411c8efdbb970`.
This is emulated x86_64 correctness evidence, not native performance evidence.
The report records the actual Python, PyTorch, platform and source revision.

Python 3.13 / PyTorch 2.5.0 and 2.5.1 both build, but their full CPU suites
fail with `RuntimeError: Dynamo is not supported on Python 3.13+`. Each run
reports 66 tests, 9 errors (including subtest errors), and 26 expected GPU skips.
These combinations are marked ⚠️ rather than fully verified.
See the [2.5.0 report](local-cpu-2.5.0-python-3.13/cpu-tests.json),
[2.5.0 log](local-cpu-2.5.0-python-3.13/cpu-checks.log),
[2.5.1 report](local-cpu-2.5.1-python-3.13/cpu-tests.json), and
[2.5.1 log](local-cpu-2.5.1-python-3.13/cpu-checks.log).
The container image is `python:3.13-bookworm` at
`sha256:933b46a028fd786c9c3d426ebabc237e29a15912231ea8de576e95f0e4f41a4c`.

Python 3.14 / PyTorch 2.9.1 also rejects `torch.compile`, and its dynamic
`opcheck` path fails on `typing.Union` metadata. The CPU report has 10 errors
and the CUDA report has 22, including subtest errors. The three limited pairs
are 2.5.0/3.13, 2.5.1/3.13 and 2.9.1/3.14; no other completed suite failed
after the setup corrections. Python 3.14 / 2.10.0, 2.11.0 and 2.12.1 also
passed the local CPU-only wheel suite and all seven build-policy tests.
Their image is `python:3.14-bookworm` at
`sha256:ecac9e212daacda8a702eae372fceebc0ee36f5805abe087880367e8d061fa5b`.
The local validation containers were removed and the Colima VM was stopped;
the installed Docker/Colima tools and images remain available.

<!-- local-results -->

### Additional local CPU results

| PyTorch | Python | Result | Evidence |
| --- | --- | --- | --- |
| 2.5.0 | 3.13.15 | 9 errors, 0 failures, 26 skipped | [report](local-cpu-2.5.0-python-3.13/cpu-tests.json), [log](local-cpu-2.5.0-python-3.13/cpu-checks.log) |
| 2.5.1 | 3.13.15 | 9 errors, 0 failures, 26 skipped | [report](local-cpu-2.5.1-python-3.13/cpu-tests.json), [log](local-cpu-2.5.1-python-3.13/cpu-checks.log) |
| 2.10.0 | 3.14.7 | 40 passed, 26 skipped | [report](local-cpu-2.10.0-python-3.14/cpu-tests.json), [log](local-cpu-2.10.0-python-3.14/cpu-checks.log) |
| 2.11.0 | 3.14.7 | 40 passed, 26 skipped | [report](local-cpu-2.11.0-python-3.14/cpu-tests.json), [log](local-cpu-2.11.0-python-3.14/cpu-checks.log) |
| 2.12.1 | 3.14.7 | 40 passed, 26 skipped | [report](local-cpu-2.12.1-python-3.14/cpu-tests.json), [log](local-cpu-2.12.1-python-3.14/cpu-checks.log) |
| 2.9.1 | 3.14.7 | 10 errors, 0 failures, 26 skipped | [report](local-cpu-2.9.1-python-3.14/cpu-tests.json), [log](local-cpu-2.9.1-python-3.14/cpu-checks.log) |

<!-- /local-results -->

## Remote CPU/CUDA

44 previously unverified CUDA combinations were attempted across five
disposable L4 Pods using the local controller. Each successful CUDA suite was
followed by a fresh CPU-only wheel build and its full suite. Each Pod is quoted
at $0.49/hour and has a 60-minute controller deadline. The combined full-deadline
compute estimate is $2.45, excluding disk charges; this is not a billing statement.
Two additional Python 3.10 shards use 20- and 40-minute deadlines to rerun
setup failures with current pip and managed Python. The combined full-deadline
compute estimate for all seven Pods is $2.94, excluding disk charges.
Their source is `e0a6c074a0a12d06427f9ca7db3965a98ab45128`; only the validation
harness differs from the initial source. Package code and tests are unchanged.
All 55 attempts are archived below. The five initial controllers returned
nonzero because their matrices included setup failures and/or the known Python
limitations. The two corrected Python 3.10 controllers passed all their cases.
Each verified cell has its own successful full-suite reports and completion
marker; an aggregate controller failure is not presented as an all-green run.

The first 2.4.0 / Python 3.10 attempt selected the image's system Python, which
lacked `Python.h`; it did not establish a library incompatibility. Managed
CPython installations with development headers were provisioned on the host
for subsequent cases. The harness now explicitly requires managed Python.
The remaining initial Python 3.10 setup failures came from its bundled old pip:
it rejected the normalized `typing_extensions` wheel name and then could not
resolve the sdist build dependency `flit_core` from the PyTorch-only index.
Upgrading pip inside each fresh environment fixed all eleven Python 3.10
retries. These were harness setup failures, not operator incompatibilities.

<!-- live-results -->

## Completed remote attempts

Successful rows have both full-suite reports and a completion marker.
“Not reached” includes dependency/build failures; it is not a failed library test.
Each linked directory contains the reports, full log and SHA-256 checksums.

| Shard | PyTorch | Python | CPU-only wheel | CUDA wheel | Evidence |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.4.0 | 3.10 | not reached | not reached | [log](remote-1/torch-2.4.0-python-3.10/driver.log) |
| 1 | 2.4.0 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-1/torch-2.4.0-python-3.12/driver.log) |
| 1 | 2.5.0 | 3.10 | not reached | not reached | [log](remote-1/torch-2.5.0-python-3.10/driver.log) |
| 1 | 2.5.0 | 3.11 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-1/torch-2.5.0-python-3.11/driver.log) |
| 1 | 2.5.0 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-1/torch-2.5.0-python-3.12/driver.log) |
| 1 | 2.5.0 | 3.13 | not reached | 66 tests, 21 errors, 0 failures | [log](remote-1/torch-2.5.0-python-3.13/driver.log) |
| 1 | 2.5.1 | 3.10 | not reached | not reached | [log](remote-1/torch-2.5.1-python-3.10/driver.log) |
| 1 | 2.5.1 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-1/torch-2.5.1-python-3.12/driver.log) |
| 1 | 2.5.1 | 3.13 | not reached | 66 tests, 21 errors, 0 failures | [log](remote-1/torch-2.5.1-python-3.13/driver.log) |
| 2 | 2.7.1 | 3.10 | not reached | not reached | [log](remote-2/torch-2.7.1-python-3.10/driver.log) |
| 2 | 2.7.1 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-2/torch-2.7.1-python-3.12/driver.log) |
| 2 | 2.7.1 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-2/torch-2.7.1-python-3.13/driver.log) |
| 2 | 2.8.0 | 3.10 | not reached | not reached | [log](remote-2/torch-2.8.0-python-3.10/driver.log) |
| 2 | 2.8.0 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-2/torch-2.8.0-python-3.12/driver.log) |
| 2 | 2.8.0 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-2/torch-2.8.0-python-3.13/driver.log) |
| 2 | 2.9.1 | 3.10 | not reached | not reached | [log](remote-2/torch-2.9.1-python-3.10/driver.log) |
| 2 | 2.9.1 | 3.11 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-2/torch-2.9.1-python-3.11/driver.log) |
| 2 | 2.9.1 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-2/torch-2.9.1-python-3.12/driver.log) |
| 3 | 2.9.1 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-3/torch-2.9.1-python-3.13/driver.log) |
| 3 | 2.9.1 | 3.14 | not reached | 66 tests, 22 errors, 0 failures | [log](remote-3/torch-2.9.1-python-3.14/driver.log) |
| 3 | 2.10.0 | 3.10 | not reached | not reached | [log](remote-3/torch-2.10.0-python-3.10/driver.log) |
| 3 | 2.10.0 | 3.11 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-3/torch-2.10.0-python-3.11/driver.log) |
| 3 | 2.10.0 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-3/torch-2.10.0-python-3.12/driver.log) |
| 3 | 2.10.0 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-3/torch-2.10.0-python-3.13/driver.log) |
| 3 | 2.10.0 | 3.14 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-3/torch-2.10.0-python-3.14/driver.log) |
| 3 | 2.11.0 | 3.10 | not reached | not reached | [log](remote-3/torch-2.11.0-python-3.10/driver.log) |
| 3 | 2.11.0 | 3.11 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-3/torch-2.11.0-python-3.11/driver.log) |
| 4 | 2.11.0 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-4/torch-2.11.0-python-3.12/driver.log) |
| 4 | 2.11.0 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-4/torch-2.11.0-python-3.13/driver.log) |
| 4 | 2.11.0 | 3.14 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-4/torch-2.11.0-python-3.14/driver.log) |
| 4 | 2.12.1 | 3.10 | not reached | not reached | [log](remote-4/torch-2.12.1-python-3.10/driver.log) |
| 4 | 2.12.1 | 3.11 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-4/torch-2.12.1-python-3.11/driver.log) |
| 4 | 2.12.1 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-4/torch-2.12.1-python-3.12/driver.log) |
| 4 | 2.12.1 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-4/torch-2.12.1-python-3.13/driver.log) |
| 4 | 2.12.1 | 3.14 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-4/torch-2.12.1-python-3.14/driver.log) |
| 5 | 2.13.0 | 3.10 | not reached | not reached | [log](remote-5/torch-2.13.0-python-3.10/driver.log) |
| 5 | 2.13.0 | 3.11 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-5/torch-2.13.0-python-3.11/driver.log) |
| 5 | 2.13.0 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-5/torch-2.13.0-python-3.12/driver.log) |
| 5 | 2.13.0 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-5/torch-2.13.0-python-3.13/driver.log) |
| 5 | 2.13.0 | 3.14 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-5/torch-2.13.0-python-3.14/driver.log) |
| 5 | 2.14.0 | 3.10 | not reached | not reached | [log](remote-5/torch-2.14.0-python-3.10/driver.log) |
| 5 | 2.14.0 | 3.11 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-5/torch-2.14.0-python-3.11/driver.log) |
| 5 | 2.14.0 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-5/torch-2.14.0-python-3.13/driver.log) |
| 5 | 2.14.0 | 3.14 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-5/torch-2.14.0-python-3.14/driver.log) |
| 6 | 2.4.0 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-6/torch-2.4.0-python-3.10/driver.log) |
| 6 | 2.5.0 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-6/torch-2.5.0-python-3.10/driver.log) |
| 6 | 2.5.1 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-6/torch-2.5.1-python-3.10/driver.log) |
| 7 | 2.7.1 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.7.1-python-3.10/driver.log) |
| 7 | 2.8.0 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.8.0-python-3.10/driver.log) |
| 7 | 2.9.1 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.9.1-python-3.10/driver.log) |
| 7 | 2.10.0 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.10.0-python-3.10/driver.log) |
| 7 | 2.11.0 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.11.0-python-3.10/driver.log) |
| 7 | 2.12.1 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.12.1-python-3.10/driver.log) |
| 7 | 2.13.0 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.13.0-python-3.10/driver.log) |
| 7 | 2.14.0 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.14.0-python-3.10/driver.log) |

## Scope, cleanup and retained artifacts

All remote rows use one NVIDIA L4 and OpenMP. PyTorch 2.4.0/2.5.0/2.5.1
use CUDA 12.4 with the controller's pinned 2.5.1 development image; the listed
2.7–2.14 versions use CUDA 12.6 with the pinned 2.14.0 development image.
Exact Python, PyTorch, toolkit and driver versions are in each JSON report.
Managed Python is standard CPython with the GIL enabled. No GitHub Actions
workflow was used for this expansion.

A successful remote CUDA suite has 64 passing tests and two expected skips:
`test_cuda.CPUOnlyBuildTest.test_cuda_input_error` and
`test_cuda.CUDAAttentionTest.test_noncurrent_device`. The first then passes
with a newly built CPU-only wheel, whose full suite has 41 passes and 25
expected CUDA-extension skips. This expansion uses `sanitizer=none` and one
GPU, so it adds neither sanitizer nor multi-GPU evidence. It is source
compatibility validation, not a release certification or a new PyPI release.

The local harness checks passed 46 tests; see [log](harness-tests.log). Ruff
checks, formatting and shell syntax checks also passed. The package's
`src/`, `csrc/`, `tests/`, `setup.py` and `pyproject.toml` are identical between
both runtime revisions and the final documentation update.

| Shard | Local run ID | Attempts | Controller exit | Cleanup record |
| --- | --- | --- | --- | --- |
| 1 | 20260917110101 | 9 | 1 | [deleted](remote-1/controller-state.json) |
| 2 | 20260917110102 | 9 | 1 | [deleted](remote-2/controller-state.json) |
| 3 | 20260917110103 | 9 | 1 | [deleted](remote-3/controller-state.json) |
| 4 | 20260917110104 | 8 | 1 | [deleted](remote-4/controller-state.json) |
| 5 | 20260917110105 | 9 | 1 | [deleted](remote-5/controller-state.json) |
| 6 | 20260917110106 | 3 | 0 | [deleted](remote-6/controller-state.json) |
| 7 | 20260917110107 | 8 | 0 | [deleted](remote-7/controller-state.json) |

Per-shard `original-artifacts-sha256.json` files record the downloaded artifact
and source-archive checksums. Raw logs, binary wheels, sdists and both source
archives are retained locally under `build/matrix-validation-2026-09-17/`
(ignored by Git). SSH private keys are excluded. The committed logs remove
ANSI escapes and trailing whitespace; JSON reports are preserved verbatim.
Each case's `sha256.json` covers the committed normalized evidence.

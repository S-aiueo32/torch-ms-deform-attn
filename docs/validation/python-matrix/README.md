# Python/PyTorch support matrix

[Validation index](../README.md) · [Support matrix](../../support.md)

## Objective and acceptance criteria

Validate CPU and CUDA installed wheels across the Python/PyTorch pairs in the
[support tables](../../support.md). Each backend has 49 in-range
pairs: **46 verified and 3 with known limitations**, with no unverified in-range
cells. Six cells per backend are outside the supported upstream Python range.

A verified pair must pass its full applicable suite, with only the documented
skips. Build success alone does not establish runtime compatibility. This record
contains 44 distinct CUDA pairs and seven local CPU cases; the remaining evidence
is in [kernel correctness](../kernel-correctness/README.md),
[minimum PyTorch](../minimum-pytorch/README.md), and
[PyTorch versions](../pytorch-versions/README.md).
Execution date: 2026-09-17.

## Test environments

Both paths build an sdist, build and install a wheel from it, and run tests
outside the checkout with OpenMP enabled.

| Path | Environment | Successful full-suite result |
| --- | --- | --- |
| Local CPU | Linux x86_64 Docker via Rosetta on Apple Silicon; Colima with 4 vCPUs / 8 GiB; CPU-only PyTorch | 40 passed, 26 GPU skips; 7 build-policy tests passed |
| Remote CUDA | One NVIDIA L4; managed standard CPython with the GIL; CUDA-enabled PyTorch | 64 passed, 2 expected skips |
| Remote CPU-only extension | Fresh CPU-only wheel on the same L4 host and CUDA-enabled PyTorch | 41 passed, 25 CUDA-extension skips; separate CPU-only-wheel GPU-input check passed |

PyTorch 2.4.0/2.5.0/2.5.1 use CUDA 12.4 and the controller's pinned 2.5.1
development image. The listed 2.7–2.14 versions use CUDA 12.6 and the pinned
2.14.0 image. Each environment uses managed Python with development headers
and current pip. Exact Python, toolkit and driver versions are in the reports.

The two CUDA-suite skips are `test_cuda.CPUOnlyBuildTest.test_cuda_input_error`
and `test_cuda.CUDAAttentionTest.test_noncurrent_device`. The CPU-only-wheel
phase separately exercises the first. This matrix uses `sanitizer=none` and
one GPU; sanitizer and multi-GPU evidence have their own records.
Local Docker results establish emulated x86_64 correctness, not native performance.

### Local container images

| Python | Image | Digest |
| --- | --- | --- |
| 3.12 | `python:3.12-bookworm` | `sha256:581429e3df12d76e6af4be5ab7d0e7fc2013eb57dc23d2de691411c8efdbb970` |
| 3.13 | `python:3.13-bookworm` | `sha256:933b46a028fd786c9c3d426ebabc237e29a15912231ea8de576e95f0e4f41a4c` |
| 3.14 | `python:3.14-bookworm` | `sha256:ecac9e212daacda8a702eae372fceebc0ee36f5805abe087880367e8d061fa5b` |

## Known limitations

| PyTorch | Python | CPU and CUDA limitation |
| --- | --- | --- |
| 2.5.0, 2.5.1 | 3.13 | Dynamo rejects Python 3.13+; CPU reports contain 9 errors and CUDA reports 21 errors |
| 2.9.1 | 3.14 | `torch.compile` rejects Python 3.14+; dynamic opcheck also fails on `typing.Union` metadata; CPU reports contain 10 errors and CUDA reports 22 errors |

Error counts include subtest errors. These wheels build, but their full suites
do not pass, so both support tables mark them ⚠️. Their CPU evidence comes from
the local runs below. All other completed suites in this record pass.

## Local CPU results

| PyTorch | Python | Result | Evidence |
| --- | --- | --- | --- |
| 2.4.0 | 3.12.14 | 40 passed, 26 skipped; 7 build-policy tests passed | [report](local-cpu-2.4.0-python-3.12/cpu-tests.json), [log](local-cpu-2.4.0-python-3.12/cpu-checks.log) |
| 2.5.0 | 3.13.15 | 9 errors, 26 skipped | [report](local-cpu-2.5.0-python-3.13/cpu-tests.json), [log](local-cpu-2.5.0-python-3.13/cpu-checks.log) |
| 2.5.1 | 3.13.15 | 9 errors, 26 skipped | [report](local-cpu-2.5.1-python-3.13/cpu-tests.json), [log](local-cpu-2.5.1-python-3.13/cpu-checks.log) |
| 2.9.1 | 3.14.7 | 10 errors, 26 skipped | [report](local-cpu-2.9.1-python-3.14/cpu-tests.json), [log](local-cpu-2.9.1-python-3.14/cpu-checks.log) |
| 2.10.0 | 3.14.7 | 40 passed, 26 skipped; 7 build-policy tests passed | [report](local-cpu-2.10.0-python-3.14/cpu-tests.json), [log](local-cpu-2.10.0-python-3.14/cpu-checks.log) |
| 2.11.0 | 3.14.7 | 40 passed, 26 skipped; 7 build-policy tests passed | [report](local-cpu-2.11.0-python-3.14/cpu-tests.json), [log](local-cpu-2.11.0-python-3.14/cpu-checks.log) |
| 2.12.1 | 3.14.7 | 40 passed, 26 skipped; 7 build-policy tests passed | [report](local-cpu-2.12.1-python-3.14/cpu-tests.json), [log](local-cpu-2.12.1-python-3.14/cpu-checks.log) |

## CPU and CUDA pair results

Each successful remote pair has `cuda-tests.json`, `cpu-tests.json`,
`cpu-only-gpu-tests.json` and `cuda-completion.json` alongside the linked log.
Each directory also contains environment details and checksums.

| PyTorch | Python | CPU-only wheel | CUDA wheel | Evidence |
| --- | --- | --- | --- | --- |
| 2.4.0 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-6/torch-2.4.0-python-3.10/driver.log) |
| 2.4.0 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-1/torch-2.4.0-python-3.12/driver.log) |
| 2.5.0 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-6/torch-2.5.0-python-3.10/driver.log) |
| 2.5.0 | 3.11 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-1/torch-2.5.0-python-3.11/driver.log) |
| 2.5.0 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-1/torch-2.5.0-python-3.12/driver.log) |
| 2.5.0 | 3.13 | [limited](local-cpu-2.5.0-python-3.13/cpu-tests.json) | 66 tests, 21 errors, 0 failures | [log](remote-1/torch-2.5.0-python-3.13/driver.log) |
| 2.5.1 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-6/torch-2.5.1-python-3.10/driver.log) |
| 2.5.1 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-1/torch-2.5.1-python-3.12/driver.log) |
| 2.5.1 | 3.13 | [limited](local-cpu-2.5.1-python-3.13/cpu-tests.json) | 66 tests, 21 errors, 0 failures | [log](remote-1/torch-2.5.1-python-3.13/driver.log) |
| 2.7.1 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.7.1-python-3.10/driver.log) |
| 2.7.1 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-2/torch-2.7.1-python-3.12/driver.log) |
| 2.7.1 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-2/torch-2.7.1-python-3.13/driver.log) |
| 2.8.0 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.8.0-python-3.10/driver.log) |
| 2.8.0 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-2/torch-2.8.0-python-3.12/driver.log) |
| 2.8.0 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-2/torch-2.8.0-python-3.13/driver.log) |
| 2.9.1 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.9.1-python-3.10/driver.log) |
| 2.9.1 | 3.11 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-2/torch-2.9.1-python-3.11/driver.log) |
| 2.9.1 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-2/torch-2.9.1-python-3.12/driver.log) |
| 2.9.1 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-3/torch-2.9.1-python-3.13/driver.log) |
| 2.9.1 | 3.14 | [limited](local-cpu-2.9.1-python-3.14/cpu-tests.json) | 66 tests, 22 errors, 0 failures | [log](remote-3/torch-2.9.1-python-3.14/driver.log) |
| 2.10.0 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.10.0-python-3.10/driver.log) |
| 2.10.0 | 3.11 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-3/torch-2.10.0-python-3.11/driver.log) |
| 2.10.0 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-3/torch-2.10.0-python-3.12/driver.log) |
| 2.10.0 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-3/torch-2.10.0-python-3.13/driver.log) |
| 2.10.0 | 3.14 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-3/torch-2.10.0-python-3.14/driver.log) |
| 2.11.0 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.11.0-python-3.10/driver.log) |
| 2.11.0 | 3.11 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-3/torch-2.11.0-python-3.11/driver.log) |
| 2.11.0 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-4/torch-2.11.0-python-3.12/driver.log) |
| 2.11.0 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-4/torch-2.11.0-python-3.13/driver.log) |
| 2.11.0 | 3.14 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-4/torch-2.11.0-python-3.14/driver.log) |
| 2.12.1 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.12.1-python-3.10/driver.log) |
| 2.12.1 | 3.11 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-4/torch-2.12.1-python-3.11/driver.log) |
| 2.12.1 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-4/torch-2.12.1-python-3.12/driver.log) |
| 2.12.1 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-4/torch-2.12.1-python-3.13/driver.log) |
| 2.12.1 | 3.14 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-4/torch-2.12.1-python-3.14/driver.log) |
| 2.13.0 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.13.0-python-3.10/driver.log) |
| 2.13.0 | 3.11 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-5/torch-2.13.0-python-3.11/driver.log) |
| 2.13.0 | 3.12 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-5/torch-2.13.0-python-3.12/driver.log) |
| 2.13.0 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-5/torch-2.13.0-python-3.13/driver.log) |
| 2.13.0 | 3.14 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-5/torch-2.13.0-python-3.14/driver.log) |
| 2.14.0 | 3.10 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-7/torch-2.14.0-python-3.10/driver.log) |
| 2.14.0 | 3.11 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-5/torch-2.14.0-python-3.11/driver.log) |
| 2.14.0 | 3.13 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-5/torch-2.14.0-python-3.13/driver.log) |
| 2.14.0 | 3.14 | 41 passed, 25 skipped | 64 passed, 2 skipped | [log](remote-5/torch-2.14.0-python-3.14/driver.log) |

## Source identity and audit records

| Evidence | Source SHA |
| --- | --- |
| Local runs and remote shards 1–5 | `12cd60774870b2849628cd24e9705ef68348d91f` |
| Remote shards 6–7 | `e0a6c074a0a12d06427f9ca7db3965a98ab45128` |

The revisions differ only in the validation harness. Package source and tests
(`src/`, `csrc/`, `tests/`, `setup.py`, `pyproject.toml`) are identical.
Offline harness checks: [46 tests passed](harness-tests.log); Ruff, formatting
and shell syntax checks passed.

The raw archive contains 55 attempts: 44 completed suites and 11 setup failures.
Setup failures supply no library verdict. Per-shard `matrix-summary.json` and
`controller.log` retain all attempts; the table above selects a completed suite
for each pair. Nonzero aggregate controller exits do not invalidate successful
individual reports or turn a limited pair into a passing one.

| Shard | Local run ID | Attempts | Controller exit | Cleanup record |
| --- | --- | --- | --- | --- |
| 1 | 20260917110101 | 9 | 1 | [deleted](remote-1/controller-state.json) |
| 2 | 20260917110102 | 9 | 1 | [deleted](remote-2/controller-state.json) |
| 3 | 20260917110103 | 9 | 1 | [deleted](remote-3/controller-state.json) |
| 4 | 20260917110104 | 8 | 1 | [deleted](remote-4/controller-state.json) |
| 5 | 20260917110105 | 9 | 1 | [deleted](remote-5/controller-state.json) |
| 6 | 20260917110106 | 3 | 0 | [deleted](remote-6/controller-state.json) |
| 7 | 20260917110107 | 8 | 0 | [deleted](remote-7/controller-state.json) |


All seven Pods were verified deleted. Each was quoted at $0.49/hour, with
deadlines of 60 minutes for shards 1–5, 20 minutes for shard 6 and 40 minutes for
shard 7. The combined full-deadline compute estimate is $2.94, excluding disk
charges; this is not a billing statement.

Each case's `sha256.json` covers committed evidence. Logs have ANSI escapes
and trailing whitespace removed; JSON reports are preserved verbatim.
Per-shard `original-artifacts-sha256.json` records downloaded artifacts and
source-archive hashes. Uncommitted raw logs, binary wheels, sdists and source
archives are retained locally under `build/matrix-validation-2026-09-17/`.

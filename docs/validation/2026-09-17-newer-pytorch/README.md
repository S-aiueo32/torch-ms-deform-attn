# PyTorch 2.8–2.14 validation — 2026-09-17

## Linux CPU

The [CPU workflow](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35169684964)
tested PR head `135e0abba22210e1176fd006d914c427b6d63bea` through GitHub's merge
commit `a918db0241b01b01cdae97b1b9c40124793f47d6` onto main
`9cd3f4b194bfe986d648ab827b81560506102b1b`.

Every row uses Linux x86_64, Python 3.12, and OpenMP. Each job builds an sdist,
builds and installs a wheel from it, and runs the full suite outside the checkout.

| PyTorch | Tests | Build-policy tests | Log |
| --- | --- | --- | --- |
| 2.8.0 | 40 passed, 26 expected GPU skips | 7 passed | [log](cpu-2.8.0.log) |
| 2.9.1 | 40 passed, 26 expected GPU skips | 7 passed | [log](cpu-2.9.1.log) |
| 2.10.0 | 40 passed, 26 expected GPU skips | 7 passed | [log](cpu-2.10.0.log) |
| 2.11.0 | 40 passed, 26 expected GPU skips | 7 passed | [log](cpu-2.11.0.log) |
| 2.12.1 | 40 passed, 26 expected GPU skips | 7 passed | [log](cpu-2.12.1.log) |
| 2.13.0 | 40 passed, 26 expected GPU skips | 7 passed | [log](cpu-2.13.0.log) |
| 2.14.0 | 40 passed, 26 expected GPU skips | 7 passed | [log](cpu-2.14.0.log) |

This covers eager/compiled forward and backward, explicit fp16/bf16 inputs,
AMP training, FakeTensor/opcheck, dynamic compilation and pinned upstream module
parity. CPU success does not establish CUDA runtime support.

## CUDA

GPU validation source: `18cda9b34c2e498e142b08d8e961e5cfda8a040d`. The later
CPU test correction changes only the Ninja fallback unit test. The package's
Python/C++/CUDA operator code is identical between the CPU and GPU runs.

PyTorch 2.8.0 / CUDA 12.6 passed on L4 with Python 3.11.13 and driver 595.91.07:
64 tests passed and 2 expected tests skipped. The skipped CPU-only-wheel test
then passed separately with a fresh CPU-only wheel. The other skip requires two
GPUs, which were not rented for this check. All four Compute Sanitizer tools
passed (four positive test methods per tool); racecheck reported no hazards or
warnings. Pod deletion was verified.

See the [2.8 GPU workflow](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35169517169),
[test report](cuda-2.8.0/cuda-tests.json), [completion record](cuda-2.8.0/cuda-completion.json),
[CPU-only wheel report](cuda-2.8.0/cpu-only-gpu-tests.json),
[test log](cuda-2.8.0/cuda-checks.log) and
[workflow/cleanup log](cuda-2.8.0/workflow.log). Per-tool sanitizer logs and
checksums are archived alongside those reports.

PyTorch 2.14.0 / CUDA 12.6 also passed on L4 with Python 3.12.3 and driver
595.91.07: 64 tests passed, the same 2 expected skips, and the separate CPU-only
wheel test passed with no skips. All four sanitizer tools passed with zero
errors; racecheck reported zero warnings/hazards. Pod deletion was verified.
See the [2.14 GPU workflow](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35169534333),
[test report](cuda-2.14.0/cuda-tests.json),
[completion record](cuda-2.14.0/cuda-completion.json),
[CPU-only wheel report](cuda-2.14.0/cpu-only-gpu-tests.json),
[test log](cuda-2.14.0/cuda-checks.log) and
[workflow/cleanup log](cuda-2.14.0/workflow.log).

Both evidence sets passed `scripts/verify_cuda_release.py` with the source SHA
above. Downloaded CUDA reports and logs are preserved verbatim and checked
against their `sha256.json` files. The per-tool logs contain the sanitizer
summaries; see [tool scope](../../gpu-runner.md#sanitizer-evidence-and-scope).
Binary wheels/sdists are retained in the Actions artifacts, not committed.
The two Pods were each quoted at $0.49/hour with a 60-minute controller deadline
and were both deleted. $0.98 is a conservative combined compute estimate if
both consumed their full deadline, excluding disk charges; it is not a billing
statement or provider-enforced spending cap.

GPU execution in this update covers 2.8.0 and 2.14.0 with CUDA 12.6. The
intermediate 2.9–2.13 rows have CPU evidence only; other CUDA versions and
two-GPU execution remain unverified for these new rows. This is source
validation, not a new package release or validation of a future release SHA.

The [GPU-free CUDA build workflow](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35170050511)
passed for 2.4.0/12.4, 2.5.1/12.4, 2.7.1/12.6, 2.8.0/12.6 and 2.14.0/12.6.
The new [2.8.0 build log](cuda-build-2.8.0.log) and
[2.14.0 build log](cuda-build-2.14.0.log) each record 40 passing installed-wheel
tests and 26 expected GPU skips. That run tested PR head
`e27705320290ade4df26495da89c6dba40a6b995`; its package and GPU harness sources
are unchanged from the GPU validation SHA above. CPU CI, controller checks and
Lint also passed at that PR head.

## Compatibility findings

PyTorch 2.8 changed the missing-Ninja fallback diagnostic from `UserWarning` to
logging. The initial CPU run failed only at the test asserting the notification
mechanism, after successfully building each wheel. The test now checks that
`use_ninja` becomes false, preserving the fallback behavior check across versions.
See the [initial CPU run](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35169503730).

PyTorch 2.13 and 2.14 require C++20. No explicit language-standard flag is added
by this package; PyTorch's `BuildExtension` supplies the required standard.
Older supported releases continue to select C++17. Compiler requirements are
documented in [installation](../../installation.md).

The recent official CUDA image uses system Python rather than the earlier
Conda environment. The GPU harness now uses the image's `python3` by default,
and the bootstrap installs `python3-venv` so a fresh isolated environment can
be created with system Python too.

The GPU-free build job intentionally installs into the disposable container's
Python. New system-Python images enforce PEP 668, so that job explicitly sets
`PIP_BREAK_SYSTEM_PACKAGES=1` inside the container. The Runpod suite continues
to use its separate venv.

The version selection is based on [official releases](https://github.com/pytorch/pytorch/releases)
and [installation pairs](https://pytorch.org/get-started/previous-versions/)
available on 2026-09-17; 2.14.0 was the latest stable release on that date.
These results cover the listed patches and environments, not future releases
or every patch/Python/toolkit combination. Job logs are normalized by removing
ANSI escapes and trailing whitespace.

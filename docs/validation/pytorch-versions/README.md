# PyTorch version compatibility

[Validation index](../README.md) · [Support matrix](../../support.md)

## Objective and conditions

Validate Linux installed-wheel compatibility across the listed PyTorch 2.8–2.14
patches, with GPU-free CUDA builds and sanitizer checks on 2.8.0 and 2.14.0.
Execution date: 2026-09-17. Python-version coverage is recorded separately in
the [Python/PyTorch matrix](../python-matrix/README.md).

## Linux CPU

The [CPU workflow](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35169684964)
tested PR head `135e0abba22210e1176fd006d914c427b6d63bea` through GitHub's merge
commit `a918db0241b01b01cdae97b1b9c40124793f47d6` onto main
`9cd3f4b194bfe986d648ab827b81560506102b1b`.

Every row uses Linux x86_64, Python 3.12, and OpenMP. Each job builds an sdist,
builds and installs a wheel from it, and runs the full suite outside the checkout.

| PyTorch | Tests | Build-policy tests | Log |
| --- | --- | --- | --- |
| 2.8.0 | 40 passed, 26 expected GPU skips | 7 passed | [log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cpu-2.8.0.log) |
| 2.9.1 | 40 passed, 26 expected GPU skips | 7 passed | [log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cpu-2.9.1.log) |
| 2.10.0 | 40 passed, 26 expected GPU skips | 7 passed | [log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cpu-2.10.0.log) |
| 2.11.0 | 40 passed, 26 expected GPU skips | 7 passed | [log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cpu-2.11.0.log) |
| 2.12.1 | 40 passed, 26 expected GPU skips | 7 passed | [log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cpu-2.12.1.log) |
| 2.13.0 | 40 passed, 26 expected GPU skips | 7 passed | [log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cpu-2.13.0.log) |
| 2.14.0 | 40 passed, 26 expected GPU skips | 7 passed | [log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cpu-2.14.0.log) |

This covers eager/compiled forward and backward, explicit fp16/bf16 inputs,
AMP training, FakeTensor/opcheck, dynamic compilation and pinned upstream module
parity. CPU success does not establish CUDA runtime support.

## CUDA

GPU validation source: `18cda9b34c2e498e142b08d8e961e5cfda8a040d`. The CPU and GPU
sources have identical Python/C++/CUDA operator code; their Ninja fallback unit
tests differ.

PyTorch 2.8.0 / CUDA 12.6 passed on L4 with Python 3.11.13 and driver 595.91.07:
64 tests passed and 2 expected tests skipped. The skipped CPU-only-wheel test
then passed separately with a fresh CPU-only wheel. The other skip requires two
GPUs, which were not rented for this check. All four Compute Sanitizer tools
passed (four positive test methods per tool); racecheck reported no hazards or
warnings. Pod deletion was verified.

See the [2.8 GPU workflow](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35169517169),
[test report](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cuda-2.8.0/cuda-tests.json), [completion record](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cuda-2.8.0/cuda-completion.json),
[CPU-only wheel report](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cuda-2.8.0/cpu-only-gpu-tests.json),
[test log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cuda-2.8.0/cuda-checks.log) and
[workflow/cleanup log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cuda-2.8.0/workflow.log). Per-tool sanitizer logs and
checksums are archived alongside those reports.

PyTorch 2.14.0 / CUDA 12.6 also passed on L4 with Python 3.12.3 and driver
595.91.07: 64 tests passed, the same 2 expected skips, and the separate CPU-only
wheel test passed with no skips. All four sanitizer tools passed with zero
errors; racecheck reported zero warnings/hazards. Pod deletion was verified.
See the [2.14 GPU workflow](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35169534333),
[test report](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cuda-2.14.0/cuda-tests.json),
[completion record](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cuda-2.14.0/cuda-completion.json),
[CPU-only wheel report](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cuda-2.14.0/cpu-only-gpu-tests.json),
[test log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cuda-2.14.0/cuda-checks.log) and
[workflow/cleanup log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cuda-2.14.0/workflow.log).

Both evidence sets passed `scripts/verify_cuda_release.py` with the source SHA
above. Downloaded CUDA reports and logs are preserved verbatim and checked
against their `sha256.json` files. The per-tool logs contain the sanitizer
summaries; see [tool scope](../../gpu-runner.md#sanitizer-evidence-and-scope).
Binary wheels/sdists are retained in the Actions artifacts, not committed.
The two Pods were each quoted at $0.49/hour with a 60-minute controller deadline
and were both deleted. $0.98 is a conservative combined compute estimate if
both consumed their full deadline, excluding disk charges; it is not a billing
statement or provider-enforced spending cap.

The [GPU-free CUDA build workflow](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35170050511)
passed for 2.4.0/12.4, 2.5.1/12.4, 2.7.1/12.6, 2.8.0/12.6 and 2.14.0/12.6.
The [2.8.0 build log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cuda-build-2.8.0.log) and
[2.14.0 build log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/pytorch-versions/cuda-build-2.14.0.log) each record 40 passing installed-wheel
tests and 26 expected GPU skips. That run tested PR head
`e27705320290ade4df26495da89c6dba40a6b995`; its package and GPU harness sources
are unchanged from the GPU validation SHA above. CPU CI, controller checks and
Lint also passed at that PR head.

## Build requirements and scope

PyTorch 2.13 and 2.14 require C++20; older listed releases use C++17.
`BuildExtension` supplies the language standard. Missing-Ninja fallback is
checked by asserting `use_ninja` is false, independently of whether PyTorch
uses warnings or logging for its diagnostic.

The GPU harness uses an isolated environment from the image's `python3`;
system-Python images need `python3-venv`. The GPU-free build job installs into
disposable container Python with `PIP_BREAK_SYSTEM_PACKAGES=1`.
See [installation requirements](../../installation.md).

The reports certify only their source revisions and listed environments.
Sanitizer coverage here is limited to 2.8.0 and 2.14.0 on one L4 with CUDA 12.6;
the [Python matrix](../python-matrix/README.md) supplies runtime evidence for
other Python/PyTorch pairs. Job logs have ANSI escapes and trailing whitespace
removed.

# Runtime CUDA metadata-check selection

The public API now accepts the keyword-only `check_cuda_metadata=False`.
The default CUDA kernels trust the documented spatial-shape/offset preconditions;
`True` selects separately compiled checked kernels in forward and backward.
The dispatcher saves the choice in the autograd context. There is no mutable
global mode, and unchecked kernels have no metadata-validation branch.

Tensor dtype/device/rank/shape checks, launch-size validation, guarded int32
selection, and the int64 fallback remain enabled. CPU/MPS retain their existing
metadata-content validation. All launch-bound annotations are unchanged.
See the [API contract](../../../api.md#cuda-constraints-and-errors).

## Validation

Direct Runpod L4, PyTorch 2.10.0+cu126, nvcc 12.6.85, sm_89. The installed
wheel was built from the working tree in [candidate-sources.json](candidate-sources.json).

| Check | Result |
| --- | --- |
| [CUDA regression](cuda-tests.json) | 74 passed, 11 expected skips, zero failures/errors |
| Compute Sanitizer | memcheck, racecheck, synccheck, initcheck: zero errors; racecheck: zero warnings |
| [Forced-int64 diagnostic](fallback-report.json) | 127 subtests passed |
| [Actual Kernel Hub binding](adapter-report.json) | Output/all-gradient reference checks and combined AOT compilation passed |
| Local CPU/MPS, PyTorch 2.5.1 | 58 passed, 27 expected skips; 11 build tests passed |
| Static checks | Ruff lint/format, ty, and `git diff --check` passed |

The CUDA suite covers both modes across FP32/FP64 reduction families, AMP,
dynamic compilation, CUDA Graph replay, and checked-mode device assertions.
Autograd retains the flag for interleaved calls; export and the adapter's AOT
forward/backward graphs retain it as well. The int64 diagnostic changes only
host index selection in a copied build. Adapter checks use the actual C++
binding, but do not cover the Nix builder or HF download/loader pipeline.

## Focused performance check

FP32 forward, batch 2, 8 heads, 32 channels, four levels of sizes 64², 32²,
16², and 8², four sampling points. Decoder uses 300 queries; encoder uses 5440.
Both modes use identical inputs on one nondefault stream, and their outputs
are bitwise equal. Seven paired repetitions shuffle mode order. Each timing
sample replays a CUDA Graph containing 50 calls; each repetition takes the
median of five samples. The table reports medians over repetitions.

| Case | Checked, µs | Unchecked, µs | Reduction |
| --- | ---: | ---: | ---: |
| Decoder forward | 35.90 | 34.43 | 4.10% |
| Encoder forward | 648.24 | 623.96 | 3.75% |

Unchecked is faster in all seven matched repetitions for each case. Encoder
paired improvements range from 1.16 to 56.53 µs, so the exact percentage should
not be treated as a universal constant. This measures the toggle alone with
launch bounds retained. It is not a new HF comparison, whole-model speedup,
or low-precision performance claim. Do not compare these absolute latencies
with the previous Pod's measurements to infer a regression.

[Raw timings](toggle-benchmark.json), [paired summary](summary.json).

The installed binary contains separate bool-specialized kernels. For FP32/int32
forward, unchecked has 232 static SASS instructions and 41 registers; checked
has 304 and 42. These match the earlier bounded check-ablation variants.
Static instruction counts are not executed instruction counts or a timing
breakdown. [Binary inspection](binary-summary.json).

## Evidence and reproduction

[toggle-evidence.tar.gz](toggle-evidence.tar.gz) contains the candidate source
archive and patch, build/test/sanitizer logs, diagnostic scripts, raw SASS and
resource dumps, and reports. All 67 manifest entries were verified after
download; [verification](evidence-verification.json).

Archive SHA256:
`d05b602f4bfc17098aae6f184d537e4295c9e6fa25cd888b9d57bbae7bb88d95`.

Inside the archive, `source/diagnostics/` contains `pod_setup.sh`,
`build_candidate.sh`, `validate_candidate.sh`, `fallback_check.py`,
`adapter_check.py`, `benchmark_toggle.py`, `collect_binary.py`, and
`pack_results.py`. They record the `/workspace/ci` layout used for this run.
Build the candidate, run regression/sanitizer tests, build/check the forced-int64
and adapter copies, then benchmark with no other GPU workload running.
`candidate-sources.json` identifies the exact tested tree.

The Pod was deleted after downloading and verifying the results; see
[cleanup verification](runpod-cleanup.json).

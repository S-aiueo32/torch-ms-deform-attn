# Guarded indexing and native autograd: measured results

Guarded int32 indexing and native C++ autograd reduced FP32 decoder training
latency by 20.8% in this same-process L4 comparison. Encoder forward improved
by 5.6% but remained about 5% slower than HF.

This historical candidate always checked CUDA metadata; the later
[runtime switch](../direct-20260917-toggle/README.md) changes that default.
The tested source is identified by [hashes](candidate-sources.json) and
[patch](candidate.patch), based on PR #16 commit
`2d604d2d9e2871ce2dc957278bbdae211e4b3fa8`.

## Changes

- [CUDA indexing](../../../../csrc/cuda/index_utils.h) selects int32 only when
  every chunk-local value, sampling and output span fits with interpolation
  and grid-loop headroom. The type now covers strides, bilinear offsets and
  every backward reduction branch. Large spans use int64; inter-chunk host
  offsets remain int64.
- [Metadata validation](../../../../csrc/cuda/ms_deform_im2col_cuda.cuh) retains
  device assertions and validates raw int64 values before narrowing. Unsigned
  range comparisons and a wide 32-by-32-bit area product replace more expensive
  equivalent checks. Invalid negative, high-bit and overflowing metadata is
  still rejected. There is no unchecked production path.
- [Native autograd](../../../../csrc/dispatcher.h) handles first-order forward
  and backward in C++, shared by the canonical package and Kernel Hub adapter.
  Python FakeTensor registrations remain, and opaque operator calls preserve
  AOT/export behavior. Saved tensor hooks, mutation detection, Compiled
  Autograd and explicit higher-order rejection are covered by regression tests.

Interpolation, gradient accumulation precision, low-precision promotion,
numerical tolerances and compiler fast-math settings are unchanged.

## Same-process results

NVIDIA L4, driver 595.91.07, CUDA 12.6, PyTorch 2.10.0+cu126, EPYC 9254.
The original PR source is rebuilt under an isolated operator namespace and
loaded alongside the candidate and the exact pinned HF artifact
`abfd4042216fa4f84c9c5c4e3e844a3143c70ad5`. The HF binary SHA256 is
`1a5053e022f3e5840ca9f40d361b5356ef0beeefdd8686029378dd4a7284b959`.
Both original and candidate use the same CUDA compiler and sm_89 target.

The main comparison has 20 qualified configurations and 280 timing rows:
decoder/encoder, FP32/FP16, five backends, two modes and seven repetitions.
Every configuration passes the independent FP64 output/all-three-gradient
oracle. FP16 uses the common FP32-compute policy with casts included.
Backend order is shuffled each repetition; each wall timing window is at
least 0.6 seconds. Inputs, qualification and graph replay share one dedicated
nondefault stream. No extra cgroup throttling was recorded during this run.

Median wall latency in microseconds, including the public API:

| Case | Dtype | Mode | Original PR | Candidate | HF | Candidate vs original |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Decoder | FP32 | Forward | 42.62 | 39.75 | 39.27 | −6.7% |
| Decoder | FP32 | Forward + backward | 207.66 | 164.38 | 165.02 | −20.8% |
| Decoder | FP16 | Forward | 67.45 | 64.83 | 64.66 | −3.9% |
| Decoder | FP16 | Forward + backward | 288.35 | 252.96 | 367.89 | Inconclusive |
| Encoder | FP32 | Forward | 670.46 | 633.12 | 600.46 | −5.6% |
| Encoder | FP32 | Forward + backward | 3,042.18 | 2,930.36 | 2,909.46 | −3.7% |
| Encoder | FP16 | Forward | 809.84 | 777.15 | 738.19 | −4.0% |
| Encoder | FP16 | Forward + backward | 3,187.57 | 3,091.79 | 3,016.60 | −3.0% |

All seven decoder FP32 training repetitions improve: candidate 162.37–170.75
µs versus original 189.93–289.73 µs. Encoder improvements also hold in all seven
matched repetitions and in CUDA Graph measurements. Encoder graph FP32
forward changes 660.82 → 629.24 µs, and forward + backward 3,116.51 → 3,018.01
µs. Graph intervals include all captured kernels and replay gaps; they are
not pure MSDA kernel durations or a direct measure of Python overhead.

Decoder FP16 training is noisy: original 263.69–380.95, candidate
229.58–326.91, HF 228.52–414.52 µs. Candidate wins only four of seven matched
wall repetitions; its graph median is 198.03 µs versus original 192.49 and HF
175.83 µs. The aggregate wall medians do not establish a speedup over HF.

The remaining encoder forward gap is about 5.3–5.4% versus HF; encoder
training is about 0.7% slower in FP32 and 2.5% slower in FP16. This is a
measured reduction of the gap, not a claim of universal HF parity.

[Main measurements](comparison-summary.json),
[host-sensitive diagnostic](host-summary.json). Full distributions, raw
measurements, and correctness details are in the evidence archive below.

## Host overhead and generated code

A fresh-process decoder FP16 confirmation alternated four backends in short
blocks. All six epoch wall medians favored the candidate, but the median
paired candidate/HF CUDA Graph ratio was 1.016. The wall benefit therefore
does not imply faster GPU execution than HF. Short-block samples are correlated;
win counts are not independent-trial significance estimates.
[Confirmation summary](confirmation-summary.json).

Six fresh-process profiles recorded all expected forward/backward launches
and selected the candidate's int32 templates. Python profiling found 100 calls
each to `fill_defaults`, Python autograd forward, and `redispatch` in the
original; none appeared in the candidate's recorded call path.
[Profiles](profiles.json).

For FP32 forward, static integer instructions fell from 328 to 205 and registers
from 54 to 42; HF used 148 and 39. Selected kernels had no local loads/stores.
These counts support reduced address/checking work, but do not measure dynamic
stalls or achieved occupancy. [Binary analysis](binary-summary.json).

## Correctness and compatibility

- Final Linux regression run: 83 tests, 72 passed, 11 expected skips, zero
  failures/errors/unexpected skips. The skipped cases require a CPU-only
  build, a second CUDA GPU, or Metal. [Report](cuda-tests.json).
- Compute Sanitizer `memcheck`, `racecheck`, `synccheck`, and `initcheck` each
  pass the four selected CUDA test methods, covering channel/reduction
  branches, padding, collisions, offsets and partial chunks.
- A separately built diagnostic forces the real int64 templates on ordinary
  inputs: all 81 subtests pass with no skips. [Report](fallback-report.json).
- The actual Kernel Hub C++ binding is built with the shared dispatcher and
  CUDA sources. Adapter/canonical parity, all-gradient FP64 checks and combined
  AOT graphs pass. This checks the binding, not the external HF loader or the
  complete Nix/CMake packaging pipeline. [Report](adapter-report.json).
- CPU PyTorch 2.4.0 and 2.5.1 each pass all ten integration tests; real Metal
  passes all nine existing tests. [Local validation and hashes](VALIDATION.md).
- The metadata helper passes 120,250 scalar equivalence cases under UBSan,
  including INT64 extrema and assertion-disabled invalid-input returns.
  All eleven local build/export tests, Ruff, formatting, type checking and
  `git diff --check` also pass.

## Evidence and reproduction

See [REPRODUCE.md](REPRODUCE.md) for the build and measurement sequence.
[fix-evidence.tar.gz](fix-evidence.tar.gz) contains raw comparisons, traces, Python profiles,
SASS/resource dumps, sanitizer logs, candidate sources and diagnostic scripts.
Its embedded `results/evidence-manifest.json` records every member's SHA256.
All 152 listed members were independently verified after downloading;
[verification record](evidence-verification.json). The archive SHA256 is
`400cc8771d164b09b89e1c540405b41d60afe362da628a2cda939c231293e83a`.
Local CPU/Metal records are stored beside this report.

Both rented Pods are deleted. The first attempt encountered a transfer failure;
the second completed the measurements. Deletion of each recorded Pod identifier
was verified: [attempt 1](runpod-attempt1-cleanup.json),
[attempt 2](runpod-attempt2-cleanup.json). The controller's final account-wide
listing was unavailable, but it separately verified deletion of the active Pod.

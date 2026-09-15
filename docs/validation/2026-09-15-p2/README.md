# P2 validation — 2026-09-15

T08, T10, T12 and T13 have implementation and execution evidence. T11 covers the
installed OpenMP/serial builds; a real native-thread-pool PyTorch build remains
unverified and explicitly best effort.

## Results

| Environment / check | Result |
| --- | --- |
| macOS arm64, Apple M2, Python 3.11, PyTorch 2.5.1 | [60 tests](cpu-tests.log): 37 passed, 23 expected GPU skips |
| Linux, 2 × L4, Python 3.11.10, PyTorch 2.5.1+cu124 | [58 tests](two-gpu-initial/cuda-tests.json): 57 passed; only CPU-only-wheel test skipped |
| Linux, 1 × L4, Python 3.11.10, PyTorch 2.5.1+cu124 | [58 tests](one-gpu/cuda-tests.json): 56 passed; two-GPU and CPU-only-wheel tests skipped |
| Additional near-knot regression on installed CUDA wheel | [2 tests](one-gpu/sampling-precision.log): CPU and CUDA both passed, zero skips |
| Fresh CPU-only wheel on 1 visible GPU | [1 test](one-gpu/cpu-only-gpu-tests.json): passed, zero skips |
| Real CPU backend transitions | [All 5 cases passed](cpu-builds/summary.json), including the expected required-OpenMP build failure |
| Ninja absent from PATH | [Build, load and 14 CPU tests passed](cpu-builds/missing-ninja.log) through automatic setuptools fallback |
| CPU strict Inductor benchmark | [24 passing measurement rows](cpu-compile.json) |
| CUDA eager/compiled, float32/fp16/bf16, small/decoder/encoder | [324 passing rows](one-gpu/benchmark-cuda.json), 3 repetitions |
| CUDA batch 1/4, D 16/32, step 2/64 | [144 passing rows](one-gpu/benchmark-cuda-sweep.json), including profiler CUDA activity times |

Build/benchmark-policy tests: 7 passed. Controller/evidence tests: 42 passed.
Ruff lint/format and ty passed. The successful [GPU completion marker](one-gpu/cuda-completion.json)
was written only after CUDA validation, benchmarks and the CPU-only-wheel test.
No new sanitizer claim is made by these P2 runs.

Each successful CPU transition binary was loaded in a fresh process and passed
14 CPU tests. The transitions used the same build tree: auto/Ninja OpenMP,
serial/setuptools, automatic fallback with invalid OMP_PREFIX, required OpenMP
failure, then restored OpenMP. Native-thread-pool selection was tested with
mocks only; it does not establish real native-backend binary compatibility.

CUDA dynamic compile produced graph counts `[1, 2, 2, 3]` for each dtype. The third
call changed only metadata values relative to the previous shape signature and
did not recompile. The fourth introduced unit dimensions. Repeating all four
signatures held the count at three. Fullgraph forward and all three gradients
matched the reference. The two-GPU test also compared all gradients and verified
current-device restoration.

## Sampling precision finding

The first run passed the entire two-GPU test suite, but its encoder float32
benchmark correctly failed and exited nonzero. It is **not** a successful overall
benchmark/release run; its failed configuration has no timing rows.

[Reproduction](one-gpu/uniform-native-reproduction.json) isolated the normalized
coordinate `0.1328124850988388`. At resolution 64, this maps to pixel
`7.999999046325684`, just before an interpolation knot. Float32 reference
normalization `2*y-1` rounds it onto pixel 8, choosing the other one-sided
derivative. Eager and compiled references agree with each other here. The native
operator preserves the original coordinate; the added CPU/CUDA regression
compares its output and all three gradients with a float64 oracle, including a
known positive slope before the knot.

The benchmark now samples at least a quarter pixel away from knots. Existing
boundary tests remain. Gradient tolerances were not loosened. The complete CUDA
measurement matrix then passed. The [reproduction script](reproduce-uniform.py.txt)
can be run with Python in a CUDA environment with `benchmarks/` on PYTHONPATH.

These short timings validate coverage and the harness, not a stable performance
baseline or a claimed regression improvement. Supplemental diagnosis/testing
also used the validation GPU during this session. For performance decisions,
rerun on an otherwise idle GPU with the [noise policy](../../benchmarks.md#p2-measurement-coverage-and-regression-policy).

## Source identity and cleanup

The controller used frozen local Git snapshots, without committing the user's
working tree. The archived source is available for exact reproduction:

- Two-GPU snapshot `4ef793ac702d4cf73a3bbc8363d0dd32f98f2260`:
  [source archive](two-gpu-initial/source.tar.gz), [manifest](source-manifest.json),
  [log](two-gpu-initial/cuda-checks.log).
- Successful one-GPU snapshot `99540d89d294e722c8305c83adaaf625c6948dac`:
  [source archive](one-gpu/source.tar.gz), [manifest](source-manifest-final.json),
  [log](one-gpu/cuda-checks.log). This validation-only snapshot adds a reproduction
  script and invokes it before benchmarking. The working tree has the same Python
  AST for shared code; later changes are comments/documentation plus the separately
  tested near-knot regression. Its test-file hash is recorded in the supplemental
  test log and final manifest. Snapshot SHAs are not published GitHub commits.

All three created Pods were verified deleted: [two-GPU](two-gpu-initial/cleanup.json),
[cancelled one-GPU attempt](cancelled/cleanup.json), and [successful one-GPU](one-gpu/cleanup.json).
Rates at creation were $0.98/hour, $0.49/hour and $0.49/hour. Each had a 60-minute
controller deadline, so even a full-hour allocation for every attempt totals
$1.96 in quoted compute charges, within the approved $3 limit. This is a
conservative compute bound, not a billing statement.

Artifact checksums are recorded in each completed run's `sha256.json`. Binary
wheels are not committed. A later release SHA still needs its own release checks.

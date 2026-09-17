# Execution semantics, CPU builds and benchmarks

## Objective and conditions

Validate noncurrent-device execution on two GPUs, dynamic compilation,
low-precision sampling, CPU backend selection, and correctness of the benchmark
measurement grid. Execution date: 2026-09-15. The source snapshot associated
with each result is listed below.

## Results

| Environment / check | Result |
| --- | --- |
| macOS arm64, Apple M2, Python 3.11, PyTorch 2.5.1 | [60 tests](cpu-tests.log): 37 passed, 23 expected GPU skips |
| Linux, 2 × L4, Python 3.11.10, PyTorch 2.5.1+cu124 | [58 tests](two-gpu-initial/cuda-tests.json): 57 passed; only CPU-only-wheel test skipped |
| Linux, 1 × L4, Python 3.11.10, PyTorch 2.5.1+cu124 | [58 tests](one-gpu/cuda-tests.json): 56 passed; two-GPU and CPU-only-wheel tests skipped |
| Near-knot regression on installed CUDA wheel | [2 tests](one-gpu/sampling-precision.log): CPU and CUDA both passed, zero skips |
| Fresh CPU-only wheel on 1 visible GPU | [1 test](one-gpu/cpu-only-gpu-tests.json): passed, zero skips |
| Real CPU backend transitions | [All 5 cases passed](cpu-builds/summary.json), including the expected required-OpenMP build failure |
| Ninja absent from PATH | [Build, load and 14 CPU tests passed](cpu-builds/missing-ninja.log) through automatic setuptools fallback |
| CPU strict Inductor benchmark | [24 passing measurement rows](cpu-compile.json) |
| CUDA eager/compiled, float32/fp16/bf16, small/decoder/encoder | [324 passing rows](one-gpu/benchmark-cuda.json), 3 repetitions |
| CUDA batch 1/4, D 16/32, step 2/64 | [144 passing rows](one-gpu/benchmark-cuda-sweep.json), including profiler CUDA activity times |

Build/benchmark-policy tests: 7 passed. Controller/evidence tests: 42 passed.
Ruff lint/format and ty passed. The successful [GPU completion marker](one-gpu/cuda-completion.json)
was written only after CUDA validation, benchmarks and the CPU-only-wheel test.
Sanitizers are covered in the separate [kernel validation](../kernel-correctness/README.md).

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

## Sampling precision contract

The near-knot regression compares native CPU/CUDA output and all three gradients
against a float64 oracle. The normalized coordinate `0.1328124850988388` maps
to pixel `7.999999046325684` at resolution 64, just before an interpolation knot.
Float32 reference normalization `2*y-1` rounds it onto pixel 8 and selects the
other one-sided derivative; eager and compiled references agree with each other.
The native operator preserves the original coordinate and its positive slope
before the knot. See the [reproduction data](one-gpu/uniform-native-reproduction.json),
[regression log](one-gpu/sampling-precision.log) and
[reproduction script](reproduce-uniform.py.txt).

Benchmark inputs stay at least a quarter pixel away from interpolation knots;
boundary behavior is covered by correctness tests. Gradient tolerances are
unchanged. The two-GPU archive contains a failed encoder float32 benchmark and
no timing rows for that configuration; its passing test suite establishes
device correctness only. The complete passing measurement grid is in the
one-GPU reports above.

These short measurements establish benchmark coverage, not a stable performance
baseline. Performance comparisons require an idle GPU and the
[noise policy](../../benchmarks.md#p2-measurement-coverage-and-regression-policy).

## Source identity and evidence

| Evidence scope | Source snapshot | Reproduction evidence |
| --- | --- | --- |
| Two-GPU correctness | `4ef793ac702d4cf73a3bbc8363d0dd32f98f2260` | [source](two-gpu-initial/source.tar.gz), [manifest](source-manifest.json), [log](two-gpu-initial/cuda-checks.log) |
| One-GPU correctness and benchmark grid | `99540d89d294e722c8305c83adaaf625c6948dac` | [source](one-gpu/source.tar.gz), [manifest](source-manifest-final.json), [log](one-gpu/cuda-checks.log) |

Snapshot SHAs identify archived local Git snapshots, not published GitHub commits.
The one-GPU snapshot includes the reproduction script and invokes it before
benchmarking. The supplemental near-knot test has its own test-file hash in the
regression log and final manifest. Each completed run has a `sha256.json`;
binary wheels are not committed.

Cleanup evidence covers all allocations: [two GPUs](two-gpu-initial/cleanup.json),
[cancelled allocation](cancelled/cleanup.json), and [one GPU](one-gpu/cleanup.json).
All were verified deleted. Quoted rates were $0.98, $0.49 and $0.49 per hour,
with 60-minute controller deadlines: $1.96 combined full-deadline compute
estimate, not a billing statement.

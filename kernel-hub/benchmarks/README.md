# Phase 3: MSDA comparison

Status: [Six-implementation L4 results and repeat comparison](../../docs/validation/kernel-hub-benchmarks/README.md).
All common-policy correctness checks passed. Small/decoder eager training
slowdowns remain an investigation item before the HF proposal.

This harness compares the installed canonical package, its Kernel Hub adapter,
the pinned published HF artifact, MMCV's MSDA CUDA source, PyTorch `grid_sample`,
and `rziga/msda-triton`. Competitor revisions and source hashes are recorded.
The MMCV comparison compiles its unmodified CUDA source with a minimal binding
and autograd adapter; it is not a benchmark of the full MMCV package wrapper.
Triton uses its strict frontend so an exception cannot silently time the
package's PyTorch fallback. Native BF16 is unsupported by that pinned frontend.

Run through the existing ownership-aware Runpod controller:

```bash
gh workflow run cuda.yml --ref perf/msda-dispatch-overhead \
  -f workload=kernel-hub -f kernel_hub_suite=benchmark \
  -f torch_version=2.8.0 -f gpu='NVIDIA L4' \
  -f max_hourly_usd=0.50 -f capacity_wait_minutes=15 -f timeout_minutes=45
```

The container selector above supplies CUDA 12.6; the workload pins PyTorch
2.10.0 and builds both the canonical package and Kernel Hub artifact. Source
downloads and Builder configuration run before GPU rental. Compilation, Triton
JIT and warmup are excluded from measurement; the Pod is deleted on completion
or failure. This suite runs operator benchmarks, not the Phase 2 model matrix.
Capacity waiting polls the selected GPU every 30 seconds without creating a
Pod and consumes the overall timeout. Price/authentication errors are not
retried. `gpu-startup.json` records whitelisted readiness observations without
Pod environment or SSH credentials if a created host fails to become ready.

## Comparison contract

- Three shapes: small (B=1, Q=100, H=4, C=16, two levels), decoder (B=2,
  Q=300, H=8, C=32, four levels), encoder (same with Q=5440). Four points/level,
  `im2col_step=64`, contiguous input, zero padding and `align_corners=False`.
- FP32, explicit FP16 and BF16. The common `fp32-compute` policy promotes
  floating inputs and casts output back. The canonical package and adapter
  already do this internally. All casts and their autograd costs are timed.
  Competitor `native` rows additionally exercise low-precision inputs without
  adaptation; these rows implement a different precision policy.
- Correctness is checked before timing against an independent FP64 PyTorch
  reference: output and all three gradients, with location gradients expressed
  in feature-pixel units. Absolute/relative tolerances are 2e-4 (FP32), 2e-3
  (FP16), and 2e-2 (BF16). Rows record each maximum error and pass/fail status.
  Incorrect/unsupported native cases receive no timing; required common-policy
  failures fail the overall run. Missing competitors are recorded and fail it.
- Sampling uses interior quarter-pixel locations on power-of-two feature
  sizes, representable even in BF16, to avoid derivative jumps at interpolation
  knots. This does not measure arbitrary boundary/offset distributions.
- Eager inference forward, backward-only on a retained graph, and fresh
  forward+backward. Five warmups, three repetitions, one-second minimum wall-time
  windows; backend/policy order is reshuffled each repetition with a recorded seed. No compile
  throughput or whole-model performance claim is made.
- Reuse the established CUDA timer: synchronized wall-clock medians/IQRs,
  CUDA-event stream intervals, and peak/incremental allocated bytes. Event
  intervals include host-induced idle time, not just CUDA kernel activity.
  Allocator baselines include the inputs and oracle tensors; backward-only
  also includes its retained graph. Reservations and driver memory are excluded.

The diagnostic workflow runs adapter eager/compile correctness, canonical
CUDA regressions and opcheck before timing. It currently selects canonical,
adapter, HF and `upstream-before-perf` via `--backends`, with decoder/encoder,
FP32/FP16 and forward/forward+backward. The previous upstream is pinned to
`e07889a886a9e3052ffc10d019ac5d88ba3f40f6`; its CUDA source is compiled unchanged
and its Python API is loaded under an isolated operator namespace. This gives
an old/new comparison in the same process on the same GPU. The summary JSON
includes old/new ratios for both repetition medians and matched repetitions.
Invoking the harness without selectors retains the full six-implementation,
three-shape, three-dtype comparison. `--native-control` measures the same
upstream C++ kernels through a minimal legacy autograd wrapper, bypassing public
dispatcher integration; it is a diagnostic control, not an alternative public
API. `--profile-kernels` optionally records summed CUDA kernel/copy activities
separately from timed windows. Missing profiler activities are recorded as a
warning and do not invalidate separately measured wall times. Sample counts and calls per sample identify windows with
too few samples for a useful IQR. Historical runs before this investigation
used 0.2-second windows; compare their settings before attributing differences
to the registration change.

See the [performance investigation](../../docs/validation/kernel-hub-performance/README.md)
for source-bound trials and limitations.

Compare identical case/dtype/policy/mode rows. Native and promoted rows quantify
the tradeoff of different precision policies and must not be presented as
equivalent arithmetic. Apply the investigation threshold in
[the benchmark policy](../../docs/benchmarks.md#measurement-and-regression-policy);
confirm an apparent regression in a fresh process before calling it repeatable.

CPU qualification test (in the Phase 2 validation environment):

```bash
python -m pytest kernel-hub/benchmarks/test_benchmark.py -v
```

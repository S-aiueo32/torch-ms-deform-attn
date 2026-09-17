# Phase 3: MSDA comparison

This harness compares the installed canonical package, its Kernel Hub adapter,
the pinned published HF artifact, MMCV's MSDA CUDA source, PyTorch `grid_sample`,
and `rziga/msda-triton`. Competitor revisions and source hashes are recorded.
The MMCV comparison compiles its unmodified CUDA source with a minimal binding
and autograd adapter; it is not a benchmark of the full MMCV package wrapper.
Triton uses its strict frontend so an exception cannot silently time the
package's PyTorch fallback. Native BF16 is unsupported by that pinned frontend.

Run through the existing ownership-aware Runpod controller:

```bash
gh workflow run cuda.yml --ref feat/kernel-hub-benchmarks \
  -f workload=kernel-hub -f kernel_hub_suite=benchmark \
  -f torch_version=2.8.0 -f gpu='NVIDIA L4' \
  -f max_hourly_usd=0.50 -f timeout_minutes=45
```

The container selector above supplies CUDA 12.6; the workload pins PyTorch
2.10.0 and builds both the canonical package and Kernel Hub artifact. Source
downloads and Builder configuration run before GPU rental. Compilation, Triton
JIT and warmup are excluded from measurement; the Pod is deleted on completion
or failure. This suite runs operator benchmarks, not the Phase 2 model matrix.

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
  forward+backward. Five warmups, three repetitions, 0.2-second minimum wall-time
  windows; backend/policy order is shuffled with a recorded seed. No compile
  throughput or whole-model performance claim is made.
- Reuse the established CUDA timer: synchronized wall-clock medians/IQRs,
  CUDA-event stream intervals, and peak/incremental allocated bytes. Event
  intervals include host-induced idle time, not just CUDA kernel activity.
  Allocator baselines include the inputs and oracle tensors; backward-only
  also includes its retained graph. Reservations and driver memory are excluded.

Compare identical case/dtype/policy/mode rows. Native and promoted rows quantify
the tradeoff of different precision policies and must not be presented as
equivalent arithmetic. Apply the investigation threshold in
[the benchmark policy](../../docs/benchmarks.md#measurement-and-regression-policy);
confirm an apparent regression in a fresh process before calling it repeatable.

CPU qualification test (in the Phase 2 validation environment):

```bash
python -m unittest discover -s kernel-hub/benchmarks -p test_benchmark.py -v
```

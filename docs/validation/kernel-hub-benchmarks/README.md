# Phase 3 benchmark evidence

Harness and methodology: [Kernel Hub benchmarks](../../../kernel-hub/benchmarks/README.md).
**The six-implementation L4 benchmark completed successfully** in
[run 35206461746](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35206461746),
source `ba51116`, PyTorch 2.10.0+cu126, NVIDIA driver 580.159.04.
All 54 common-policy backend/shape/dtype configurations passed output and all
three gradient checks against the independent FP64 reference. Each produced
three modes × three repetitions: 486 qualified timing rows.

- [Complete tables](run-35206461746/benchmark-tables.md)
- [Raw measurements, correctness and environment](run-35206461746/benchmark-kernel-hub.json)
- [Summary, IQRs and HF comparison flags](run-35206461746/benchmark-summary.json)
- [Pinned sources](run-35206461746/benchmark-sources.json),
  [build log](run-35206461746/kernel-hub-checks.log), and
  [verified Pod deletion](run-35206461746/runpod-state.json)

## Findings and limits

These measurements precede PR #16. The subsequent
[performance investigation](../kernel-hub-performance/README.md) records the
implemented dispatch/indexing improvements and optional metadata validation,
with separate before/after measurements. The historical latency ratios below
must not be presented as measurements of the current implementation.

The common policy computes in FP32 and includes casts for FP16/BF16 inputs.
On the encoder shape, adapter forward+backward takes 3.045 / 3.233 / 3.199 ms
for FP32 / FP16 / BF16, compared with HF 2.877 / 3.031 / 3.029 ms: roughly
6–7% slower. Canonical package results are similar. Incremental allocation is
identical to HF at 37.19 / 69.06 / 69.06 MiB. These are eager operator results,
not compiled or whole-model throughput.

Small/decoder eager training needs further investigation before the HF
proposal. Adapter decoder FP16 forward+backward is 0.636 ms versus HF 0.373 ms
(1.71×); the earlier process measured 1.51×. FP32 encoder forward is 1.12× HF
in both processes. Across all modes/dtypes, 29 of 54 adapter/canonical-to-HF
comparisons exceed the investigation threshold in both runs. This establishes
repeatable differences in some configurations, not a stable multiplier for
all workloads or proof of their cause. In particular, small/decoder combined
measurements vary substantially, including between the canonical package and
adapter. Short timer windows sometimes yield zero within-window IQR; raw
repetition medians must also be inspected. Profiling with longer measurement
windows is the next step before choosing an optimization.

The independent process is [run 35205545939](run-35205545939/benchmark-tables.md)
(source `65fc1e7`). It measured the other five implementations with the same
settings and recorded CPU/GPU/software versions, but failed overall because
Triton lacked installed package metadata. `ba51116` fixes installation without
changing those five implementations. Adding Triton changes shuffled execution
order, so this is not a controlled host-noise or kernel-only experiment.

Native HF FP16/BF16, MMCV FP16 and Triton FP16 failed this FP64 accuracy gate;
MMCV and Triton native BF16 are unsupported. All 18 native configurations are
excluded from timing. This reflects the stated accuracy contract, not an
unsupported-dtype regression in the canonical package: all six implementations
passed all three dtypes under the common policy. No tolerance was relaxed.

The benchmark implementation and evidence are ready for review. See the linked
performance investigation for the follow-up results and remaining HF gaps;
this report does not declare the adoption gate satisfied.

## Provisioning attempts

| Run | GPU | Outcome |
| --- | --- | --- |
| [35201349228](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35201349228) | L4 | SSH readiness timed out; workload never started; Pod deletion verified |
| [35203260598](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35203260598) | L4 | No matching capacity; no Pod requested |
| [35203378149](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35203378149) | RTX A5000 | No matching capacity; no Pod requested |
| [35203516504](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35203516504) | RTX 4090 | Quote $0.74/hour exceeded the initial $0.50/hour cap; no Pod requested |
| [35203625272](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35203625272) | RTX 4090 | Cancelled before workload startup to remove unnecessary FP32 cast dispatch from comparator wrappers; Pod deletion verified |
| [35203791682](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35203791682) | RTX 4090 | SSH readiness timed out after five minutes; workload never started; Pod deletion verified |
| [35204527482](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35204527482) | L4 | No matching capacity; no Pod requested |
| [35204792661](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35204792661) | L4 | CUDA builds passed; benchmark loader needed a Path argument; Pod deletion verified |
| [35205545939](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35205545939) | L4 | Five implementations measured; Triton metadata missing; Pod deletion verified |
| [35206461746](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35206461746) | L4 | All six implementations measured successfully; Pod deletion verified |

The two later RTX 4090 attempts used a $0.75/hour cap. Current requests use
L4 only, capped at $0.50/hour, with up to 15 minutes of capacity polling before
renting a Pod. Benchmark SSH
startup waits are bounded to five minutes; execution retains a 45-minute cap
and mandatory ownership-checked cleanup. Builder preparation occurs before
Pod creation. Provisioning failures do not constitute benchmark evidence.

The first three created Pods were deleted, as recorded in the archived states:
[first L4 attempt](run-35201349228/runpod-state.json),
[cancelled 4090 attempt](run-35203625272/runpod-state.json), and
[final 4090 attempt](run-35203791682/runpod-state.json).
The full workflow logs linked above contain the startup/price/capacity errors.
Run 35204792661 also verified Pod deletion; its [state](run-35204792661/runpod-state.json)
and [startup observations](run-35204792661/gpu-startup.json) are archived.
Run 35205545939 also verified [Pod deletion](run-35205545939/runpod-state.json).
Its benchmark source is `65fc1e7`; `ba51116` fixes Triton installation without
changing the other five implementations or their measurement settings.

Local validation passed: CPU precision/gradient qualification checks across
three dtypes (including rejection of a forward-correct, gradient-broken
implementation), a complete driver/serialization check with external backends
substituted on CPU, 52 runner/controller tests, Ruff and type checks.

# Phase 3 benchmark evidence

Harness and methodology: [Kernel Hub benchmarks](../../../kernel-hub/benchmarks/README.md).
**Phase 3 measurement is in progress on L4.** Run 35205545939 measured five
implementations across all three shapes and dtypes. All common FP32-compute
checks passed (output and all gradients); the overall run failed because the
Triton source was not installed with its required package metadata. That
preparation error is fixed in `ba51116`; a fresh L4 run is in progress.

[Initial tables](run-35205545939/benchmark-tables.md) and
[raw results](run-35205545939/benchmark-kernel-hub.json) are partial evidence,
not a completed six-backend comparison. Native HF low-precision and MMCV FP16
rows failed the FP64 accuracy gate; MMCV native BF16 is unsupported. Those
rows have no timings. Some small/decoder eager forward+backward measurements
are slower than HF, with substantial between-repetition variation; the
fresh run must establish whether those differences repeat.

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

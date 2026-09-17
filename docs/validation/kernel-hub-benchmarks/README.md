# Phase 3 benchmark evidence

Harness and methodology: [Kernel Hub benchmarks](../../../kernel-hub/benchmarks/README.md).
**GPU measurement is blocked at Pod startup; Phase 3 is not complete.** No
latency or memory results are established by the provisioning attempts below.
The benchmark workload has not yet executed on CUDA, so comparator compilation
and GPU correctness remain unvalidated. Do not infer performance from CPU tests.

## Provisioning attempts

| Run | GPU | Outcome |
| --- | --- | --- |
| [35201349228](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35201349228) | L4 | SSH readiness timed out; workload never started; Pod deletion verified |
| [35203260598](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35203260598) | L4 | No matching capacity; no Pod requested |
| [35203378149](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35203378149) | RTX A5000 | No matching capacity; no Pod requested |
| [35203516504](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35203516504) | RTX 4090 | Quote $0.74/hour exceeded the initial $0.50/hour cap; no Pod requested |
| [35203625272](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35203625272) | RTX 4090 | Cancelled before workload startup to remove unnecessary FP32 cast dispatch from comparator wrappers; Pod deletion verified |
| [35203791682](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35203791682) | RTX 4090 | SSH readiness timed out after five minutes; workload never started; Pod deletion verified |

Subsequent RTX 4090 requests explicitly use a $0.75/hour cap. Benchmark SSH
startup waits are bounded to five minutes; execution retains a 45-minute cap
and mandatory ownership-checked cleanup. Builder preparation occurs before
Pod creation. Provisioning failures do not constitute benchmark evidence.

All three created Pods were deleted, as recorded in the archived states:
[first L4 attempt](run-35201349228/runpod-state.json),
[cancelled 4090 attempt](run-35203625272/runpod-state.json), and
[final 4090 attempt](run-35203791682/runpod-state.json).
The full workflow logs linked above contain the startup/price/capacity errors.
The final benchmark source was `979c745`; documentation-only updates follow it.

Local validation passed: CPU precision/gradient qualification checks across
three dtypes (including rejection of a forward-correct, gradient-broken
implementation), 48 runner/controller tests, Ruff and type checks.

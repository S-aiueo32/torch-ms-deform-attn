# L4 performance investigation

This follows the [initial Phase 3 comparison](../kernel-hub-benchmarks/README.md).
The objective is to reduce eager overhead without changing interpolation,
gradient accumulation precision, or the FP16/BF16 promotion policy.

## Implementation

The canonical extension and HF adapter share `csrc/dispatcher.h`. Their forward
and backward operators now call native kernels directly through the C++
dispatcher. Layout normalization happens in C++, while FakeTensor and autograd
registrations remain in Python. Higher-order gradients still raise explicitly.

CUDA forward uses 32-bit quotient/remainder to decompose an output index only
when the launch work count fits int32. All tensor offsets, strides and sampling
indices remain int64; larger work retains the previous decomposition. Thus a
small output does not imply that input or sampling buffers must fit int32.
Device metadata checks remain in place. No fast-math flag or tolerance changes
were introduced.

## Measurement

The initial diagnostic trials compare the canonical package, adapter, pinned HF
artifact, and a direct-native control through minimal legacy autograd. The
control bypasses public dispatcher integration and is not a new public API.
The final focused trial instead includes the previous upstream source pinned
at `e07889a886a9e3052ffc10d019ac5d88ba3f40f6`, compiled from its unchanged CUDA
source and loaded with its Python API under an isolated operator namespace.
Backend order is reshuffled per repetition. This compares old/new in one
process and avoids attributing cross-host differences to the code change.
All comparisons use the common FP32-compute policy including explicit casts.
Each configuration must pass the FP64 output/all-gradient oracle before timing.

One-second timing windows replace the initial 0.2-second windows. Reports keep
sample counts, calls per sample, repetition medians and IQRs. CPU affinity and
cgroup counters are recorded to help assess scheduling variability. Profiling
is optional and outside timing; unavailable activities are missing information,
not zero GPU time. Ordinary timings are retained with a warning in that case.

## Trials

### Python registration trial: 35208208001

[Workflow](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35208208001),
source `369b124`, PyTorch 2.10.0+cu126, L4. Adapter numerical/compile tests and
canonical CUDA opcheck passed. The overall benchmark failed because optional
profiling returned no CUDA activities for some configurations; the initial
error handling discarded their timings. This handling is fixed subsequently.

The surviving small FP32 combined measurements have roughly 0.018 ms of CUDA
activities versus much larger and variable wall latencies (including 0.451 ms
canonical, 0.361 ms adapter and 0.263 ms HF medians). Encoder low-precision
forward also retains a native execution gap. These observations motivated the
direct C++ bindings and integer decomposition change; this first trial does
not establish that the slowdown was solved.

[Raw report](run-35208208001/benchmark-kernel-hub.json),
[partial tables](run-35208208001/benchmark-tables.md),
[test/build log](run-35208208001/kernel-hub-checks.log),
[adapter summary](run-35208208001/kernel-hub-summary.json), and
[verified Pod deletion](run-35208208001/runpod-state.json).

### Native C++ and index decomposition: 35210946607

Source `6fde720`; [workflow passed](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35210946607).
All 36 common-policy configurations passed the output/all-gradient gate,
producing 324 timing rows. Adapter tests (3), canonical CUDA tests (11 passed,
2 skipped), and CUDA opcheck/dynamic compile tests (2) passed. Incorrect or
unsupported native low-precision rows were excluded.

| Adapter vs HF, wall ms | Adapter | HF | Ratio |
| --- | ---: | ---: | ---: |
| Decoder FP32 forward+backward | 0.5683 | 0.4225 | 1.35 |
| Decoder FP16 forward+backward | 0.8862 | 0.7461 | 1.19 |
| Decoder BF16 forward+backward | 0.9207 | 0.8072 | 1.14 |
| Encoder FP32 forward+backward | 3.0470 | 2.9176 | 1.04 |
| Encoder FP16 forward+backward | 3.1727 | 3.0084 | 1.05 |
| Encoder BF16 forward+backward | 3.1599 | 2.9912 | 1.06 |

Encoder forward remains 9–12% slower than HF. Its direct-native control has
similar latency, so dispatcher overhead does not explain the entire gap.
Encoder incremental peak allocation matches HF: 37.19 MiB for FP32 and
69.06 MiB for the low-precision policy.

This host uses EPYC 7542 / driver 570.195.03, versus EPYC 9254 / driver
580.159.04 in the initial Phase 3 run. Therefore the smaller decoder FP16 ratio
(1.19 versus 1.71) does **not** isolate improvement due to this change. No
additional cgroup CPU throttling was recorded during this trial; that does not
rule out other scheduling variability.

[Raw report](run-35210946607/benchmark-kernel-hub.json),
[tables](run-35210946607/benchmark-tables.md),
[test/build log](run-35210946607/kernel-hub-checks.log), and
[verified Pod deletion](run-35210946607/runpod-state.json).

Local updated-source tests passed on PyTorch 2.5.1 (including MPS) and 2.10
(CPU); Linux CI passed on 2.4.0 serial/OpenMP and 2.14.0 OpenMP after updating
the export test's emulated native namespace. The latter test change does not
alter runtime implementation.

### Same-process old/new comparison: 35212849809

[Workflow passed](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35212849809),
source `4a818a3`, L4 / EPYC 9254 / driver 570.195.03 / PyTorch 2.10.0+cu126.
The canonical package, adapter, HF baseline and pinned previous upstream all
passed the FP64 output/all-gradient gate for decoder/encoder × FP32/FP16:
16 qualified configurations, 96 timing rows across two modes and three
repetitions. Two native HF FP16 cases failed the separate precision gate and
received no timings. Adapter and canonical CUDA regression/compile tests also
passed. Every timing window collected at least 12 samples.

| Forward+backward, wall ms | Previous upstream | New canonical | New adapter | HF |
| --- | ---: | ---: | ---: | ---: |
| Decoder FP32 | 0.3299 | 0.1911 | 0.2890 | 0.1626 |
| Decoder FP16 | 0.5381 | 0.4793 | 0.4655 | 0.3999 |
| Encoder FP32 | 3.0607 | 3.0571 | 3.0579 | 2.9148 |
| Encoder FP16 | 3.2099 | 3.2150 | 3.2100 | 3.0365 |

Decoder FP16 adapter latency is 13.5% below the previous upstream median;
matched-repetition ratios are 0.848, 0.913 and 0.956. It remains 16.4% slower
than HF. Decoder FP32 adapter improves by 12.4% in the aggregate but remains
77.7% slower than HF; adapter repetitions span 0.187–0.301 ms while canonical
spans 0.186–0.288 ms. These share native implementation, so their differing
aggregate medians must not be treated as a reliable distribution-specific
speed difference. A fresh-process repeat would be needed to establish a stable
improvement percentage.

Encoder is effectively unchanged from the previous upstream. Forward medians
also change by less than 1% in all four old/new configurations. This trial does
not establish a benefit from the index decomposition change. Against HF,
encoder forward remains 8.8–12.3% slower and forward+backward 4.9–5.7% slower.
Thus this is a partial eager-overhead improvement, **not** a resolved HF
performance gate. The PR remains a draft for further performance work.

[Raw report](run-35212849809/benchmark-kernel-hub.json),
[summary with matched-repetition ratios](run-35212849809/benchmark-summary.json),
[tables](run-35212849809/benchmark-tables.md),
[test/build log](run-35212849809/kernel-hub-checks.log),
[pinned source hashes](run-35212849809/benchmark-sources.json), and
[verified Pod deletion](run-35212849809/runpod-state.json).

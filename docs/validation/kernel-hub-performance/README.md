# MSDA performance changes and validation

PR #16 reduces CUDA indexing and eager autograd overhead, and makes CUDA
metadata-content validation optional. This page summarizes the implemented
changes, their measured effects, and the remaining HF performance gap.

## Implemented behavior

The canonical package and Kernel Hub adapter share a native C++ dispatcher
and first-order autograd implementation. Python registers FakeTensor behavior;
opaque operators remain visible to AOT compilation and export.

CUDA uses int32 for chunk-local indexing, strides, and bilinear offsets when
host-side bounds prove they fit. Larger spans use int64, and offsets between
batch chunks remain int64.

`ms_deform_attn(..., check_cuda_metadata=False)` selects unchecked CUDA kernels
by default. Pass `True` to check spatial-shape and level-offset contents in
both forward and backward. The flag is saved per call and captured by compiled
graphs. Tensor shape/device/dtype checks and integer-range guards remain enabled;
CPU/MPS still validate metadata contents. See the [API contract](../../api.md).

Interpolation, accumulation precision, FP16/BF16 promotion, numerical tolerances,
and launch bounds are unchanged.

## Measured effects

Each row below compares variants on the same L4 within one experiment. Rows
from different experiments must not be combined into an overall speedup.

| Experiment | Workload | Before → after | Interpretation |
| --- | --- | --- | --- |
| [Indexing and native autograd](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-performance/direct-20260917-fix/README.md) | FP32 decoder forward + backward, wall | 207.66 → 164.38 µs | 20.8% lower latency with metadata checks retained |
| Same experiment | FP32 encoder forward, wall | 670.46 → 633.12 µs | 5.6% lower latency; HF measured 600.46 µs |
| [Metadata switch](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-performance/direct-20260917-toggle/README.md) | FP32 encoder forward, CUDA Graph | 648.24 → 623.96 µs | Unchecked is 3.75% faster than checked |

The same-toolchain [HF rebuild](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-performance/direct-20260917-rebuild/README.md) found little
encoder FP32 forward difference between published and rebuilt HF. A diagnostic
removal of the unchecked forward's `__launch_bounds__(1024)` reduced latency
further. That change is **not implemented**: the FP64/int64 paths need separate
launch-resource validation before changing the annotation.

The evidence does not establish HF parity across all shapes and dtypes.
Decoder FP16 training timings vary, and the final toggle experiment measures
FP32 forward only.

## Validation of the final implementation

The [metadata-switch report](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-performance/direct-20260917-toggle/README.md) identifies the
exact tested source and contains the final results:

- L4/PyTorch 2.10: 74 passed, 11 expected skips; both modes cover numerical
  gradients, AMP, dynamic compilation, CUDA Graphs, and invalid metadata.
- Compute Sanitizer memcheck, racecheck, synccheck, and initcheck: no errors.
- Forced-int64 diagnostic: 127 subtests passed.
- Actual Kernel Hub C++ binding: numerical parity and combined AOT forward and
  backward passed, including propagation of the checked flag.
- Local PyTorch 2.5.1 CPU/MPS: 58 passed, 27 expected skips; 11 build tests passed.

The adapter validation does not cover the external HF loader or Nix build.
All direct Runpod sessions were cleaned up; each report links deletion records.

## Evidence index

| Report | Question answered |
| --- | --- |
| [Initial direct profiling](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-performance/direct-20260917/README.md) | Which indexing, validation, and Python-call costs explain the gap? |
| [Production indexing/autograd changes](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-performance/direct-20260917-fix/README.md) | Do the guarded implementation and native autograd improve performance and preserve correctness? |
| [Matched HF rebuild](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-performance/direct-20260917-rebuild/README.md) | Does the toolchain explain the remaining gap, and what does removing launch bounds change? |
| [Runtime metadata switch](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-performance/direct-20260917-toggle/README.md) | Does the public switch preserve integration behavior and remove checking cost? |

Earlier GHA comparisons remain available as historical evidence:
[initial trial](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-performance/run-35208208001/benchmark-tables.md),
[native dispatcher trial](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-performance/run-35210946607/benchmark-tables.md), and
[same-process comparison](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-performance/run-35212849809/benchmark-tables.md).
Their measurements precede the final implementation.

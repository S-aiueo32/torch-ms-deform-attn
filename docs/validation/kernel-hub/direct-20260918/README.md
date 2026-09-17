# BF16 AMP compiled-training diagnostics

These L4 diagnostics use source `86d64a54d210863b63ef7a15bfa1f74af6ba63b6`,
PyTorch 2.10.0+cu126 and the pinned published HF baseline
`abfd4042216fa4f84c9c5c4e3e844a3143c70ad5`. They investigate the failed
[full-suite case](../run-35233588181/e2e-bf16-amp.json); they are not a replacement
for a full-suite gate.

The actual exported C++ binding and CUDA sources were compiled with
`torch.utils.cpp_extension.load` at `-O3`, for SM 8.9. The generated Python
namespace helper and loader metadata were synthesized. Both candidate and HF
were loaded through the real `kernels.get_local_kernel` path and bound through
Transformers. This is not the Kernel Builder CMake or Nix distribution route.
The [upstream manifest](UPSTREAM.json) records exact exported file hashes.

## Results

- The uninstrumented BF16 AMP runner passed all four eager/compiled,
  inference/training cases with the diagnostic namespace:
  [report](plain-bf16-amp.json), [log](plain.log).
- Rebuilding with the exact native namespace used by the failed GHA run also
  passed all four cases: [report](exact-bf16-amp.json), [build log](exact-build.log),
  [run log](exact.log). This changes the namespace alone, not the arithmetic.
- An instrumented run, after the same three preceding BF16 AMP cases, also
  passed compiled training: [report](matrix-report.json).
- Both layers' captured forward inputs and outputs were bitwise equal between
  candidate and HF. Backward inputs matched too; value-gradient differences
  were at most `7.45e-9`, and location/weight gradients were identical.
  Replaying each implementation on the other's captured inputs gave the same
  bounds. See the [operator comparison](matrix-comparison.json).

The instrumented run replaces Python access to each operator's default overload
only after compilation/warmup, records the actual calls, then restores it.
It retains operator/profiler requirements and the original numerical tolerances.
The [generated-code log](compiled-code.log.gz) and
[diagnostic scripts](diagnostic-scripts.tar.gz) preserve the procedure. The raw
captured tensor file was temporary on the Pod; the numeric comparison and code
log were recovered before/at cleanup.

The original `0.0703125` violating bias-gradient difference did not recur in
these diagnostics. These observations therefore do **not** prove its cause or
justify silently discarding that failed run. The standard runner now saves the
initial model state, input, labels, both graphs and comparison tensors in a
`*-debug.tar.gz` for BF16 AMP compiled training, allowing an exact fixture replay
if the discrepancy recurs in a formal build.

The session reached its bounded lifetime and the controller collected logs and
verified [Pod deletion](runpod-state.json). Dependency versions and GPU details
are in [packages](python-packages.txt) and [nvidia-smi](nvidia-smi.txt).

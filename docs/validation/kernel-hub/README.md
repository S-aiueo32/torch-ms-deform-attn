# Kernel Hub / Transformers integration evidence

Recorded on 2026-09-17. This is **CPU fixture evidence**, not a CUDA Kernel
Builder artifact run and not completion of the Phase 1/2 GPU gates.

The fixture loaded exported Python adapter code through the real kernels local
loader, then used Transformers' `kernelize` integration to replace two RT-DETR
MSDA layers. A test-only `_ops.py` supplied the upstream CPU extension.
No loader or Transformers API was mocked. Model weights and labels are synthetic.

| Component | Version |
| --- | --- |
| PyTorch | 2.10.0 |
| torchvision | 0.25.0 |
| Transformers | 5.17.0 |
| kernels / kernels-data | 0.16.0 / 0.16.2 |
| SciPy | 1.17.1 |

All 3 fixture tests passed, including 20 matrix cases:

- FP32, explicit FP16/BF16, and FP16/BF16 autocast.
- Eager and `torch.compile(backend="aot_eager")`, each in inference and training.
- Logits, boxes, supervised detection loss, input gradients and 241 parameter
  gradient tensors per training case matched the reference within the recorded
  tolerances. The maximum absolute difference across all checked tensors was
  `6.103515625e-05`.
- Each inference compile captured one full graph. Each training compile captured
  nine graphs, permitting RT-DETR's training-only data-dependent finite checks.
  Captured graphs contained MSDA and the profiler observed actual namespaced
  forward/backward calls.
- The separate criterion test matched Transformers' public detector loss.
  The negative test rejected using the candidate as its own HF baseline.

The reference used Transformers' grid-sample implementation, with FP32 MSDA
interpolation inside the same-precision detector for low-precision cases.
This adaptation does not establish unmodified HF artifact dtype support.

Artifacts:

- [Matrix, source hashes, native binary hash and dependency versions](cpu-fixture.json)
- [Test execution log](cpu-fixture-tests.log)

The native CPU extension was built from `8afa986` in an isolated temporary
environment. The export manifest records the same base commit plus hashes of
the actual working-tree files, including uncommitted Phase 2 changes. Results
apply to those hashes, not arbitrary subsequent changes.

An earlier PyTorch 2.5.1 attempt failed full-graph capture in Transformers'
decorator code. With PyTorch 2.10.0, requiring full-graph training also failed at
RT-DETR's tensor-valued finite check; the final harness records training graph
breaks rather than claiming full-graph training. Initial cross-precision model
comparison changed proposal ordering; the final comparison keeps both models in
the same precision and records the reference MSDA precision adaptation.

Also checked: 10 build tests, Ruff lint/format and ty passed. The GPU runner
returned nonzero and wrote `status="failed"` when invoked without CUDA.

Still unverified: native HF C++/CUDA binding and builder output, CUDA/Inductor
execution, pinned published-HF artifact comparison, pretrained detection
accuracy, and other Transformers model families. See the
[reproduction protocol](../../../kernel-hub/e2e/README.md).

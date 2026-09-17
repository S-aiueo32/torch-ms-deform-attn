# Kernel Hub / Transformers integration evidence

Recorded on 2026-09-17. **CUDA E2E is partially validated; the full Phase 1/2
regression gates have not passed.**

## NVIDIA L4 / real Kernel Hub artifacts

[Run 35191748862](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35191748862)
tested source `fe516b7f063b10bed7e6945df17d988d935be8ca` with PyTorch
2.10.0+cu126, Transformers 5.17.0 and kernels 0.16.0. Builder 0.16.0 generated
the project on the CPU runner; its CMake build and `local_install` targets built
the CUDA artifact on the L4. This is a local development artifact, not a
publishable Nix distribution build.

The real loader and Transformers kernelization replaced both MSDA layers in a
small, randomly initialized complete RT-DETR detector. The published baseline
was pinned to `abfd4042216fa4f84c9c5c4e3e844a3143c70ad5`.

| Precision | Eager inference | Eager training | Inductor inference | Inductor training |
| --- | --- | --- | --- | --- |
| FP32 | Pass | Pass | Numerical mismatch | Numerical mismatch |
| FP16 | Pass | Pass | Pass | Numerical mismatch |
| BF16 | Pass | Pass | Pass | Numerical mismatch |
| FP16 autocast | Pass* | Pass* | Pass* | Numerical mismatch |
| BF16 autocast | Pass* | Pass* | Pass* | Numerical mismatch |

14 of 20 cases passed. Each passing training case checked the actual detection
loss, input gradients and 241 parameter gradient tensors. Profiler observations
verified namespaced MSDA forward/backward execution, and passing compiled cases
contained MSDA in captured graphs.

*The unmodified HF baseline rejected mixed autocast inputs. Those comparisons
used explicit FP32 interpolation in the baseline MSDA layer; reports retain its
original dtype error. They do not establish unmodified baseline AMP parity.*

Unresolved differences, with the original tolerances retained:

- FP32 compiled logits: maximum reported mismatch 0.0005014 in inference and
  0.0001178 in training (`atol=2e-5`, `rtol=2e-4`).
- FP16 compiled training logits: reported mismatches up to 2.75. Proposal
  ordering is a hypothesis to investigate, not an established cause.
- BF16 compiled training gradients: reported mismatches up to 0.0525
  (`atol=0.02`, `rtol=0.05`).
- Phase 1's published-HF direct operator comparison passed FP32/FP64 but failed
  FP16 gradient parity at its stricter 0.002 tolerance. The maximum reported
  difference was 0.009765625; that test stopped before BF16. The separate
  reference-based layer test passed all four dtypes in eager and compile modes,
  as did autocast and device/dtype validation.

Compiled-candidate versus eager-baseline comparison does not isolate compiler
numerics from kernel replacement. Next isolate candidate eager/compiled and
baseline eager/compiled behavior, including proposal indices, before changing
any tolerance. No pretrained dataset accuracy claim is made.

Evidence: [suite summary](run-35191748862/kernel-hub-summary.json),
[individual reports and logs](run-35191748862/),
[source manifest](run-35191748862/UPSTREAM.json),
[candidate hashes](run-35191748862/candidate-files.json),
[baseline hashes](run-35191748862/baseline-files.json), and
[verified Pod deletion](run-35191748862/runpod-state.json).
The overall run correctly failed. No Pod remains from this run.
Earlier setup failures and fixes are recorded in [Runpod attempts](runpod-attempts.md).

## Earlier CPU fixture

This section is CPU fixture evidence, separate from the CUDA run above.

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

The CPU fixture alone does not validate native CUDA artifacts. Pretrained
detection accuracy and other Transformers model families remain unverified. See the
[reproduction protocol](../../../kernel-hub/e2e/README.md).

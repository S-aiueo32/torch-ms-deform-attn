# Kernel Hub / Transformers integration evidence

Recorded on 2026-09-17. **CUDA E2E is partially validated; the full Phase 1/2
regression gates have not passed.**

The latest full run at `2e7cdb5` passed Phase 1 and 19/20 RT-DETR cases on L4.
BF16 autocast compiled training still fails its gradient comparison, so the
full Phase 2 gate remains open. Earlier focused passes do not override this run.

## Full regression after native dispatcher changes

[Run 35233588181](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35233588181)
tested `2e7cdb5d36922b82f97b2f9b0155752c2a1b9efa` with the harness corrections
below, using L4, PyTorch 2.10.0+cu126, Transformers 5.17.0 and kernels 0.16.0.
The three Phase 1 tests passed. All 20 E2E cases executed; 19 passed.

| Precision | Eager inference | Eager training | Inductor inference | Inductor training |
| --- | --- | --- | --- | --- |
| FP32 | Pass | Pass | Pass | Pass |
| FP16 | Pass | Pass | Pass | Pass |
| BF16 | Pass | Pass | Pass | Pass |
| FP16 autocast | Pass | Pass | Pass | Pass |
| BF16 autocast | Pass | Pass | Pass | **Gradient mismatch** |

All runs retain the documented precision/reference adaptations and compiler
policies. The failure is in
`model.decoder.layers.0.self_attn.o_proj.bias`: one of 32 elements exceeds
`atol=0.02, rtol=0.05`, with absolute error 0.0703125 and relative error
0.09677419 at that element. The tensor's overall maximum absolute difference
is 0.5; larger reference elements can satisfy the relative tolerance despite
larger absolute errors. No tolerance was changed.

The failed case's diagnostics show:

- Eager candidate versus eager HF: every parameter gradient, logits, boxes and
  loss match exactly; the maximum input-gradient difference is `2.98e-8`.
- Compiled candidate versus compiled HF: maximum logit difference `0.0078125`
  and loss difference `0.0006218`; the bias gradient above fails.
- Both compiled models differ from their own eager execution. The same bias
  tensor has maximum differences of 1.5 (candidate) and 1.4375 (HF).
- Initial proposals match without a query permutation.

These observations establish a remaining compiled BF16 AMP discrepancy; they
do not isolate its cause or prove it harmless. Subsequent
[L4 diagnostics](direct-20260918/README.md) compared captured MSDA inputs,
outputs and backward results, including a rebuild with the failed run's exact
namespace. The discrepancy did not recur: forward inputs/outputs matched
exactly and native value-gradient differences were at most `7.45e-9`. This does
not establish why the formal run failed. A fresh full run is still required;
the runner now archives BF16 AMP fixtures and graphs for exact replay.
Nix distribution validation and HF adoption remain
separate, uncompleted gates.

Evidence: [suite summary](run-35233588181/kernel-hub-summary.json),
[BF16 AMP cases and diagnostics](run-35233588181/e2e-bf16-amp.json),
[Phase 1 log](run-35233588181/phase1.log),
[source manifest](run-35233588181/UPSTREAM.json),
[candidate hashes](run-35233588181/candidate-files.json),
[baseline hashes](run-35233588181/baseline-files.json), and
[verified Pod deletion](run-35233588181/runpod-state.json).

Local checks of the corrected harness passed all 20 CPU fixture cases (four
tests). Exporter tests passed 3/3; controller/evidence tests passed 55/55, as did
Ruff and the PR's ordinary CI. CPU results do not replace the failed CUDA gate.

### Initial attempt and harness correction

[Run 35231276149](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35231276149)
tested merged main `bf256ebadce5e61841d97bbae72b9c92353a0cac` on L4 with
PyTorch 2.10.0+cu126. The real builder/loader and all three Phase 1 tests passed.
All 20 RT-DETR cases failed in the harness before completing comparison: its
profiler namespace lookup used `CustomOpDef._qualname`, and its graph counter
used `CustomOpDef._opoverload`. After PR #16 the registered forward is already
an `OpOverload`, so neither attribute exists. This run supplies no completed
E2E parity result.

The harness now compares graph targets directly with the registered overload
and uses its `name()` for profiler matching. Numerical tolerances, compiler
policies and operator execution requirements are unchanged.

Evidence: [suite summary](run-35231276149/kernel-hub-summary.json),
[Phase 1 log](run-35231276149/phase1.log),
[FP32 failures](run-35231276149/e2e-fp32.json),
[source manifest](run-35231276149/UPSTREAM.json), and
[verified Pod deletion](run-35231276149/runpod-state.json).

## Earlier integration evidence

The six initially failing compiled cases now have passing, source-bound
rechecks under explicit numerical policies and query-identity comparison.
BF16 AMP was also checked against a compiled HF reference. This is **not** a
claim that default Inductor matches eager element-for-element, nor that a
single final-revision full matrix has passed. Phase 1's native FP16/BF16
gradient differences have been traced to differing computation precision; see
the operator investigation below. Strict native-half numerical parity is not
provided by the FP32-compute adapter.

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

Differences recorded in the initial run, with the original tolerances retained:

- FP32 compiled logits: maximum reported mismatch 0.0005014 in inference and
  0.0001178 in training (`atol=2e-5`, `rtol=2e-4`).
- FP16 compiled training logits: reported mismatches up to 2.75. Proposal
  ordering was initially a hypothesis; the investigation below distinguishes
  changed proposals from a permutation of the same proposals.
- BF16 compiled training gradients: reported mismatches up to 0.0525
  (`atol=0.02`, `rtol=0.05`).
- Phase 1's published-HF direct operator comparison passed FP32/FP64 but failed
  FP16 gradient parity at its stricter 0.002 tolerance. The maximum reported
  difference was 0.009765625; that test stopped before BF16. The separate
  reference-based layer test passed all four dtypes in eager and compile modes,
  as did autocast and device/dtype validation.

Compiled-candidate versus eager-baseline comparison does not isolate compiler
numerics from kernel replacement. The investigation below isolates candidate
eager/compiled and baseline eager/compiled behavior, including proposal identity,
without changing tolerances. No pretrained dataset accuracy claim is made.

Evidence: [suite summary](run-35191748862/kernel-hub-summary.json),
[individual reports and logs](run-35191748862/),
[source manifest](run-35191748862/UPSTREAM.json),
[candidate hashes](run-35191748862/candidate-files.json),
[baseline hashes](run-35191748862/baseline-files.json), and
[verified Pod deletion](run-35191748862/runpod-state.json).
The overall run correctly failed. No Pod remains from this run.
Earlier setup failures and fixes are recorded in [Runpod attempts](runpod-attempts.md).

## Compiler-numerics investigation

[Run 35193235122](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35193235122)
at source `2cc2998` added comparisons against the candidate's own eager execution
and recorded initial decoder proposals. The original tolerances were unchanged.
The same failed tensors appeared when comparing compiled candidate to eager
candidate as when comparing it to HF; eager candidate versus HF passed.

With default compiler settings, explicit FP16 training changed proposals before
MSDA (maximum difference 4.17578125), along with logits (2.451416015625).
Inductor's `emulate_precision_casts` and `emulate_divison_rounding` settings
removed the proposal difference and reduced the logit difference to 0.009765625,
within the original tolerance. Explicit BF16 compiled training also passed;
its candidate-versus-eager logits were identical and the maximum checked
gradient difference was 0.00390625, within tolerance.

The eager-numerics matrix passed 16/20. FP32 and AMP training still required
investigation. FP16 AMP's proposal difference (5.41634464263916) also occurred
upstream of MSDA. This identifies model/compiler numerics as a contributor;
it is not evidence that all default-mode kernel replacements are equivalent.

See [per-case diagnostics](run-35193235122/),
[suite summary](run-35193235122/kernel-hub-summary.json), and
[verified Pod deletion](run-35193235122/runpod-state.json).

[Run 35195362147](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35195362147)
at `a5b226c` fixed the math policy further: TF32 disabled, math SDPA selected,
and Inductor pattern matching disabled. Both FP32 compiled cases passed with
the original tolerances. Maximum checked absolute differences were
`9.5367431640625e-07` in inference and `6.103515625e-05` in training (241
parameter gradients checked). AMP training still failed. This combined control
does not isolate which individual optimization caused the FP32 discrepancy.
See [reports](run-35195362147/) and [Pod deletion](run-35195362147/runpod-state.json).

[Run 35196661730](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35196661730)
at `138e4fd` corrected AOT's backward autocast assumption and checked query
identity by initial proposals. FP16 AMP passed: queries 4 and 5 in the second
image had exchanged places, while the selected proposal set was unchanged.
Only output rows were aligned; loss and all gradients passed unchanged.
BF16 AMP still had the same gradient discrepancy, so changing the backward
autocast assumption alone did not resolve that case.
See [reports](run-35196661730/) and [Pod deletion](run-35196661730/runpod-state.json).

[Run 35197412721](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35197412721)
at `a433fa0` passed the focused BF16 AMP compiled training check. The historical
HF artifact received shape-only FakeTensor registrations, with no change to its
CUDA arithmetic, and its model was compiled with the same backend, numerical
settings and backward-autocast assumption. The existing FP32 reference
interpolation adaptation for HF's mixed-AMP dtype error remained in place.
Candidate versus compiled HF maximum error was `9.5367431640625e-07` across
outputs, loss, input gradients and 241 parameter gradients.

The raw compiled-versus-eager comparison also happened to pass in this run
(maximum error `0.001313924789428711`). Thus this run does **not** prove that
compiling the baseline itself removed the previously reported `0.0634765625`
gradient discrepancy. It establishes substitution parity under matched
compilation; the earlier run-to-run eager/compiled variation remains recorded.

See [passing report](run-35197412721/e2e-bf16-amp.json),
[source manifest](run-35197412721/UPSTREAM.json),
[suite summary](run-35197412721/kernel-hub-summary.json), and
[verified Pod deletion](run-35197412721/runpod-state.json).

The confirmed rechecks are:

| Originally failing case | Passing run | Comparison conditions |
| --- | --- | --- |
| FP32 compiled inference/training | 35195362147 | Controlled math policy; original tolerances |
| Explicit FP16/BF16 compiled training | 35193235122 | Eager rounding policy; original tolerances |
| FP16 AMP compiled training | 35196661730 | Controlled math; matching initial proposals before output-row comparison |
| BF16 AMP compiled training | 35197412721 | Controlled math; compiled HF reference with shape-only registration |

No runtime numerical behavior of the distributed MSDA adapter was changed by
this investigation. Fixes concern compiler policy, AOT assumptions and a
comparison that previously conflated model/compiler numerics with replacement
of the kernel. Pretrained accuracy and performance under these policies still
require separate validation.

## Operator FP16/BF16 precision investigation

[Run 35199275357](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35199275357)
at `15f90c9` retained the original native-HF assertions and added two controls:
the pinned HF kernel on FP32-promoted inputs (results cast back to input dtype),
and an independent FP64 `grid_sample` forward/backward oracle. It checked the
same two-level shapes, seed and four input dtypes as the original test. All
four dtypes were evaluated even when a subtest failed.

For FP16 and BF16, **every candidate output and all three gradients exactly
matched HF with FP32 computation**. Against the FP64 oracle, the candidate
passed the original tolerances and had lower RMS error than native HF for
every output/gradient tensor. Native HF failed the oracle tolerance for FP16
location/weight gradients and BF16 location gradients.

| Tensor / input dtype | Candidate RMS error vs FP64 | Native HF RMS error vs FP64 | Candidate vs FP32-compute HF |
| --- | --- | --- | --- |
| Sampling-location gradient / FP16 | 0.00134653 | 0.00324162 | Exact |
| Attention-weight gradient / FP16 | 0.000358368 | 0.00104703 | Exact |
| Sampling-location gradient / BF16 | 0.00811373 | 0.02758888 | Exact |

The adapter computes interpolation and gradient accumulation in FP32 before
casting results back, as the upstream API already documents. Native HF uses
low-precision intermediate arithmetic and accumulation. Reproducing those
rounding errors would change the upstream precision contract; this is not a
candidate gradient defect demonstrated by this fixture.

The test now requires both matched-computation HF parity and independent FP64
accuracy, with the **original tolerances unchanged**. It additionally requires
the candidate's RMS error not to exceed native HF's on this fixture. Raw native
comparisons remain in `phase1-numerics.json`, but are diagnostic for FP16/BF16;
FP32/FP64 native comparisons remain gates. This is an explicit correction of
the comparison contract, not a claim that native-half differences vanished.
No production kernel arithmetic changed. This synthetic fixture does not
establish pretrained model accuracy or every possible input shape.

The diagnostic run failed its three retained native-half assertions; all added
accuracy controls, layer eager/compile checks and autocast checks passed.
See [numerical report](run-35199275357/phase1-numerics.json),
[test log](run-35199275357/phase1.log),
[source manifest](run-35199275357/UPSTREAM.json), and
[verified Pod deletion](run-35199275357/runpod-state.json).

[Run 35199874809](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35199874809)
at `4e21ffc` reran the corrected Phase 1 suite on L4 with PyTorch 2.10.0+cu126.
All three tests passed, covering FP32/FP64 native-HF parity, FP16/BF16
FP32-compute parity and FP64 accuracy, eager/compiled layer gradients, autocast,
and device/dtype validation. This was an operator-only run, not a new E2E matrix.
See [passing suite](run-35199874809/kernel-hub-summary.json),
[numerical comparisons](run-35199874809/phase1-numerics.json),
[test log](run-35199874809/phase1.log), and
[verified Pod deletion](run-35199874809/runpod-state.json).

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

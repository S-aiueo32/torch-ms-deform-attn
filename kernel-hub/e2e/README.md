# Phase 2: Transformers RT-DETR

The runner exercises a complete `RTDetrForObjectDetection` with a small ResNet,
hybrid encoder, two decoder layers, and detection heads. It uses random weights
and synthetic images/labels, so it establishes execution and numerical parity,
not pretrained accuracy or dataset mAP. Denoising and dropout are disabled to
make comparisons deterministic.

The candidate follows the actual integration path:

```text
RTDetrForObjectDetection
  -> transformers.integrations.hub_kernels.kernelize
  -> kernels.LocalLayerRepository / get_local_kernel
  -> exported layers.MultiScaleDeformableAttention
  -> upstream custom operator and native implementation
```

Every MSDA forward binding must match the loaded adapter. A profiler must observe
its namespaced forward (and backward in training). Compiled cases must also
contain the operator in captured Dynamo graphs. Missing bindings or calls fail
the run; other unrelated HF kernels are left on their original implementations.

## Run on a CUDA host

The repository's [Runpod workload](../../docs/gpu-runner.md#kernel-hub-and-transformers-workload)
automates building, testing, artifact collection, and verified Pod deletion.

First complete the Phase 1 builder run. `--kernel-dir` must refer to a local
Kernel Builder artifact or repository containing a compatible `build/` variant,
not the unbuilt export. Install the host's matching PyTorch/torchvision builds,
then install the separately pinned integration dependencies:

```bash
python -m pip install -r kernel-hub/e2e/requirements.txt
python kernel-hub/e2e/rt_detr.py \
  --kernel-dir /path/to/built/kernel \
  --dtype fp32 --numerics controlled --report /path/to/results/fp32.json
```

Each invocation runs eager/compiled inference and eager/compiled training.
Repeat with `--dtype fp16`, `--dtype bf16`, `--dtype fp16 --autocast`, and
`--dtype bf16 --autocast` for 20 cases in total. The runner uses Inductor on
CUDA, fails without a GPU, writes errors to its JSON report, and returns nonzero
on failure. No missing GPU or unsupported dtype is silently skipped.

To compare against the existing HF implementation, download a compatible
artifact at a pinned Hub commit, then add:

```bash
--baseline-kernel-dir /path/to/pinned/hf/artifact
```

The candidate and baseline must resolve to distinct modules. The report records
loaded module file hashes, versions, replaced layers, executed operator counts,
graph capture information, error tolerances and per-tensor maximum errors.
Retain the artifact's Hub commit and builder/export `UPSTREAM.json` alongside
the report. Without this option, the reference is Transformers' grid-sample
implementation; a pass does not establish parity with a published HF artifact.

## Precision and training protocol

Both models have identical weights and use the same surrounding-model precision.
Comparing an entire half detector with an FP32 detector can change discrete
proposal top-k selection before MSDA, obscuring the operator comparison.

For low-precision grid-sample reference cases, MSDA evaluates interpolation in
FP32 and casts the result back outside autocast, matching the candidate's
documented computation policy. This is recorded as
`reference_msda_fp32_adapter=true`.

With an HF baseline artifact, the runner first executes its unmodified layer.
Only a specific dtype-unsupported error permits retrying with FP32 reference
interpolation; the original exception is recorded as `baseline_original_error`.
Other baseline errors and all candidate errors fail. A promoted-reference pass
does not claim unmodified baseline support for that precision mode.

Training uses Transformers' supervised detection criterion, including encoder
proposals and auxiliary decoder losses. Hungarian matching and loss calculation
run eagerly in FP32; gradients flow through these casts. Tests check loss,
logits, boxes, input gradients, and every parameter gradient present in the
reference, requiring identical gradient coverage and finite values.

Compilation covers the learned detector core, including the heads attached to
its decoder. Inference requires `fullgraph=True`. RT-DETR's training-only
finite-value check uses data-dependent branching, so training permits model
graph breaks and records every compiled graph. The criterion is outside the
compiled region. This is not a claim of full-graph compiled training.

### Isolating compiler numerics

`--numerics default` preserves Inductor's normal optimizations. `--numerics eager`
uses PyTorch 2.10's `emulate_precision_casts=True` and
`emulate_divison_rounding=True` settings (the latter spelling is upstream's).
The first preserves intermediate low-precision rounding and disables Triton
floating-point fusion; the second matches eager division rounding. These are
explicit validation settings, not changes to the distributed kernel or to error
tolerances. They may affect performance and should not be used silently in a
benchmark.

`--numerics controlled` additionally disables TF32 for convolution/matmul,
selects the math SDPA backend for both models, and disables Inductor's pattern
matcher. This pins more of the surrounding model's numerical policy while
retaining the compiled model and native MSDA calls.

The AOT backward assumption is `backward_pass_autocast="off"`, matching this
runner's backward outside autocast. See the
[PyTorch compiler autograd guidance](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_backward.html).

Compiled reports additionally compare the candidate against its own eager
execution and compare that eager result against HF. The `proposals` tensor
records initial decoder reference points, upstream of MSDA, to reveal changes
in proposal selection/order. All per-tensor differences survive failed cases.
Use `--compiled-training-only` to reproduce training failures without rerunning
the entire matrix, or `--compiled-only` for both compiled inference/training.
The full GPU suite uses `controlled` for FP32/AMP and `eager` for explicit
FP16/BF16. Pass `--diagnostic-controls` to `gpu_suite.py` to additionally run
default-numerics FP32/FP16/BF16 training controls. Controls can deliberately
reproduce failures, so they are not enabled in normal validation.

For BF16 AMP the suite also uses `--compile-reference`: the historical HF ops
receive shape-only FakeTensor registrations so the reference model can run
under the same compiler. Their CUDA arithmetic is unchanged. The report retains
both sides' raw eager/compiled differences and the compiled-to-compiled result.
This is an explicit reference adaptation, not a claim that the unmodified HF
artifact supports compilation. `--compile-reference` is also available for
other precisions when isolating replacement from compiler behavior.

Top-k ties can reorder decoder queries. Output rows are matched using only their
initial proposals, with `atol`/`rtol` capped at `1e-5` for the proposal-set check.
Changed proposals fail; predictions are never used to choose a permutation.
Loss and all gradient tensors are compared without any permutation or relaxed
tolerance. Reports retain raw comparisons and the applied query permutation.
This establishes proposal-matched detector parity, not identical output row order.

## Local CPU fixture tests

Run these separately from the core suite with the E2E dependencies and a CPU
build of `torch-ms-deform-attn` installed against the same PyTorch:

```bash
python -m unittest discover -s kernel-hub/e2e -p test_rt_detr.py -v
```

The fixture exports the real Python adapter, then supplies a **test-only**
`_ops.py` backed by the installed upstream CPU extension and synthetic CPU
metadata. The real kernels loader and Transformers integration run unchanged.
The fixture registers a test-only Python autograd formula that calls its own
namespaced backward operator, preserving graph and profiler visibility. It does
not substitute a second operator through an Autograd dispatch implementation.
This validates integration plumbing, not Kernel Builder/native binding/CUDA
compatibility. The fixture's PyPI import never enters the distribution export.

Set `MSDA_CPU_FIXTURE_REPORT` to a JSON output path to record passing matrix
cases. A complete fixture matrix has 20 passing cases; use the test exit status
as well as the count. Optional validation dependencies require Python 3.11+.

Transformers 5.17.0 requires `0.16.0 <= kernels < 0.17.0`. The environment pins
kernels 0.16.2, the latest compatible patch release, together with matching
kernels-data 0.16.2. PyTorch 2.5.1 failed full-graph inference in Transformers'
decorator code; the fixture uses PyTorch 2.10.0 / torchvision 0.25.0 for
compilation checks.

The [L4 evidence](../../docs/validation/kernel-hub/README.md) records a full
post-dispatcher run at `2e7cdb5`: Phase 1 passed and 19/20 RT-DETR cases passed.
BF16 autocast compiled training still fails a bias-gradient comparison against
compiled HF, so the full regression gate remains open. The earlier 14/20 run
and passing focused rechecks are retained as historical evidence. Native
FP16/BF16 operator differences were traced to precision policy; the validation
record documents FP64 accuracy checks and the FP32-compute comparison contract.
Pretrained model
accuracy, RF-DETR and PP-DocLayoutV2 remain separate follow-up coverage.

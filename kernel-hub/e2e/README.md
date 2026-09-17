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
  --dtype fp32 --report /path/to/results/fp32.json
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

Compiled reports additionally compare the candidate against its own eager
execution and compare that eager result against HF. The `proposals` tensor
records initial decoder reference points, upstream of MSDA, to reveal changes
in proposal selection/order. All per-tensor differences survive failed cases.
Use `--compiled-training-only` to reproduce training failures without rerunning
the entire matrix. The Runpod workload runs the eager-numerics matrix plus
default-numerics FP32/FP16/BF16 training controls on the same GPU.

## Local CPU fixture tests

Run these separately from the core suite with the E2E dependencies and a CPU
build of `torch-ms-deform-attn` installed against the same PyTorch:

```bash
python -m unittest discover -s kernel-hub/e2e -p test_rt_detr.py -v
```

The fixture exports the real Python adapter, then supplies a **test-only**
`_ops.py` backed by the installed upstream CPU extension and synthetic CPU
metadata. The real kernels loader and Transformers integration run unchanged.
This validates integration plumbing, not Kernel Builder/native binding/CUDA
compatibility. The fixture's PyPI import never enters the distribution export.

Set `MSDA_CPU_FIXTURE_REPORT` to a JSON output path to record passing matrix
cases. A complete fixture matrix has 20 passing cases; use the test exit status
as well as the count. Optional validation dependencies require Python 3.11+.

Transformers 5.17.0 requires `0.16.0 <= kernels < 0.17.0`, hence the explicit
pin. PyTorch 2.5.1 failed full-graph inference in Transformers' decorator code;
the fixture uses PyTorch 2.10.0 / torchvision 0.25.0 for compilation checks.

The [L4 real-artifact run](../../docs/validation/kernel-hub/README.md) passed
14/20 cases, including all eager inference/training modes. Compiled numerical
differences and strict low-precision operator parity remain unresolved; the full
regression gate has not passed. Pretrained model
accuracy, RF-DETR and PP-DocLayoutV2 remain separate follow-up coverage.

# Kernel Hub adapter (experimental)

This is an experimental adapter, not an adopted or published Hugging Face kernel.
`torch-ms-deform-attn` remains the implementation source. No runtime import of,
or dependency on, its PyPI package is added to the exported kernel.

The repository is already at 0.1.0 with MPS support. This integration is a
subsequent milestone; the initial HF build targets CUDA only.

## Export

From the upstream repository root, choose a new output directory:

```bash
python kernel-hub/export.py /tmp/msda-kernel
```

The output contains `build.toml`, `flake.nix`, `csrc/`, `torch-ext/`, tests,
LICENSE, NOTICE, and `UPSTREAM.json`. CUDA sources are copied verbatim. Python
registration and functional code are copied with checked substitutions for the
builder namespace and local imports. An upstream change that invalidates a
substitution fails the export. The manifest records HEAD and hashes of the
actual source files, including any uncommitted changes. Export refuses to
overwrite an existing directory.

For a source archive without `.git`, pass `--revision FULL_SOURCE_SHA`; the
Runpod controller supplies the exact commit it archived.

Keep generated files out of upstream version control. Sync downstream by
regenerating from a reviewed upstream revision; do not edit generated kernels.
Changes found downstream should be applied upstream before the next export.

## Contracts

- `layers.MultiScaleDeformableAttention.forward` preserves HF's seven-argument
  contract, including the unused `value_spatial_shapes_list` argument.
- `ms_deform_attn_forward` / `ms_deform_attn_backward` preserve the explicit
  forward/backward API, including FP16/BF16 input compatibility. Low-precision
  computation is promoted to FP32 and explicit low-precision gradients are
  returned in the input dtype.
- The layer and `ms_deform_attn` use upstream dtype, contiguous-input,
  FakeTensor and first-order autograd handling. Explicit FP16/BF16 inputs
  compute in FP32 and return the input dtype; autocast returns FP32.
- Generated Python registrations use the builder's `add_op_namespace_prefix`.
  PyPI and multiple HF builds can therefore coexist without sharing op names.
- No CPU/MPS dispatch is registered in the HF native binding. The CPU test
  shim described below is only a test fixture.

The configuration is validated with builder 0.16.1. `flake.nix` pins the same
released builder line; generate and retain `flake.lock` before validating a
publishable Nix build. The recorded L4 evidence predates this patch-level update
and used builder 0.16.0's local CMake development route with PyTorch 2.10.0 and
CUDA 12.6.

## Validation

Local Python-glue checks (requires an installed upstream CPU extension):

```bash
uv run --locked pytest build_tests/test_kernel_hub_export.py -v
```

These substitute the CPU native extension for the builder-generated native
ops, testing independent namespaces, forward/gradients, low precision and
full-graph AOT tracing. They do **not** test HF loading, native binding
compilation, CUDA, or Inductor code generation.

On a Linux NVIDIA host with Kernel Builder/Nix installed, export the tree,
enter its builder testshell, and run:

```bash
python -m pytest tests/test_kernel.py -v
```

The tests require CUDA and `LOCAL_KERNELS` as configured by the builder's
testshell. Ensure its mapping resolves `kernels-community/deformable-detr`
to this export. Set `MSDA_HF_BASELINE_REVISION` to the full 40-character Hub
commit SHA of the existing artifact being compared. The baseline test loads
that revision in a subprocess without `LOCAL_KERNELS` and compares FP32/FP64
forward and backward over two feature levels. It requires Hub access or a
cached artifact compatible with the host's PyTorch/CUDA build. FP16/BF16
comparisons use the upstream FP32-compute contract and an independent FP64
oracle at the original tolerances. Raw native-HF low-precision differences are
recorded separately: native HF rounds intermediate arithmetic in the input
dtype, so strict native-half parity is not the adapter's precision contract.
The layer tests load via `kernels.get_kernel` and compare eager and
compiled layer outputs/gradients against an independent grid-sample reference.
Missing CUDA is an error, not a successful skipped validation.

The [L4 validation record](../docs/validation/kernel-hub/README.md) confirms native
build/loading, reference-based eager/compiled layer gradients, autocast and
invalid device/dtype checks. FP32/FP64 published-HF operator comparisons passed;
native FP16/BF16 gradient differences were traced to precision policy, with the
candidate closer to an independent FP64 oracle and exactly matching HF under
FP32 computation. The [Phase 2 RT-DETR runner](e2e/README.md)
initially passed 14/20 CUDA E2E cases, followed by passing focused rechecks.
After the native dispatcher changes and harness corrections, the full run at
`2ae90ab` passed all three Phase 1 tests and all 20 E2E cases, including BF16
autocast compiled training. The preceding 19/20 run remains archived; its BF16
discrepancy did not recur, but its cause has not been isolated. Wider PyTorch
coverage and a reproducible Nix build remain open.
[Phase 3 measurements](../docs/validation/kernel-hub-benchmarks/README.md)
cover six implementations on L4. The subsequent
[performance investigation](../docs/validation/kernel-hub-performance/README.md)
led to shared native autograd, guarded int32 CUDA indexing, and optional CUDA
metadata-content checks. These changes passed core correctness and sanitizer
checks, but do not establish HF performance parity across all shapes/dtypes.
Do not claim HF adoption or complete CUDA E2E compatibility.

## References

- [Builder layout, registration and namespace rules](https://github.com/huggingface/kernels/blob/main/docs/source/builder/writing-kernels.md)
- [Current downstream source and layer API](https://github.com/huggingface/kernels-community/tree/main/deformable-detr)
- [Transformers Kernel Hub mapping](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py)

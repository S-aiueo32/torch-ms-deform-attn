# Kernel Hub adapter (experimental)

This is a Phase 1 prototype, not an adopted or published Hugging Face kernel.
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

Keep generated files out of upstream version control. Sync downstream by
regenerating from a reviewed upstream revision; do not edit generated kernels.
Changes found downstream should be applied upstream before the next export.

## Contracts

- `layers.MultiScaleDeformableAttention.forward` preserves HF's seven-argument
  contract, including the unused `value_spatial_shapes_list` argument.
- `ms_deform_attn_forward` / `ms_deform_attn_backward` preserve the low-level
  API: contiguous CUDA FP32/FP64 tensors, with explicit backward.
- The layer and `ms_deform_attn` use upstream dtype, contiguous-input,
  FakeTensor and first-order autograd handling. Explicit FP16/BF16 inputs
  compute in FP32 and return the input dtype; autocast returns FP32.
- Generated Python registrations use the builder's `add_op_namespace_prefix`.
  PyPI and multiple HF builds can therefore coexist without sharing op names.
- No CPU/MPS dispatch is registered in the HF native binding. The CPU test
  shim described below is only a test fixture.

The builder configuration follows edition 5. `flake.nix` currently follows
the builder's main branch: generate and retain `flake.lock` on the Linux build
host before recording reproducible build evidence. CUDA, PyTorch and compiler
compatibility have not yet been established for this adapter.

## Validation

Local Python-glue checks (requires an installed upstream CPU extension):

```bash
uv run --locked python -m unittest discover -s build_tests -p test_kernel_hub_export.py -v
```

These substitute the CPU native extension for the builder-generated native
ops, testing independent namespaces, forward/gradients, low precision and
full-graph AOT tracing. They do **not** test HF loading, native binding
compilation, CUDA, or Inductor code generation.

On a Linux NVIDIA host with Kernel Builder/Nix installed, export the tree,
enter its builder testshell, and run:

```bash
python -m unittest discover -s tests -p test_kernel.py -v
```

The tests require CUDA and `LOCAL_KERNELS` as configured by the builder's
testshell. Ensure its mapping resolves `kernels-community/deformable-detr`
to this export. Set `MSDA_HF_BASELINE_REVISION` to the full 40-character Hub
commit SHA of the existing artifact being compared. The baseline test loads
that revision in a subprocess without `LOCAL_KERNELS` and compares FP32/FP64
forward and backward over two feature levels. It requires Hub access or a
cached artifact compatible with the host's PyTorch/CUDA build.
The layer tests load via `kernels.get_kernel` and compare eager and
compiled layer outputs/gradients against an independent grid-sample reference.
Missing CUDA is an error, not a successful skipped validation.

Phase 1 remains open until a real builder/loader/CUDA run passes and numerical
comparison against a pinned existing HF artifact is recorded. Also outstanding:
execution of the included GPU autocast and invalid device/dtype checks, PyTorch version coverage,
and a reproducible builder lock. Phase 2 real-model E2E and Phase 3 benchmarking
have not been run. Do not claim HF adoption or end-to-end compatibility yet.

## References

- [Builder layout, registration and namespace rules](https://github.com/huggingface/kernels/blob/main/docs/source/builder/writing-kernels.md)
- [Current downstream source and layer API](https://github.com/huggingface/kernels-community/tree/main/deformable-detr)
- [Transformers Kernel Hub mapping](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py)

# Kernel Hub adapter (experimental)

This is an experimental adapter, not an adopted or published Hugging Face kernel.
`torch-ms-deform-attn` remains the implementation source. No runtime import of,
or dependency on, its PyPI package is added to the exported kernel.

The PyPI package includes MPS support. Kernel Hub integration is a separate
milestone; the initial HF build targets CUDA only.

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

The configuration is validated with builder 0.16.1. `flake.nix` selects that
release; `flake.lock` fixes its transitive dependencies and is included in the
export. The full L4 regression used builder 0.16.1's local CMake
development route with PyTorch 2.10.0 and CUDA 12.6.

## Nix distribution build

On an x86_64 Linux Nix host, run from the upstream checkout:

```bash
bash scripts/run_kernel_hub_nix.sh /tmp/msda-nix-build
```

The [Nix build workflow](../.github/workflows/kernel-hub-nix.yml) runs the same
script. It builds `redistributable.torch211-cxx11-cu126-x86_64-linux` using the
Hugging Face binary cache and retains the builder's ABI and `get_kernel` checks.
Builder 0.16.1's Nix version set starts at PyTorch 2.11, so this artifact is a
different PyTorch variant from the previous 2.10 CMake regression.

The script exports upstream sources, uses a deterministic Git commit for the
builder's namespace identity, and builds with `--no-update-lock-file`.
The committed `kernel-hub/flake.lock` is required; dependencies are never
silently updated by the build script.
The `evidence/` directory contains the lock, upstream and export revisions,
exported source archive, derivation, build log, store metadata, and the
`distribution.tar.gz` artifact with a SHA-256 checksum and a
`distribution-files.json` manifest of its individual file hashes for checked GPU
preparation. The manifest is generated from the archived bytes and rejects
unsafe paths, duplicate file paths, and non-regular files. A successful build
writes `status.txt`; failed attempts still retain their available evidence.

This step builds one distribution variant without a GPU. CUDA Phase 1 and
RT-DETR validation must subsequently load this exact artifact. It does not
publish to the Hub or validate all supported build variants.

The Nix PR workflow also runs for changes under `csrc/` and `src/`, and for
LICENSE/NOTICE changes, because the export consumes those upstream files.
To validate a particular build, pass `nix_build_run=BUILD_RUN_ID` with the CUDA
workflow's `kernel_hub_suite=nix-full`; see the [GPU runner](../docs/gpu-runner.md).

## Long-term evidence storage

Actions artifacts expire. Before expiration, save the original Nix artifact ZIP
and the complete passing GPU artifact ZIP as assets on a dedicated **draft**
GitHub release. Draft assets remain available to authorized maintainers and
repository workflows; they are not public downloads. Keep this archive release
unpublished: publishing a release invokes the PyPI workflow. Use a tag such as
`validation/kernel-hub-nix-BUILD_RUN_ID` and attach:

- `kernel-hub-nix-BUILD_RUN_ID.zip`: the complete original Actions artifact,
  including `distribution.tar.gz`, its checksums, source manifest and build logs.
- `cuda-runpod-GPU_RUN_ID-ATTEMPT.zip`: the complete GPU artifact, including the
  20-case reports, loaded-file hashes and Pod deletion record.
- `SHA256SUMS`: checksums of both ZIP files.

Download the original ZIPs with `gh api
repos/OWNER/REPO/actions/artifacts/ARTIFACT_ID/zip > ASSET_NAME.zip`; the artifact
IDs are listed by `gh api repos/OWNER/REPO/actions/runs/RUN_ID/artifacts`.
Extract a separate copy of each ZIP with `python -m zipfile -e` for verification.
Put the GPU artifact's `runpod-state.json` alongside its `artifacts/` contents,
then run `scripts/verify_kernel_hub.py --sha VALIDATION_SHA --evidence GPU/artifacts`.
Also prepare the Nix artifact against a checkout with matching implementation
sources:

```bash
python scripts/prepare_kernel_hub_nix.py NIX_EVIDENCE NEW_PREPARED_DIRECTORY \
  --revision VALIDATION_SHA --build-run BUILD_RUN_ID
```

Before uploading, compare the prepared `NIX_BUILD.json` with the GPU report's
`NIX_BUILD.json`: the build run, archive hash, original source/export revisions,
variant and every file hash must agree. Record both Actions links and source
SHAs in the release notes. Create the archive using `gh release create` with
`--draft --target VALIDATION_SHA --notes-file NOTES_FILE` and the three assets;
do not publish it or replace assets from an earlier validation.

To reuse the archive, supply both `nix_build_run=BUILD_RUN_ID` and
`nix_archive_release=RELEASE_TAG` to the CUDA workflow. It loads the ZIP's original
distribution through the same source/hash checks as the Actions download.
An archive tag is an explicit selection, not an automatic fallback to a different
build. Reading drafts requires push access. The workflow grants `contents: write`
only to an archive-download job that runs no repository code, then passes the ZIP
to the controller job with its existing read-only permissions. The CLI equivalent
requires a maintainer login:

```bash
python scripts/prepare_kernel_hub_nix.py NEW_DOWNLOAD_DIRECTORY NEW_PREPARED_DIRECTORY \
  --revision VALIDATION_SHA --build-run BUILD_RUN_ID \
  --download --release RELEASE_TAG
```

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
coverage and independent byte-for-byte reproducibility remain open. The
[locked Nix distribution](../docs/validation/kernel-hub-nix/README.md) for
PyTorch 2.11 / CUDA 12.6 / x86_64 passed ABI/loading checks and, without
recompilation, Phase 1 3/3 and RT-DETR 20/20 on L4. Both AMP precisions compare
compiled candidate and HF under the same compiler policy.
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

# Kernel Hub Nix distribution build

This record covers the `torch211-cxx11-cu126-x86_64-linux` redistributable
variant, using Kernel Builder 0.16.1 and the committed
[`flake.lock`](../../../kernel-hub/flake.lock). It is separate from the
[PyTorch 2.10 CMake/GPU validation](../kernel-hub/README.md).

## Passing GPU validation of the distribution artifact

[Run 35291667815](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35291667815)
passed **Phase 1 3/3 and RT-DETR 20/20** on NVIDIA L4, PyTorch 2.11.0+cu126,
Python 3.11, kernels 0.16.2 and Transformers 5.17.0. Validation source revision:
`d9d1838af51416053ffd3ff289d5f04f4956fd40`. The candidate was the unchanged
distribution archive from Nix run 35280414318; no candidate compilation occurred
on the GPU host. Every loaded candidate file matched its recorded Nix hash.

| Precision | Eager inference | Eager training | Inductor inference | Inductor training |
| --- | --- | --- | --- | --- |
| FP32 | Pass | Pass | Pass | Pass |
| FP16 | Pass | Pass | Pass | Pass |
| BF16 | Pass | Pass | Pass | Pass |
| FP16 autocast | Pass | Pass | Pass | Pass |
| BF16 autocast | Pass | Pass | Pass | Pass |

AMP compiled cases compare candidate and HF under the same compiler, using the
documented shape-only fake registrations for HF. Both AMP training cases check
241 gradient tensors. The previously failing bias gradient matches compiled HF
exactly; the largest difference across all compared tensors is `2.91e-11`
(FP16 AMP) and `5.82e-11` (BF16 AMP). Both implementations retain the same
eager-versus-compiled bias-gradient differences (`0.171875` and `1.5`
respectively). This establishes replacement parity under the recorded compiler
policy, not equality between eager and compiled whole-model execution.
No tolerance or CUDA arithmetic was changed.

Evidence: [suite summary](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35291667815/kernel-hub-summary.json),
[Nix artifact identity](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35291667815/NIX_BUILD.json),
[validation source manifest](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35291667815/UPSTREAM.json),
[original Nix source manifest](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35291667815/nix-UPSTREAM.json),
[FP16 AMP report](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35291667815/e2e-fp16-amp.json),
[BF16 AMP report](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35291667815/e2e-bf16-amp.json), and
[verified Pod deletion](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35291667815/runpod-state.json).
The full matrix, source/artifact hashes and cleanup pass the evidence verifier:

```bash
python scripts/verify_kernel_hub.py \
  --sha d9d1838af51416053ffd3ff289d5f04f4956fd40 \
  --evidence docs/validation/kernel-hub-nix/run-35291667815
```

Replay archives for both AMP training cases are retained in GHA artifact
`cuda-runpod-35291667815-1`; their hashes are in
[debug-artifacts.json](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35291667815/debug-artifacts.json). All three GPU
attempts have verified Pod deletion. Local controller/evidence tests passed
58/58, including tampered archive/source rejection, plus three targeted E2E
harness tests and Ruff.

## Passing distribution build

[Run 35280414318](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35280414318)
passed on 2026-09-18 JST. It built the native CUDA extension and passed the
builder's layout/registration checks, manylinux_2_28 and Python ABI 3.9 checks,
and `get_kernel` loading check. The environment used PyTorch 2.11.0, CUDA 12.6,
Python 3.13.12 and Kernel Builder/runtime 0.16.1 from the locked Nix inputs.

- Tested upstream revision (GitHub PR merge commit):
  `edf0f06f97f06279fcd7c6b1b05087d5043a7786`.
- Export revision: `69d65ca2fdb13d8cda49065d7ecde41f012b0b2e`.
- Native module: `_deformable_detr_cuda_69d65ca.abi3.so`.
- Distribution archive SHA-256:
  `75c9a2786d7db81137f021e2bfd6fe3b99d968e9bad0ac34e7fb66fb74e51dc4`.
- Compiled CUDA architectures: 7.0, 7.2, 7.5, 8.0, 8.6, 8.7, 8.9, 9.0+PTX.

Evidence includes the [build log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35280414318/build.log.gz),
[upstream manifest](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35280414318/UPSTREAM.json),
[dependency lock](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35280414318/flake.lock),
[derivation](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35280414318/derivation.json),
[store metadata](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35280414318/store-path.json), and
[per-file hashes](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35280414318/distribution-files.json).
The downloaded archive's checksum, exported source hashes, and lock were
verified against the working tree. The distribution and source archives are
retained in the workflow artifact `kernel-hub-nix-35280414318`:

```bash
gh run download 35280414318 --repo S-aiueo32/torch-ms-deform-attn \
  --name kernel-hub-nix-35280414318 --dir /tmp/msda-nix-artifact
cd /tmp/msda-nix-artifact
sha256sum -c distribution.sha256
mkdir distribution
tar -xzf distribution.tar.gz -C distribution
```

The artifact is a distribution root containing the variant directory. The GPU
validation above loaded this exact binary with a compatible PyTorch 2.11
environment. No GPU execution occurred in the Nix build job itself.
Other Nix variants, a second independent byte-for-byte reproducibility check,
and Hub publication remain unvalidated. Dependency locking alone is not proof
of byte-for-byte reproducibility.

The workflow builds the Nix `redistributable` output, which enables the
builder's layout, namespaced registration, ABI and Kernel Hub import checks.
It retains the source/export revisions, dependency lock, derivation, build log,
distribution archive and its SHA-256 checksum. The GPU numerical and RT-DETR
tests above consumed that distribution archive.

## Initial build and registration correction

[Run 35279363509](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35279363509)
resolved the dependency lock and reached the native extension's source checks.
It failed the builder's static `torch.library` registration check before
compilation: the exported FakeTensor decorators passed the `forward` and
`backward` OpOverload variables, while the checker requires a direct call to
`add_op_namespace_prefix` at each decorator.

The exporter now emits those explicit namespaced strings. Native operator
lookup, FakeTensor implementations and CUDA arithmetic are unchanged. The
builder's own static checker passes on the corrected export, and the existing
CPU export tests pass both namespace fixtures, low precision, gradients and
full-graph AOT tracing. The lock generated by this initial run is now committed
and required for every export; builds refuse lock updates.

The failed run's [compressed log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35279363509/build.log.gz),
[derivation](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35279363509/derivation.json) and
[source manifest](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35279363509/UPSTREAM.json) are retained.

## Initial GPU run and PyTorch 2.11 harness adaptation

[Run 35282647966](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35282647966)
loaded the exact Nix distribution on NVIDIA L4 with PyTorch 2.11.0+cu126,
Python 3.11, kernels 0.16.2 and Transformers 5.17.0. The loaded files matched the
distribution hashes. Phase 1 passed 3/3 and all ten eager RT-DETR cases passed.
All ten compiled RT-DETR cases failed before compilation because the harness
still used PyTorch 2.10's `emulate_divison_rounding` config name.

[PyTorch 2.11's config](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_inductor/config.py)
moves the same eager division-rounding policy to
`eager_numerics.division_rounding`. The harness now selects the available name
for both candidate and compiled HF reference. The precision policy and numerical
tolerances are unchanged. The Nix binary is unchanged; validation harness
revisions and hashes are recorded independently from the original Nix build.

The [suite summary](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35282647966/kernel-hub-summary.json),
[Nix identity](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35282647966/NIX_BUILD.json),
[Phase 1 log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35282647966/phase1.log), and
[Pod deletion](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35282647966/runpod-state.json) are archived with all per-case
reports. Debug archives remain in the GHA artifact; their checksums are in
[debug-artifacts.json](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35282647966/debug-artifacts.json).

[Run 35290373885](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35290373885)
then passed Phase 1 and 19/20 RT-DETR cases. FP16 AMP compiled training failed
one element of `model.decoder.layers.0.self_attn.o_proj.bias`: absolute difference
`0.0517578125`, relative difference `0.0837283`, with unchanged `atol=0.02` and
`rtol=0.05`. The tensor's overall maximum absolute difference was `0.171875`.
Candidate eager versus HF eager matched that gradient, logits and loss exactly;
candidate compiled versus its own eager execution showed the same bias-gradient
discrepancy. Raw output/proposal differences also include query permutations,
which the harness handles only through the documented proposal identity check.

This run compared compiled FP16 AMP to eager HF, unlike the BF16 AMP protocol.
The suite now applies the same compiled-to-compiled reference adaptation to
both AMP precisions, retaining all raw eager/compiled diagnostics and unchanged
tolerances. FP16 AMP also saves replay fixtures/graphs. This distinguishes
kernel replacement parity under the same compiler from whole-model
eager-versus-compiled numerical drift; it does not claim that drift is absent.

The [19/20 summary](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35290373885/kernel-hub-summary.json),
[FP16 AMP comparisons](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35290373885/e2e-fp16-amp.json), and
[verified Pod deletion](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/kernel-hub-nix/run-35290373885/runpod-state.json) remain archived.

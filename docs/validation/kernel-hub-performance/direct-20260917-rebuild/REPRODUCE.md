# Same-toolchain HF rebuild and metadata-check ablation

This follows the [production-fix validation](../direct-20260917-fix/README.md).
No production source is changed by this experiment. The candidate is the same
working-tree snapshot on base commit `2d604d2d9e2871ce2dc957278bbdae211e4b3fa8`;
its exact contents are recorded in [candidate-sources.json](candidate-sources.json).

The source provenance of the rebuilt HF kernel is recorded in
[hf-source-provenance.json](hf-source-provenance.json). The official CUDA files
are unchanged since their initial addition to `huggingface/kernels-community`.
Their Git blobs match the fixed source snapshot and the public Python wrapper
matches the pinned Hub artifact byte for byte. The older binary metadata does
not record its original build commit; a later metadata-signing commit is not
treated as the original build commit. The published baseline is still loaded
at exact Hub revision `abfd4042216fa4f84c9c5c4e3e844a3143c70ad5`, with binary
SHA256 `1a5053e022f3e5840ca9f40d361b5356ef0beeefdd8686029378dd4a7284b959`.

Use a fresh CUDA 12.6 development environment with an L4 and `/workspace/ci`.
Unpack the evidence archive into a scratch directory and verify its embedded
SHA256 manifest. Clone this repository at the candidate's base commit into
`/workspace/ci/source`, create `/workspace/ci/results`, and copy the archive's
`source/diagnostics` there. The following scripts, archived with their exact
contents, reproduce the build and measurement sequence:

1. `bash source/diagnostics/pod_setup.sh`: install PyTorch 2.10.0+cu126 and kernels
   0.16.0 into `/workspace/ci/profile-env`; record all installed packages.
2. `bash source/diagnostics/build_candidate.sh`: unpack the candidate snapshot,
   build/install its normal wheel, and check the actual imported extension.
3. Unpack `source/diagnostics/hf-source.tar.gz` into `/workspace/ci`, producing
   `/workspace/ci/hf-source`. Set `PATH=/workspace/ci/profile-env/bin:$PATH`,
   `CUDA_HOME=/usr/local/cuda`, `TORCH_CUDA_ARCH_LIST=8.9`, and `MAX_JOBS=2`.
4. Run `python source/diagnostics/build_rebuilt.py`. The same
   `torch.utils.cpp_extension.load` compiler, `-O3`, C++17, architecture and
   minimal binding are used for HF, candidate and unchecked candidate. The
   source-specific `im2col_step` parameter type is preserved. No fast-math or
   lineinfo flags are added. Compiler hashes/versions, complete Ninja commands,
   source hashes before/after building and extension hashes are recorded.
5. Run `bash source/diagnostics/benchmark_rebuilt.sh` with no concurrent GPU
   work or builds. This compares six backends, both production shapes, two
   dtypes, two modes and seven shuffled repetitions. The CUDA Graph and eager
   measurements share a dedicated nondefault stream. All native backends share
   the same diagnostic autograd wrapper and FP32-compute policy.
6. Run `bash source/diagnostics/profile_rebuilt.sh` separately for fresh-process
   encoder FP32 forward traces and exact-binary sm_89 SASS/resource extraction.
7. Run `python source/diagnostics/build_followup.py`, then
   `bash source/diagnostics/benchmark_followup.sh`. These separately remove
   the unchecked candidate's forward launch bound, substitute HF's loop
   arithmetic, or change HF's forward output allocation from zeros to empty.
   The original build manifests are retained. A combined manifest loads all
   controls into a fresh encoder-forward comparison; extra traces and binary
   dumps are taken after timing. These controls are independent changes,
   not a combined production patch. In particular, the HF loop macro is
   safe only for the benchmark's bounded int32 inputs, not the int64 fallback.
8. Run `python source/diagnostics/pack_rebuilt.py` to verify completion, numeric
   gates, trace counts and binary identities, then archive the evidence. Verify
   the downloaded archive before deleting the Pod.

`candidate_unchecked` modifies only the copied `valid_spatial_level` helper to
return true. This lets compilation remove its device comparisons, assertion
and early-return checks. Host checks for dtype/device/layout/indexing remain.
Only fixed valid inputs with metadata and spans checked by the Python harness
are admitted; every backend must also pass the independent FP64 output and all
three gradient comparisons before timing. Do not run the harness with Python
assertions disabled. This is not an unchecked public API or a supported mode
for malformed metadata.

The main reports retain all timing samples and repetition medians. Timing
differences between HF's published artifact and its rebuild include any
unreproduced artifact-build settings; they should not be attributed solely to
one compiler option. CUDA Graph results largely remove Python dispatch, but
include output initialization, all captured kernels and replay gaps.

The check itself requires positive level height/width, nonnegative start, and
`start + height * width <= S`, before accessing flattened features. It is an
input range check, distinct from the numerical test oracle. PyTorch's
`CUDA_KERNEL_ASSERT` remains active under `NDEBUG`; the additional boolean
return guard would also retain the comparisons if only assertion reporting
were disabled. A public runtime opt-out could instead choose separate checked
and unchecked kernel template instantiations on the host, with that choice
carried into backward. Such an API is not implemented in this experiment.

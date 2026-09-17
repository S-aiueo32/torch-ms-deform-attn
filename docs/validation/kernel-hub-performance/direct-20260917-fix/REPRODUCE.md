# Reproducing the production-fix comparison

Use a CUDA 12.6 development environment on an sm_89 L4. The recorded run used
PyTorch 2.10.0+cu126, Python 3.11.13, kernels 0.16.0 and driver 595.91.07. All
GPU work was performed through direct SSH to Runpod; no GHA workflow was run.
The scripts use `/workspace/ci` as their diagnostic workspace.

1. Verify `fix-evidence.tar.gz` against `evidence-verification.json`, then unpack
   into a scratch directory. Verify every member listed in
   `results/evidence-manifest.json` before using it. This archive contains source,
   logs and traces, not executable extension binaries or credentials.
2. In a fresh `/workspace/ci`, create `results/` and clone this repository into
   `source/` at `2d604d2d9e2871ce2dc957278bbdae211e4b3fa8`. Copy the archive's
   `source/diagnostics/` directory into that clone. Keep `before/` absent until
   the next step; the setup script creates the unchanged baseline there.
3. Run `bash source/diagnostics/pod_setup.sh`. It copies the baseline, installs
   the matching PyTorch/dependencies, and builds the old extension with only an
   isolated operator namespace. The CPU baseline build is serial; GPU code and
   CUDA compiler flags are unchanged. Namespace-only binding edits and build
   commands are recorded in `before/diagnostic-build/`.
4. Run `bash source/diagnostics/build_candidate.sh`. This overlays the exact
   candidate source archive, builds and installs its CUDA/OpenMP wheel, and
   checks the imported extension and indexing selector. `candidate-sources.json`
   identifies the uncommitted source contents; the base revision field alone
   does not identify this candidate.
5. Run `bash source/diagnostics/validate_candidate.sh` for the regression suite
   and all four sanitizer modes. Separately run the forced-fallback and actual
   adapter checks using the commands below.
6. With no concurrent builds, tests or profiling, run
   `bash source/diagnostics/benchmark_candidate.sh`. This performs seven
   randomized repetitions for both production shapes and dtypes, plus the tiny
   host-sensitive case. The old/new public APIs, direct-native controls and
   pinned HF are qualified against FP64 output and all three gradients first.
7. Run the focused confirmation in a new process, then run
   `bash source/diagnostics/profile_candidate.sh`. Profiling is deliberately
   separate from ordinary timing. The latter also disassembles the exact
   measured sm_89 binaries and checks their hashes.
8. Run `python source/diagnostics/pack_results.py` to require successful
   correctness/trace-count gates and archive the evidence. Archive/download
   and verify the results before deleting the rented Pod.

The diagnostic Python environment is `/workspace/ci/profile-env/bin/python`.
Set `PATH=/workspace/ci/profile-env/bin:$PATH`, `MAX_JOBS=2`,
`TORCH_CUDA_ARCH_LIST=8.9` and `CUDA_HOME=/usr/local/cuda`. From
`/workspace/ci/source`, the supplementary commands are:

```sh
python diagnostics/fallback_check.py
python diagnostics/adapter_check.py \
  --revision 2d604d2d9e2871ce2dc957278bbdae211e4b3fa8
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python diagnostics/confirm_candidate.py \
  --output-dir /workspace/ci/results/confirmation
```

The fallback and adapter build/test invocations, along with the generated build
inputs, are retained in `results/fallback/` and `results/adapter/`. They build
separate extensions and do not replace the installed candidate. The fallback
changes only the index selector to force int64 templates on small inputs. The
adapter compiles the repository's actual `torch_binding.cpp`, shared dispatcher
and CUDA source with a local builder registration header; it does not reproduce
the entire HF loader or Nix/CMake build pipeline.

The confirmation alternates backends after ten calls and retains every paired
round, with six epochs per case. It pins threads to two distinct physical cores
selected using a short `/proc/stat` load sample and limits PyTorch threads to
one. It records affinity, clock samples and cgroup counters. These absolute
times should not be mixed with the main unpinned measurement as though their
measurement conditions were identical.

For the local metadata equivalence check, compile
`scripts/metadata_guard_check.cpp` using C++17 and
`-fsanitize=undefined -fno-sanitize-recover=all`, then execute it. It checks the
actual helper against the original predicate in 120,250 cases, with CUDA
assertions disabled to verify the safe false-return path. CPU/Metal build and
test provenance is documented in [VALIDATION.md](VALIDATION.md).

Exact packages, environment, commands, source hashes and measured binary hashes
are kept in the archive. Reproduction on another host may change latency;
compare implementations within the same process and inspect repeat variation.

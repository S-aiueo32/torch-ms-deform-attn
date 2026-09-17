# Run CUDA checks on Runpod

[Back to README](../README.md) · [Development and CI](development.md) · [GPU benchmarks](benchmarks.md#gpu-benchmarks)

The **CUDA correctness** workflow (`cuda.yml`) rents one Runpod GPU, builds and
tests the installed CUDA wheel, uploads logs and distributions, and deletes the
Pod. Start it from GitHub Actions with **Run workflow**. The default configuration
is one L4 on Secure Cloud, on-demand, with a 45-minute controller deadline
and a $0.50/hour compute-price limit.

An ordinary GitHub-hosted Ubuntu job controls the Pod over SSH. This works with
a personally owned repository and GitHub Pro. It does not register a persistent
self-hosted Actions runner: no GitHub runner-administration token is needed.

## Account setup

1. Create a Runpod account and add prepaid credits in its Billing page.
2. Create a dedicated API key in Runpod Settings with access to read GPU catalog
   information and create, read, and delete Pods. A read-only or Serverless-only
   key cannot run this workflow.
3. Save the key as this repository's Actions secret **`RUNPOD_API_KEY`**. Use the
   GitHub settings page or the interactive command below; do not put the key in
   source code, workflow inputs, shell arguments, or chat.

   ```bash
   gh secret set RUNPOD_API_KEY --repo S-aiueo32/torch-ms-deform-attn
   ```

The workflow supplies the GPU selection, pinned CUDA development image, temporary
disk, SSH server configuration, and a fresh SSH key pair on each run. It creates
no network volumes, saved templates, or account-wide SSH keys. Runpod API
credentials stay on the GitHub-hosted controller and are not sent to the GPU.
The Pod's SSH host key is also generated for the run and pinned by the controller.

First dispatch `operation=check`. This checks API access and the GPU price without
creating or modifying a Pod. It does not prove that credit or quota is sufficient
for a subsequent deployment.

If creation returns HTTP 402, check prepaid credits and spending limits in Runpod
Billing before dispatching again. HTTP 401/403 indicates an API key or permission
problem. An unavailable GPU is rejected before a Pod is requested; choose another
listed model within your accepted price limit or retry when capacity returns.

```bash
gh workflow run cuda.yml --repo S-aiueo32/torch-ms-deform-attn \
  --ref main --field operation=check
```

## Run tests

```bash
gh workflow run cuda.yml --repo S-aiueo32/torch-ms-deform-attn \
  --ref main \
  --field operation=test --field sanitizer=none \
  --field gpu='NVIDIA L4' --field max_hourly_usd=0.50 \
  --field timeout_minutes=45
```

Select another branch with `--ref` when needed. The checkout's exact commit is transferred via
`git archive`; GitHub credentials and the `.git` directory are excluded.
`scripts/run_cuda_checks.sh` builds an sdist and wheel in an isolated environment,
installs the wheel, then executes the complete CPU/CUDA test suite outside the
checkout. Select `memcheck`, `racecheck`, `synccheck`, `initcheck`, or `all` to additionally
run positive reduction, partial-batch, padding, offset and collision cases under
Compute Sanitizer. `all` reuses one wheel/Pod for all four tools.

Logs and built distributions appear in the run's `cuda-runpod-*` artifact, retained
for 90 days. A failed remote test fails the Actions job. Each new run rents a
fresh Pod; the workflow never substitutes a more expensive GPU model by itself.
Only one CUDA workflow runs at a time.

Use `torch_version=2.4.0` for the minimum supported PyTorch with CUDA 12.4.
The other selectable pairs are 2.5.1/CUDA 12.4 and 2.7.1, 2.8.0 or 2.14.0 with
CUDA 12.6. The local controller accepts the same selection through
`--torch-version`. Validation uses the selected container's `python3`; override
it with `CUDA_CHECKS_PYTHON` when invoking `run_cuda_checks.sh` directly.

Recorded environments and outcomes are in the [validation index](validation/README.md).

## Run the same checks from a local controller

Set `RUNPOD_API_KEY` in the local controller environment. It is never transferred
to the GPU. Use a unique numeric run identifier and fresh, separate directories:

```bash
python scripts/runpod_ci.py run \
  --repository OWNER/REPO --repository-id NUMERIC_REPOSITORY_ID \
  --run-id UNIQUE_NUMERIC_ID --attempt 1 --source . \
  --state-dir /tmp/msda-state-UNIQUE_ID --output-dir /tmp/msda-results-UNIQUE_ID \
  --gpu L4 --sanitizer memcheck --max-hourly-usd 0.50 --timeout-minutes 60
```

This uses the same source-archive, installed-wheel tests, artifact collection and
verified Pod deletion as Actions. Preserve the controller log and final state
alongside the output artifacts. Do not copy the temporary SSH private keys into
release evidence. The source SHA is resolved once before archiving, so later
checkout changes do not change the code being tested.

### Validate several Python versions on one Pod

The local controller can reuse a Pod for up to nine pairs that share a CUDA
toolkit. Select the container with `--torch-version`, then list the pairs:

```bash
python scripts/runpod_ci.py run \
  --repository OWNER/REPO --repository-id NUMERIC_REPOSITORY_ID \
  --run-id UNIQUE_NUMERIC_ID --attempt 1 --source . \
  --state-dir /tmp/msda-matrix-state --output-dir /tmp/msda-matrix-results \
  --gpu L4 --max-hourly-usd 0.50 --timeout-minutes 60 \
  --torch-version 2.14.0 --matrix-cases 2.9.1:3.13 2.14.0:3.14
```

Matrix mode provisions standard, GIL-enabled managed CPython with uv, runs the
full CUDA installed-wheel suite, then builds a fresh CPU-only wheel and runs its
full suite too. Every pair has fresh build and virtual-environment directories.
Failures do not prevent later pairs from running, but any failure makes the
controller fail. The normal ownership, price checks, deadline, artifact
collection and verified Pod deletion still apply. Up to nine pairs share the
same deadline; this is not nine separate 60-minute allowances.

Use the 2.5.1 container for PyTorch 2.4.0/2.5.0/2.5.1 with CUDA 12.4, and the
2.14.0 container for the listed 2.7–2.14 versions with CUDA 12.6. Matrix mode
requires one GPU, `sanitizer=none` and no benchmark. Its compatibility evidence
does not add sanitizer or multi-GPU coverage.

Artifacts are prefixed `torch-VERSION-python-VERSION-`. Strip that prefix into
a separate directory per pair before running `verify_cuda_release.py` against
the archived source SHA. `cpu-tests.json` records the separate CPU-only suite;
`matrix-summary.json` records each attempted pair's exit status. Missing reports
mean the corresponding validation phase was not reached, even when wheels were
downloaded or built successfully. Preserve failed attempts as well as successes.

## Kernel Hub and Transformers workload

The same controller can build the experimental Kernel Hub distribution and run
its operator tests plus RT-DETR E2E checks:

```bash
gh workflow run cuda.yml --repo S-aiueo32/torch-ms-deform-attn \
  --ref feat/kernel-hub-transformers-e2e \
  --field workload=kernel-hub --field operation=test \
  --field torch_version=2.14.0 --field sanitizer=none \
  --field gpu='NVIDIA L4' --field max_hourly_usd=0.50 \
  --field timeout_minutes=60
```

Here `torch_version` selects the CUDA 12.6 container; the workload installs its
own pinned PyTorch 2.10.0 / torchvision 0.25.0 environment. It requires one GPU,
no sanitizer, and no benchmark or support-matrix options. The local controller
equivalent is `--workload kernel-hub --torch-version 2.14.0`.

`scripts/run_kernel_hub_checks.sh` uses the official kernel-builder 0.16.0 local
development build (`create-pyproject`, then CMake build and `local_install`).
The workflow caches the pinned builder on the CPU runner and generates the
project before renting a GPU. `--prepared-kernel` transfers that generated tree
with the source archive after checking its upstream revision. The GPU host only
installs runtime/build dependencies, compiles CUDA, and executes the tests. This is a
real native adapter build and loader run, but not a Nix release build or proof
of distribution portability. Builder artifacts are not uploaded to HF.

The existing HF kernel revision is pinned in that script. Collected artifacts
include source and binary hashes, builder metadata, individual test logs/E2E
JSON reports, the overall `kernel-hub-summary.json`, and `runpod-state.json`.
Require all requested runs to pass and the Pod state to be `deleted`. These
results do not replace core CUDA release-validation evidence.

## Cleanup and costs

The controller deletes the Pod in a `finally` block, handles cancellation signals,
and the workflow retries deletion in an `always()` step. Pod names and metadata
bind them to the repository ID, run ID, and attempt, including recovery from a
creation request whose response was lost. Cleanup never selects arbitrary
account Pods by GPU type or a short name prefix alone.

For recovery after a lost GitHub runner, merge `runpod-cleanup.yml` and the scripts
into the default branch, then enable this repository variable:

```bash
gh variable set RUNPOD_CLEANUP_ENABLED --body true \
  --repo S-aiueo32/torch-ms-deform-attn
```

This recovery workflow runs after **CUDA correctness** or **CUDA benchmark**
finishes and once per hour.
It deletes Pods for the finished run and this repository's marked Pods older than
two hours. It can also be started manually. It executes trusted default-branch
code. Recovery triggers require the workflow to exist on the default branch;
they are not active during branch-only testing. Scheduled recovery consumes
GitHub-hosted runner minutes even when no GPU exists.

The price limit checks the current catalog quote before creation and the assigned
price immediately afterward. Runpod's create API has no atomic maximum-price
parameter: an over-limit assignment is deleted immediately but can incur a small
charge. Compute limits exclude disk charges and GitHub-hosted controller usage.
Creation, image download, and initialization can also consume paid time.

This is not a provider-enforced spending cap. GitHub/API outages can delay cleanup.
The in-container process timeout alone does not stop Runpod billing. If recovery
fails, delete the run's named Pod from the Runpod console. No persistent network
volume is created, so deleting the Pod removes its allocated storage as well.

References: [Runpod API keys](https://docs.runpod.io/get-started/api-keys),
[Pod creation](https://docs.runpod.io/api-reference-v2/pods/create-a-pod),
[GPU catalog](https://docs.runpod.io/api-reference-v2/catalog/get-a-gpu-type),
[SSH](https://docs.runpod.io/pods/configuration/use-ssh),
[Pod pricing](https://docs.runpod.io/pods/pricing).

## Required release validation

Before publishing any release, the maintainer must:

1. Resolve the release tag to a full source SHA. Run **CUDA correctness** on that
   commit/branch, or use the local controller with a checkout at that SHA.
   Ordinary **CUDA package build** is build-only evidence.
2. Require a successful test run. For Actions, its `headSha` must match that SHA.
   For a local controller, retain the source archive and controller log, require
   exit status zero and `phase: deleted` in its state file, and verify the SHA in
   the output JSON. `operation=check` does not run tests.
3. Download its `cuda-runpod-RUN-ATTEMPT` artifact (or use the local controller's
   output `artifacts` directory) and run
   `python scripts/verify_cuda_release.py --sha FULL_SHA --evidence PATH/TO/artifacts`.
   Missing evidence, a different SHA, a failed suite or unexpected skips reject
   the release. Only the explicitly named CPU-only-wheel and two-GPU tests may
   skip in this single-GPU CUDA-build configuration.
4. Attach `cuda-tests.json`, `cuda-completion.json`, `cuda-checks.log` and
   `nvidia-smi.txt` to the GitHub release **before publishing it**, and include
   the source SHA, Actions run URL or local run identifier, GPU/toolchain, counts and skip reasons in its
   release notes. These release assets are the long-term record; the 90-day
   Actions artifact is only temporary storage. Repeat for each claimed CUDA pair.

The JSON records hardware, driver, Python/PyTorch/toolkit and build flags. A
completion record is written only after the requested sanitizer also succeeds.
The [PyPI publish workflow](../.github/workflows/publish.yml) downloads these
four release assets and runs the verifier before building the distribution.
The verifier validates contents, not provenance; download
only from the trusted successful workflow or local controller run and check its
SHA and completion status.

The local controller path and evidence verifier were exercised on both L4
PyTorch 2.5.1/CUDA 12.4 and 2.7.1/CUDA 12.6 at source
`c29ca3640b8a7a0cf4fc9907d40ccde4d42f9045`. See the
[archived reports, logs and cleanup records](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/fedcad0064f3cea79146be9ca181c97f53aeeda8/docs/validation/2026-09-14-p1/README.md).

## Sanitizer evidence and scope

Use `sanitizer=all` for release-candidate validation. Each tool must return zero
with `--error-exitcode 1`; inspect the tool summary in its individual
`sanitizer-TOOL.log`. Archive those logs with the exact source SHA and tool version
alongside the [required release evidence](#required-release-validation). A historical memcheck pass is
not racecheck/synccheck/initcheck evidence for another revision. All four tools passed on both PyTorch 2.5.1/CUDA 12.4 and 2.7.1/CUDA 12.6
with L4; see the [kernel correctness evidence](validation/kernel-correctness/README.md).

Only named positive sampling/reduction tests run under sanitizers. Tests that
intentionally trigger device assertions still run in isolated subprocesses in
the normal suite, separate from sanitizer runs.

Racecheck diagnoses supported shared-memory hazards; it does not prove absence
of global-memory races. Synccheck diagnoses supported synchronization misuse.
Initcheck covers uninitialized device global-memory reads under its default
scope; it is not a complete memory-safety proof. Memcheck complements these
checks. Consult the installed Compute Sanitizer version's documentation and do
not infer guarantees for memory types or execution paths the tool did not check.


## Additional GPU checks

The local controller accepts `--gpu-count 2` for the noncurrent-device test.
`--max-hourly-usd` is the **total Pod compute rate**; its catalog preflight
multiplies the per-GPU rate by the requested count and validates the returned
Pod rate again. The standard workflows still default to one GPU. A two-GPU run
requires two visible devices and rejects a skipped noncurrent-device test.

After CUDA tests/optional benchmarks, `run_cuda_checks.sh` builds a fresh CPU-only
wheel from the same sdist on the GPU host, reinstalls it, and runs the CUDA-input
error test with zero skips. Results are in `cpu-only-gpu-tests.json`; this phase
does not replace the earlier CUDA-wheel evidence. The final completion marker is
written only after both phases pass. CUDA tests also print dynamic-compile graph
counts: metadata-value changes cannot recompile, seen shapes must reuse graphs,
and the unit-dimension call is explicitly identified.

The benchmark workload includes eager/compiled float32/fp16/bf16 and encoder Q,
with three repetitions. Use the standalone harness for additional batch/channel/
step sweeps. See [benchmark metrics](benchmarks.md#measurement-and-regression-policy)
and [pinned module compatibility](compatibility.md).

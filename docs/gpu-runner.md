# Run CUDA checks on Runpod

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
   gh secret set RUNPOD_API_KEY --repo S-aiueo32/torch-deform-attn
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
gh workflow run cuda.yml --repo S-aiueo32/torch-deform-attn \
  --ref codex/attention-performance-and-safety --field operation=check
```

## Run tests

```bash
gh workflow run cuda.yml --repo S-aiueo32/torch-deform-attn \
  --ref codex/attention-performance-and-safety \
  --field operation=test --field sanitizer=none \
  --field gpu='NVIDIA L4' --field max_hourly_usd=0.50 \
  --field timeout_minutes=45
```

After merging, use `--ref main`. The checkout's exact commit is transferred via
`git archive`; GitHub credentials and the `.git` directory are excluded.
`scripts/run_cuda_checks.sh` builds an sdist and wheel in an isolated environment,
installs the wheel, then executes the complete CPU/CUDA test suite outside the
checkout. Select `memcheck`, `racecheck`, or `synccheck` to additionally run the
kernel reduction and partial-batch tests under Compute Sanitizer.

Logs and built distributions appear in the run's `cuda-runpod-*` artifact, retained
for seven days. A failed remote test fails the Actions job. Each new run rents a
fresh Pod; the workflow never substitutes a more expensive GPU model by itself.
Only one CUDA workflow runs at a time.

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
  --repo S-aiueo32/torch-deform-attn
```

This recovery workflow runs after **CUDA correctness** finishes and once per hour.
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

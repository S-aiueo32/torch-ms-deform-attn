# Minimum PyTorch compatibility

## Objective and conditions

The declared minimum is tested on Linux CPU and CUDA 12.4. The CUDA development
image is pinned to
`pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel@sha256:e96c6896ecfbb50d89c87bf94110206ef444f27268c5f72201eb29fba9c90331`.

## Results

Tested source: `4863ba7ffa23af96766541ea8d780db3e9db7ac1`.
Execution date: 2026-09-17.

| Environment | Result |
| --- | --- |
| Python 3.10 / PyTorch 2.4.0 / serial | 66 tests: 39 passed, 27 expected skips; 7 build-policy tests passed |
| Python 3.11 / PyTorch 2.4.0 / OpenMP | 66 tests: 40 passed, 26 expected GPU skips; 7 build-policy tests passed |
| Python 3.11.9 / PyTorch 2.4.0+cu124 / NVIDIA L4 | 66 tests: 64 passed, 2 expected skips |
| Fresh CPU-only wheel on the same GPU host | 1 test passed, no skips |
| Compute Sanitizer 2024.1.1.0 | memcheck, racecheck, synccheck, initcheck: 4 positive tests each; zero errors, racecheck zero warnings/hazards |
| CUDA build without a GPU | sdist-to-wheel build and installed-wheel tests passed: 40 passed, 26 GPU skips |

The serial build additionally skips the parallel-worker test because it has no
parallel backend. Both jobs build an sdist, build and install its wheel, and run
the suite outside the checkout. CPU results include AMP, explicit low-precision
inputs, FakeTensor/opcheck, dynamic compilation and upstream module parity.
See the [CPU workflow](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35168376104),
[serial log](cpu-py310-serial.log) and [OpenMP log](cpu-py311-openmp.log).

The [GPU workflow](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35168376014)
used driver 580.159.04 and CUDA toolkit 12.4.131. The two expected skips were the
two-GPU/noncurrent-device test and the CPU-only-wheel test during the CUDA-wheel
phase. The latter then passed in its separate CPU-only-wheel phase. There were
no unexpected skips. CUDA checks include forward/backward, metadata rejection,
AMP training, explicit fp16/bf16 inputs, dynamic compilation, opcheck, and pinned
upstream module parity.

- [Test report](cuda/cuda-tests.json), [completion marker](cuda/cuda-completion.json),
  [CPU-only wheel report](cuda/cpu-only-gpu-tests.json), [full test log](cuda/cuda-checks.log).
- [memcheck](cuda/sanitizer-memcheck.log), [racecheck](cuda/sanitizer-racecheck.log),
  [synccheck](cuda/sanitizer-synccheck.log), [initcheck](cuda/sanitizer-initcheck.log).
- [Workflow log including verified Pod deletion](cuda/workflow.log),
  [environment](cuda/nvidia-smi.txt), [artifact checksums](cuda/sha256.json).
- [CUDA build workflow](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35168376086)
  and [2.4 build log](cuda-build.log). The existing 2.5.1/12.4 and 2.7.1/12.6
  build jobs also passed. CPU CI, including rebuilding across PyTorch versions,
  and [Lint](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35168376050)
  passed. Controller/evidence tests: 42 passed.

`scripts/verify_cuda_release.py` accepted the archived evidence for the source
SHA above. Sanitizers exercise positive sampling/reduction cases; their scope
is described in the [GPU guide](../../gpu-runner.md#sanitizer-evidence-and-scope).
The Actions artifact retains the sdist and wheel; binaries are not committed.
Downloaded CUDA artifacts are preserved verbatim and checksummed. Actions job
logs have ANSI escape sequences and trailing whitespace removed for readability.

## Evidence boundaries

The results above apply to the stated source SHA. The
[compile diagnostic archive](initial-cuda-failure.log) belongs to source
`1107b8a054ecf3aa8ab08403909a39c0ab21c903` and supplies no runtime or sanitizer
evidence. PyTorch 2.4 device checks use `CUDA_KERNEL_ASSERT`; subprocess tests
verify forward/backward metadata assertions and their diagnostic text.

Both allocated L4 Pods were verified deleted. Each was quoted at $0.49/hour
with a 60-minute deadline, for a $0.98 combined full-deadline compute estimate
excluding disk charges. This is not a billing statement.

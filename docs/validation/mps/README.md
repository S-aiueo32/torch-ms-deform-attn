# Apple Silicon MPS validation

Validated on Apple M2, macOS 26.6.2 arm64, Python 3.11.16 on 2026-09-17.
These results describe the current checkout, not the published `0.1.0rc2`.
The final source snapshot is recorded in [source-sha256.json](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/mps/source-sha256.json).
Test and benchmark JSON files also identify the loaded extension binary by SHA-256.

## Correctness and builds

| PyTorch | Tests run | Passed | Skipped | Failures/errors | Evidence |
| --- | ---: | ---: | ---: | --- | --- |
| 2.4.0 | 75 | 48 | 27 | 0 / 0 | [log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/mps/pytorch-2.4.0-tests.log), [environment](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/mps/pytorch-2.4.0-tests.json) |
| 2.5.1 | 75 | 49 | 26 | 0 / 0 | [log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/mps/pytorch-2.5.1-tests.log), [environment](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/mps/pytorch-2.5.1-tests.json) |
| 2.14.0 | 75 | 49 | 26 | 0 / 0 | [log](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/mps/pytorch-2.14.0-tests.log), [environment](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/mps/pytorch-2.14.0-tests.json) |

The CUDA tests are skipped because this host has no CUDA GPU. PyTorch 2.4 also
skips MPS autocast, which that release does not implement. MPS forward/backward,
explicit float16/bfloat16 inputs, noncontiguous inputs, storage offsets,
overlapping levels, boundary coordinates, invalid metadata, deterministic-mode
handling, and higher-order-gradient rejection are exercised on the actual GPU.
The numerical oracle uses the CPU float64 operator with matching incoming gradients.

PyTorch 2.4/2.14 were tested through separately installed local wheels; 2.5.1 used
the editable checkout. Both Ninja and distutils compiled the Metal bridge.
The 2.5.1 sdist rebuilt into an arm64 wheel with the embedded shaders; source
archives include all three `csrc/mps/` files. `FORCE_CPU=1 USE_NINJA=0` also built
successfully and passed a CPU forward/backward check with `_C.with_mps == False`.
Build-policy tests (8), Ruff, type checking, cpplint, and the documented clang-tidy
checks passed; clang-tidy diagnostics from dependency headers were excluded.

Package tests contain no Grounding DINO fixture or model dependency. Integration
was checked in the separate local GroundingDINO checkout, where the MPS branch
was removed and module output/gradient/optimizer tests passed on CPU and MPS.
Full detector accuracy and end-to-end inference are outside this validation.

## Performance

Eight synthetic four-level encoder/decoder shapes, float32, 5 warmup calls and
20 timed calls per backend/mode. Latency includes host metadata validation and
GPU synchronization. The first call (including lazy shader compilation and
pipeline creation) is recorded separately. These are local operator measurements,
not whole-model speedups or controlled fleet-wide benchmarks.

PyTorch 2.5.1 forward speedups range from 3.67× to 5.35×. Its MPS `grid_sample`
backward is unimplemented, so no reference backward speedup is reported.
See [2.5.1 raw measurements](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/mps/pytorch-2.5.1-benchmark.json).

PyTorch 2.14.0 has a faster reference and supports its backward. Forward speedups
range from 1.43× to 2.27×. Forward+backward ranges from 0.71× to 1.29×:
the dedicated kernel is slower on the encoder cases and one decoder case.
No automatic fallback is selected from these measurements.

| Image | Batch | Stage | Metal forward ms | Reference forward ms | Metal forward+backward ms | Reference forward+backward ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 800×800 | 1 | encoder | 32.33 | 51.32 | 230.96 | 217.39 |
| 800×800 | 1 | decoder | 2.53 | 4.07 | 13.98 | 18.09 |
| 800×800 | 2 | encoder | 64.99 | 93.23 | 485.33 | 344.25 |
| 800×800 | 2 | decoder | 4.62 | 8.50 | 30.67 | 28.52 |
| 800×1344 | 1 | encoder | 54.56 | 86.90 | 403.59 | 373.06 |
| 800×1344 | 1 | decoder | 2.49 | 4.46 | 16.77 | 19.81 |
| 800×1344 | 2 | encoder | 111.64 | 165.22 | 834.79 | 597.93 |
| 800×1344 | 2 | decoder | 4.57 | 10.39 | 30.63 | 33.41 |

See [2.14.0 raw measurements](https://github.com/S-aiueo32/torch-ms-deform-attn/blob/324030908cd66a625d6794a5fb0389228b257a5a/docs/validation/mps/pytorch-2.14.0-benchmark.json), including first-call
latency, minimum/maximum timings, and observed tensor memory. Memory is sampled
at PyTorch operator boundaries in a separate untimed run. It excludes allocator
cache and may miss intra-operator workspace; it is not an exact peak counter.

## Reproduce

Install the current source into an environment with the chosen PyTorch version
and an accessible Apple GPU. Do not enable `PYTORCH_ENABLE_MPS_FALLBACK`.

```bash
FORCE_MPS=1 python -m pip install --no-build-isolation .
python -c 'import torch; from torch_ms_deform_attn import _C; assert _C.with_mps and torch.backends.mps.is_available()'
python -m unittest discover -s tests -v
python -m unittest discover -s build_tests -v
python benchmarks/benchmark_mps.py --warmup 5 --iterations 20 --output benchmark-mps.json
```

The support floor is macOS 13.3 (14 for bfloat16), but these runs only validate
macOS 26.6.2 on M2. Other Apple GPUs/OS releases, MPS `torch.compile`, CUDA execution
of this change, and higher-order gradients are not validated here.

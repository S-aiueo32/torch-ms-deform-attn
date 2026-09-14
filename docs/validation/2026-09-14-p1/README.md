# P1 validation — 2026-09-14

Tested source: [`c29ca3640b8a7a0cf4fc9907d40ccde4d42f9045`](https://github.com/S-aiueo32/torch-ms-deform-attn/tree/c29ca3640b8a7a0cf4fc9907d40ccde4d42f9045).
This integration revision combines T01–T07 and T09 and the Ninja include-path fix.
It is retained by the `validation/p1-2026-09-14` tag; these results do not certify future
merge commits. Run validation again for the exact release source SHA.

All eight tasks ran together: one installed-wheel build per GPU/toolchain pair,
then the full suite and all four sanitizers against that wheel. The controller
ran locally, not through GitHub Actions. Both L4 Pods cost $0.49/hour at creation
and were verified deleted. This is the quoted compute rate, not a billing total.

| Environment | Build / tests |
| --- | --- |
| macOS arm64, Python 3.10.21, PyTorch 2.5.0 | CPU OpenMP build; 54 tests, 35 passed, 19 GPU skips |
| Same Python 3.10 environment, upgraded to PyTorch 2.7.1 | Forced extension rebuild in same checkout; 54 tests, 35 passed, 19 GPU skips |
| macOS arm64, Python 3.11.16, PyTorch 2.5.1 | CPU OpenMP build; 54 tests, 35 passed, 19 GPU skips |
| macOS arm64, Python 3.12.14, PyTorch 2.7.1 | sdist-to-wheel build, installed-wheel tests outside checkout; 54 tests, 35 passed, 19 GPU skips |
| Linux L4, PyTorch 2.5.1 / CUDA 12.4 | sdist-to-wheel build; 54 tests, 52 passed, 2 expected skips; all four sanitizers passed |
| Linux L4, PyTorch 2.7.1 / CUDA 12.6 | sdist-to-wheel build; 54 tests, 52 passed, 2 expected skips; all four sanitizers passed |

The two GPU-suite skips are the CPU-only-wheel-on-GPU test and the two-GPU test
(T08/P2), with exact IDs and reasons in each `cuda-tests.json`. Every sanitizer
runs four positive test methods, including the full reduction dtype/channel
subtest table and shared padding/offset/collision cases. Deliberate device
assertions execute only in subprocesses in the normal suite.

- [CUDA 12.4 report](cu124/cuda-tests.json), [complete log](cu124/cuda-checks.log), [cleanup](cu124/cleanup.json).
- [CUDA 12.6 report](cu126/cuda-tests.json), [complete log](cu126/cuda-checks.log), [cleanup](cu126/cleanup.json).
- Per-tool `sanitizer-*.log` files contain zero-error summaries; racecheck also reports zero warnings/hazards.
- `sha256.json` identifies downloaded artifacts, including the uncommitted binary wheel and source archive.
- Root CPU logs cover lower bounds and the same-environment upgrade/rebuild. Controller unit tests: 39 passed. Ruff and ty passed on the integrated source.

These checks establish only the tested paths and tool scopes. Racecheck covers
supported shared-memory hazards, synccheck supported synchronization misuse,
and initcheck default device global-memory initialization checks; they do not
prove absence of all races or all memory errors. The standalone Linux CPU-only
matrix rows were not run locally; GitHub Actions was unavailable for this task.

# Validation

Validation is organized by the compatibility or behavior being checked.
The [support matrix](../support.md) summarizes CPU and CUDA
status; the records below provide conditions, results and source-bound evidence.

| Verification target | Record | Coverage |
| --- | --- | --- |
| Python/PyTorch combinations | [Support matrix](python-matrix/README.md) | Each backend: 46 verified pairs, 3 known limitations, no unverified in-range cells |
| Minimum supported PyTorch | [PyTorch 2.4.0](minimum-pytorch/README.md) | CPU serial/OpenMP, GPU-free CUDA build, L4 runtime and four sanitizers |
| PyTorch version compatibility | [PyTorch 2.8–2.14](pytorch-versions/README.md) | Linux CPU versions; GPU-free builds and L4 sanitizers on 2.8.0 / 2.14.0 |
| Kernel correctness | [Correctness and sanitizers](kernel-correctness/README.md) | CPU builds, CUDA reduction/sampling, four sanitizers on 2.5.1 / 2.7.1 |
| Execution and measurement | [Execution, builds and benchmarks](execution-and-benchmarks/README.md) | Two-GPU behavior, dynamic compile, near-knot precision, CPU backends and benchmark grids |
| Kernel Hub / Transformers | [RT-DETR CPU and L4 validation](kernel-hub/README.md) | CUDA artifact built; initial 14/20 plus passing focused rechecks of six compiled failures under documented policies; FP16/BF16 operator differences traced to precision policy; final full matrix remains open; CPU fixture 20/20 |
| Kernel Hub benchmarks | [Phase 3 L4 measurements](kernel-hub-benchmarks/README.md) | Six implementations measured on L4; all common-policy correctness checks passed; fresh-process comparison identifies eager training slowdowns to investigate |
| MSDA dispatch overhead | [L4 performance investigation](kernel-hub-performance/README.md) | Direct C++ registration, native control and pinned previous-source comparison; precision policy unchanged |

## Reading the evidence

Results apply to the source SHA and environment recorded with each suite.
Test totals differ across source revisions. Expected skips are stated per
environment; failed suites and setup-only attempts do not establish full support.
Benchmark timings and emulated Docker runs are not native performance baselines.

Each record links to reports, logs, environment details and available checksum
manifests. Raw evidence filenames retain their run identifiers for traceability.
Binary artifacts are committed only where explicitly linked; other manifests
may also identify binaries retained outside the repository.

For reproduction, see [local CPU validation](../development.md),
[GPU validation](../gpu-runner.md), and [benchmark methodology](../benchmarks.md).
Release checks must target the exact release source SHA.

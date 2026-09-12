# CPU benchmarks

[Back to README](../README.md)

Measure CPU operator latency for the installed extension against the PyTorch
reference. Run from the repository root after [installing the package](installation.md).

## Run benchmarks

```bash
python benchmarks/benchmark_cpu.py --threads 1 --min-run-time 1 --json
python benchmarks/benchmark_cpu.py --threads 4 --min-run-time 1 --json
```

The harness checks outputs and all three gradients against the reference before
timing three synthetic shapes. It reports median forward and forward+backward
latency; speedup above 1 means C++ is faster. Check the
[CPU parallel backend](installation.md#configure-cpu-parallelism) before comparing
thread counts: a serial extension does not gain parallelism from `--threads`.

## Compare eager and compiled execution

```bash
python benchmarks/benchmark_compile.py --threads 1
```

This harness compares eager and Inductor (`fullgraph=True`) for both
implementations and emits JSON. Correctness checks, compilation, and warmup
precede timing. The reference uses static shape tuples to avoid graph breaks,
so its baseline differs from `benchmark_cpu.py`.

## Interpret results

Results measure synthetic CPU operators, not whole-model latency, peak memory,
or CUDA performance. Compare runs with the same PyTorch version, build settings,
shapes, and thread count. Exact shapes are defined in the
[eager harness](../benchmarks/benchmark_cpu.py) and
[compile harness](../benchmarks/benchmark_compile.py).

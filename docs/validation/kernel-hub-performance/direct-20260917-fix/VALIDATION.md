The native autograd change passed these local correctness checks on macOS 26.6.2 arm64. Every run asserted the imported Python package and `_C` extension came from the intended candidate directory.

| Device | PyTorch | Completed checks | Results |
|---|---|---|---|
| CPU | 2.4.0 | 10 passed; no skips | [JSON](cpu-autograd-torch240.json), [log](cpu-autograd-torch240.log) |
| CPU | 2.5.1 | 10 passed; no skips | [JSON](cpu-autograd-torch251.json), [log](cpu-autograd-torch251.log) |
| Apple Metal | 2.5.1 | 9 passed; no skips | [JSON](mps-autograd-torch251.json), [log](mps-autograd-torch251.log) |

CPU 2.5.1 imported the rebuilt extension and Python modules from `src/torch_ms_deform_attn`. CPU 2.4.0 used a separate build under `/private/tmp/msda-autograd24-lib`. The Metal run used another independent build with `FORCE_MPS=1` under `/private/tmp/msda-mps-autograd251-lib`; GPU availability and native MPS support were both confirmed outside the filesystem sandbox.

The CPU checks cover dispatcher opcheck, dynamic fullgraph AOT/Inductor execution, AMP, legacy calls, export, saved tensor hooks, and saved-input mutation detection. Higher-order differentiation is rejected both through the public forward call and when only the backward operator's `grad_output` requires gradients.

Compiled Autograd checks require an actual compiler invocation and compare all three gradients at two shapes and step values. Their assertions allow PyTorch versions to represent C++ autograd nodes differently in FX. The native function declares `is_traceable` because its entire backward state is held in the context's saved tensors and data. Regular AOT/export checks separately exercise the opaque operator contract.

The nine Metal tests cover output/gradient parity, FP16/BF16, autocast, channel sizes, boundaries and offsets, noncontiguous layouts, invalid inputs, near-knot sampling, FakeTensor, deterministic-mode behavior, and higher-order rejection. These are correctness checks; no Metal performance claim is made.

The [source audit](local-source-audit.json) confirms every recorded source hash matches the final local tree, and every imported Python file and native binary still matches its recorded hash. The result JSON files retain the actual import paths, platform, PyTorch version, and hashes. The PyTorch 2.4 export log includes an upstream tracing warning; all assertions passed.

The [recording script](scripts/validate_cpu_autograd.py) defaults to the repository in the current directory and accepts `--repo-root`. After building the extension with the matching PyTorch environment, run from the repository root:

```sh
rtk proxy env PYTHONPATH=src .venv/bin/python docs/validation/kernel-hub-performance/direct-20260917-fix/scripts/validate_cpu_autograd.py --output-prefix /private/tmp/cpu-autograd --expected-package-root src/torch_ms_deform_attn
```

For an independent build, set `PYTHONPATH` and `--expected-package-root` to that build's parent/package paths and use its matching Python environment. Add `--test-pattern test_mps.py` for the existing Metal suite, with GPU access enabled. The committed script adds a portable repository argument and formatting to the recorded harness; the test logic is unchanged.

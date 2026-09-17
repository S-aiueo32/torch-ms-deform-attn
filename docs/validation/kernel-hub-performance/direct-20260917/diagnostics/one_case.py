#!/usr/bin/env python3
"""Fresh-process profiler / Nsight Compute target for a single qualified case.

Example Nsight invocation (metadata capture may require GPU profiling privileges):
  ncu --profile-from-start off --section LaunchStats --section Occupancy \
      --export encoder-native --force-overwrite \
      python one_case.py --backend native_control --case encoder --mode forward \
      --output-dir results/ncu-native --ncu-mode
"""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

import torch
from probe import (
    CASES,
    backend_context,
    common_precision,
    comparison,
    evaluate,
    fingerprint,
    inputs,
    load_backends,
    oracle,
    profile,
    profile_python,
    runner,
)


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as exc:
        return {"command": command, "error": repr(exc)}


def resource_metadata():
    properties = torch.cuda.get_device_properties(0)
    names = (
        "name",
        "major",
        "minor",
        "total_memory",
        "multi_processor_count",
        "max_threads_per_multi_processor",
        "max_threads_per_block",
        "warp_size",
        "regs_per_block",
        "regs_per_multiprocessor",
        "shared_memory_per_block",
        "shared_memory_per_multiprocessor",
        "clock_rate",
        "memory_clock_rate",
        "memory_bus_width",
        "L2_cache_size",
        "is_multi_gpu_board",
        "is_integrated",
    )
    resources = {name: getattr(properties, name) for name in names if hasattr(properties, name)}
    resources["full_properties"] = str(properties)
    return resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--variants-dir", type=Path)
    parser.add_argument("--case", choices=CASES, default="encoder")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--mode", choices=["forward", "forward_backward"], default="forward")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--ncu-mode", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    # Match probe.py: one live stream for input creation, autograd and profiling.
    benchmark_stream = torch.cuda.Stream()
    torch.cuda.set_stream(benchmark_stream)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "pid": os.getpid(),
            "gpu": resource_metadata(),
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
        },
        "setup_errors": {},
        "metadata_notes": {
            "fresh_process": "One torch.profiler context maximum; avoids repeated Kineto initialization issues",
            "stream": "one dedicated nondefault stream for all inputs, autograd and profiling",
            "precision": "FP32 computation with explicit FP16 casts included for every backend",
            "qualification": "FP64 grid_sample output and all3 input gradients",
            "ncu": "CUDA profiler range includes exactly one complete operator iteration; only kernels in this range should be collected",
        },
        "script_sha256": fingerprint(__file__),
    }
    for package in ("kernels", "huggingface-hub", "torch"):
        try:
            report["environment"][package + "_package"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    report["environment"]["nvidia_smi"] = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,clocks.sm,clocks.mem,power.limit",
            "--format=csv",
        ]
    )
    report["environment"]["ncu_version"] = command_output(["ncu", "--version"])

    def save():
        destination = args.output_dir / "one-case.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)

    try:
        # Reuse provenance/loading semantics but avoid any unrelated HF downloads.
        load_args = argparse.Namespace(
            variants_dir=args.variants_dir, skip_hf=args.backend != "hf", backends=[args.backend]
        )
        backends = load_backends(load_args, report)
        if args.backend not in backends:
            raise RuntimeError(f"Backend {args.backend!r} failed loading: {report['setup_errors']}")
        dtype = getattr(torch, args.dtype)
        raw = backends[args.backend]
        fn = (
            raw
            if dtype == torch.float32 or args.backend.startswith("canonical")
            else common_precision(raw)
        )
        data, grad, sizes, scale = inputs(args.case, dtype)
        truth = oracle(data, grad, sizes)
        with backend_context(args.backend):
            report["correctness"] = comparison(evaluate(fn, data, grad), truth, dtype, scale)
        save()
        if not report["correctness"]["passed"]:
            raise RuntimeError("Backend did not pass independent FP64 qualification")
        del truth
        run = runner(fn, data, grad, args.mode)
        with backend_context(args.backend):
            for _ in range(args.warmup):
                run()
            torch.cuda.synchronize()
            report["memory_before_profile"] = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "free_and_total_bytes": torch.cuda.mem_get_info(),
            }
            torch.cuda.reset_peak_memory_stats()
            if args.ncu_mode:
                # ncu should be called with --profile-from-start off so oracle /
                # loader / warmup kernels cannot contaminate the operator range.
                torch.cuda.profiler.start()
                try:
                    with torch.cuda.nvtx.range("MSDA_ONE_QUALIFIED_ITERATION"):
                        result = run()
                        torch.cuda.synchronize()
                finally:
                    torch.cuda.profiler.stop()
                report["ncu_range"] = {
                    "iterations": 1,
                    "nvtx_name": "MSDA_ONE_QUALIFIED_ITERATION",
                    "outputs_held_until_after_synchronize": True,
                }
                del result
            else:
                report["profile"] = profile(
                    run, args.output_dir / "trace.json", args.profile_iterations, mode=args.mode
                )
                if args.case == "decoder" and args.mode == "forward_backward":
                    report["python_profile"] = profile_python(
                        run, args.output_dir / "python.pstats"
                    )
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["completed"] = True
        save()
        print(
            json.dumps(
                {
                    "output": str(args.output_dir / "one-case.json"),
                    "cuda_us": report.get("profile", {}).get("cuda_total_us_per_iteration"),
                    "warning": report.get("profile", {}).get("warning"),
                }
            ),
            flush=True,
        )
    except Exception:
        report["error"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()

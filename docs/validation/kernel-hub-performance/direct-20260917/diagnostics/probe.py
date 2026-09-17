#!/usr/bin/env python3
"""Standalone PR16 CUDA diagnosis. All modifications live in this process.

The optional variants-dir/manifest.json contains a variants list, each item with
name, module_name, path (.so), and optional forward/backward symbol names.
Variants with no path must be importable. Metadata is preserved in the report.
All variants must accept the canonical positional native extension interface.
"""

import argparse
import cProfile
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import pstats
import random
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch

from torch_ms_deform_attn import _C, ms_deform_attn, ms_deform_attn_core_pytorch

CASES = {
    "decoder": (2, 300, 8, 32, ((64, 64), (32, 32), (16, 16), (8, 8))),
    "encoder": (2, 5440, 8, 32, ((64, 64), (32, 32), (16, 16), (8, 8))),
}
HF_REVISION = "abfd4042216fa4f84c9c5c4e3e844a3143c70ad5"


def cpu_scheduling():
    result = {}
    for name in (
        "cpu.max",
        "cpu.stat",
        "cpu/cpu.cfs_quota_us",
        "cpu/cpu.cfs_period_us",
        "cpu/cpu.stat",
    ):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            result[name] = path.read_text().strip()
    result["cpu_model"] = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        None,
    )
    return result


@contextmanager
def backend_context(name):
    """Scope the fixed-schema diagnostic outside all measured operations."""
    if name != "canonical_no_fill_defaults":
        yield
        return
    import torch._library.utils as library_utils

    original = library_utils.fill_defaults

    def fixed_schema(schema, positional, keywords):
        if schema.name == "torch_ms_deform_attn::forward" and len(positional) == 6 and not keywords:
            return positional, keywords
        return original(schema, positional, keywords)

    library_utils.fill_defaults = fixed_schema
    try:
        yield
    finally:
        library_utils.fill_defaults = original


def native_function(forward, backward):
    class Native(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, shapes, starts, locations, weights):
            ctx.save_for_backward(value, shapes, starts, locations, weights)
            return forward(value, shapes, starts, locations, weights, 64)

        @staticmethod
        def backward(ctx, grad):
            value, shapes, starts, locations, weights = ctx.saved_tensors
            gv, gl, gw = backward(value, shapes, starts, locations, weights, grad.contiguous(), 64)
            return gv, None, None, gl, gw

    return Native.apply


def common_precision(fn):
    def run(value, shapes, starts, locations, weights):
        return fn(value.float(), shapes, starts, locations.float(), weights.float()).to(value.dtype)

    return run


def inputs(case, dtype):
    batch, queries, heads, channels, sizes = CASES[case]
    shapes = torch.tensor(sizes, device="cuda", dtype=torch.int64)
    starts = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
    value = torch.randn(
        batch, sum(h * w for h, w in sizes), heads, channels, device="cuda", dtype=dtype
    )
    loc_shape = (batch, queries, heads, len(sizes), 4, 2)
    scale = shapes.flip(-1).reshape(1, 1, 1, -1, 1, 2)
    pixels = (torch.rand(loc_shape, device="cuda") * (scale - 1)).floor() + 0.75
    locations = (pixels / scale).to(dtype)
    weights = torch.rand(loc_shape[:-1], device="cuda").flatten(-2).softmax(-1)
    weights = weights.reshape(loc_shape[:-1]).to(dtype)
    for tensor in (value, locations, weights):
        tensor.requires_grad_()
    grad = torch.randn(batch, queries, heads * channels, device="cuda", dtype=dtype)
    data = (value, shapes, starts, locations, weights)
    # Diagnostic unchecked / int32 variants are only called with proven valid metadata.
    # Half-range also accommodates interpolation-neighbor offsets and the final
    # rounded-up grid-stride increment in diagnostic int32 builds.
    limit = (2**31 - 1) // 2
    assert all(t.numel() <= limit for t in (*data, grad))
    assert batch * value.shape[1] * heads * channels <= limit
    assert batch * queries * heads * len(sizes) * 4 * 2 <= limit
    starts_cpu = starts.cpu().tolist()
    offset = 0
    for (height, width), start in zip(sizes, starts_cpu):
        assert height > 0 and width > 0 and start == offset
        offset += height * width
        assert offset <= value.shape[1] and max(height, width, offset) <= limit
    return data, grad, sizes, scale


def oracle(data, grad, sizes):
    tensors = tuple(x.detach().double().requires_grad_() for x in (data[0], data[3], data[4]))
    out = ms_deform_attn_core_pytorch(tensors[0], sizes, tensors[1], tensors[2])
    return (out.detach(), *torch.autograd.grad(out, tensors, grad.double()))


def comparison(actual, expected, dtype, scale):
    tolerance = 2e-4 if dtype == torch.float32 else 2e-3
    details = {}
    for index, (name, a, b) in enumerate(
        zip(("output", "grad_value", "grad_locations", "grad_weights"), actual, expected)
    ):
        a, b = a.detach().double(), b.detach().double()
        if index == 2:
            a, b = a / scale, b / scale
        error = (a - b).abs()
        details[name] = {
            "passed": bool(torch.allclose(a, b, atol=tolerance, rtol=tolerance)),
            "finite": bool(torch.isfinite(a).all()),
            "max_abs": error.max().item(),
            "rms": error.square().mean().sqrt().item(),
        }
    return {
        "passed": all(x["passed"] and x["finite"] for x in details.values()),
        "atol": tolerance,
        "rtol": tolerance,
        "location_gradient_units": "feature pixels",
        "errors": details,
    }


def evaluate(fn, data, grad):
    out = fn(*data)
    gradients = torch.autograd.grad(out, (data[0], data[3], data[4]), grad)
    return (out.detach(), *gradients)


def runner(fn, data, grad, mode):
    if mode == "forward":

        def run():
            with torch.no_grad():
                return fn(*data)
    else:
        tensors = (data[0], data[3], data[4])

        def run():
            out = fn(*data)
            return (out, *torch.autograd.grad(out, tensors, grad))

    return run


def distribution(values):
    values = sorted(values)
    return {
        "median": statistics.median(values),
        "min": values[0],
        "max": values[-1],
        "p25": values[len(values) // 4],
        "p75": values[3 * len(values) // 4],
        "samples": values,
    }


def measure_eager(run, min_time, latency_samples):
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    # Match the original benchmark's amortized synchronized wall-time semantics.
    # Also retain a single-call sync latency to expose queueing/CPU sensitivity.
    block = 10
    wall, elapsed = [], 0.0
    while elapsed < min_time or len(wall) < 4:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(block):
            run()
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        elapsed += duration
        wall.append(duration * 1000 / block)
    event, synchronized_wall = [], []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(latency_samples):
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        begin.record()
        run()
        end.record()
        end.synchronize()
        synchronized_wall.append((time.perf_counter() - wall_start) * 1000)
        event.append(begin.elapsed_time(end))
    return {
        "wall_ms": distribution(wall),
        "calls_per_wall_sample": block,
        "synchronized_single_call_wall_ms": distribution(synchronized_wall),
        "single_call_event_interval_ms": distribution(event),
    }


def capture(run):
    # All inputs, autograd nodes, eager calls and captures share the dedicated
    # stream selected by main. Reusing leaves across per-backend streams causes
    # AccumulateGrad stream mismatches and can contaminate eager measurements.
    stream = torch.cuda.current_stream()
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = run()
    graph.replay()
    torch.cuda.synchronize()
    return graph, outputs


def measure_graph(graph, samples=8, replays=30):
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    event_times, wall_times = [], []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        begin.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        wall_times.append((time.perf_counter() - wall_start) * 1000 / replays)
        event_times.append(begin.elapsed_time(end) / replays)
    return {
        "event_ms_per_replay": distribution(event_times),
        "wall_ms_per_replay": distribution(wall_times),
        "replays_per_sample": replays,
    }


def profile(run, trace_path, iterations, mode=None):
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(iterations):
            with torch.profiler.record_function("MSDA_DIAGNOSTIC_ITERATION"):
                run()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    # FunctionEvent CUDA records also contain synthetic GPU annotation ranges
    # that overlap real kernels. Use explicit Chrome activity categories instead.
    trace = json.loads(Path(trace_path).read_text())
    cuda = {}
    selected_categories = {"kernel", "gpu_memcpy", "gpu_memset"}
    main_counts = {"forward": 0, "backward": 0}
    for event in trace.get("traceEvents", []):
        categories = set(event.get("cat", "").split(","))
        if event.get("ph") != "X" or not categories.intersection(selected_categories):
            continue
        name = event.get("name", "unnamed CUDA activity")
        item = cuda.setdefault(name, {"count": 0, "total_us": 0.0, "category": event.get("cat")})
        item["count"] += 1
        item["total_us"] += event["dur"]
        if "ms_deformable_im2col_gpu_kernel" in name:
            main_counts["forward"] += 1
        elif "ms_deformable_col2im_gpu_kernel" in name:
            main_counts["backward"] += 1
    for item in cuda.values():
        item["us_per_iteration"] = item["total_us"] / iterations
        item["us_per_call"] = item["total_us"] / item["count"]
    cpu = []
    for event in prof.key_averages():
        cpu.append(
            {
                "name": event.key,
                "count": event.count,
                "cpu_total_us": event.cpu_time_total,
                "self_cpu_total_us": event.self_cpu_time_total,
                "device_total_us": getattr(event, "device_time_total", 0.0),
            }
        )
    cpu.sort(key=lambda item: item["self_cpu_total_us"], reverse=True)
    expected_counts = {"forward": iterations}
    if mode == "forward_backward" or (mode is None and main_counts["backward"]):
        expected_counts["backward"] = iterations
    complete = bool(cuda) and all(
        main_counts[name] == expected for name, expected in expected_counts.items()
    )
    observed_total = sum(x["total_us"] for x in cuda.values()) / iterations if cuda else None
    result = {
        "iterations": iterations,
        "trace": str(trace_path),
        "cuda": cuda,
        "cuda_activity_source": "Chrome trace complete events in kernel/gpu_memcpy/gpu_memset categories; overlapping annotations excluded",
        "msda_kernel_counts": main_counts,
        "expected_msda_kernel_counts": expected_counts,
        "cuda_counts_complete": complete,
        "observed_cuda_total_us_per_iteration": observed_total,
        "cuda_total_us_per_iteration": observed_total if complete else None,
        "cpu": cpu,
    }
    warnings = []
    if not cuda:
        warnings.append("Profiler returned no CUDA activities; pure kernel duration is unavailable")
    for name, expected in expected_counts.items():
        if main_counts[name] != expected:
            warnings.append(
                f"Incomplete or unexpected {name} kernel count: recorded {main_counts[name]}, expected {expected}; per-iteration totals are unavailable, per-call means cover observed events only"
            )
    if warnings:
        result["warnings"] = warnings
        result["warning"] = "; ".join(warnings)
    return result


def profile_python(run, path, iterations=100):
    torch.cuda.synchronize()
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(iterations):
        run()
    profiler.disable()
    torch.cuda.synchronize()
    profiler.dump_stats(str(path))
    rows = []
    for (filename, line, name), (primitive, total, own, cumulative, _) in pstats.Stats(
        profiler
    ).stats.items():
        rows.append(
            {
                "file": filename,
                "line": line,
                "name": name,
                "primitive_calls": primitive,
                "total_calls": total,
                "self_us_per_iteration": own * 1e6 / iterations,
                "cumulative_us_per_iteration": cumulative * 1e6 / iterations,
            }
        )
    rows.sort(key=lambda row: row["cumulative_us_per_iteration"], reverse=True)
    return {"iterations": iterations, "profile": str(path), "functions": rows}


def fingerprint(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def module_metadata(module):
    result = {"name": module.__name__, "file": getattr(module, "__file__", None), "files": []}
    if result["file"]:
        root = Path(result["file"]).parent
        # HF Python code lives in an inner package; the extension binary and
        # build metadata can be siblings one or two levels above that package.
        for candidate in (root, *list(root.parents)[:2]):
            if (candidate / "metadata.json").is_file() or any(candidate.glob("*.so")):
                root = candidate
                break
        result["artifact_root"] = str(root.resolve())
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in (".so", ".py", ".json", ".toml"):
                if ".git" not in path.parts and "__pycache__" not in path.parts:
                    result["files"].append(fingerprint(path))
    return result


def load_backends(args, report):
    backends = {
        "canonical": ms_deform_attn,
        "canonical_no_fill_defaults": ms_deform_attn,
        "native_control": native_function(_C.ms_deform_attn_forward, _C.ms_deform_attn_backward),
    }
    report["canonical_binary"] = fingerprint(_C.__file__)
    if not args.skip_hf:
        from kernels import get_kernel

        hf = None
        report["hf"] = {
            "repository": "kernels-community/deformable-detr",
            "requested_revision": HF_REVISION,
            "load_attempts": [],
        }
        for kwargs in ({"revision": HF_REVISION}, {"version": 1}):
            try:
                hf = get_kernel("kernels-community/deformable-detr", **kwargs)
                report["hf"]["loaded_with"] = kwargs
                report["hf"]["module"] = module_metadata(hf)
                report["hf"]["exact_requested_revision_loaded"] = "revision" in kwargs
                break
            except Exception as exc:
                report["hf"]["load_attempts"].append({"arguments": kwargs, "error": repr(exc)})
        if hf is None:
            report["setup_errors"]["hf"] = "Both pinned revision and version1 failed"
        else:
            backends["hf"] = native_function(hf.ms_deform_attn_forward, hf.ms_deform_attn_backward)
    if args.variants_dir:
        root = args.variants_dir
        manifest = json.loads((root / "manifest.json").read_text())
        report["variants_manifest"] = manifest
        variants = manifest.get("variants", manifest)
        if isinstance(variants, dict):
            variants = [
                {"name": name, **({"path": val} if isinstance(val, str) else val)}
                for name, val in variants.items()
            ]
        for variant in variants:
            name = variant["name"]
            try:
                path = variant.get("path", variant.get("so_path"))
                module_name = variant.get("module_name", variant.get("module", name))
                if path:
                    path = Path(path)
                    if not path.is_absolute():
                        path = root / path
                    spec = importlib.util.spec_from_file_location(module_name, path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                else:
                    module = importlib.import_module(module_name)
                forward_name = variant.get("forward", "ms_deform_attn_forward")
                backward_name = variant.get("backward", "ms_deform_attn_backward")
                forward = getattr(module, forward_name, getattr(module, "forward", None))
                backward = getattr(module, backward_name, getattr(module, "backward", None))
                backends[name] = native_function(forward, backward)
            except Exception as exc:
                report["setup_errors"][name] = repr(exc)
    if args.backends:
        backends = {name: fn for name, fn in backends.items() if name in args.backends}
    return backends


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants-dir", type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument(
        "--dtypes", nargs="+", choices=["float32", "float16"], default=["float32", "float16"]
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["forward", "forward_backward"],
        default=["forward", "forward_backward"],
    )
    parser.add_argument("--backends", nargs="+")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-run-time", type=float, default=0.12)
    parser.add_argument("--latency-samples", type=int, default=10)
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--profile-dtypes", nargs="+", default=["float32"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    # Keep this object alive for the entire process's benchmark and create all
    # inputs/autograd nodes on it. CUDA graph capture requires a nondefault stream.
    benchmark_stream = torch.cuda.Stream()
    torch.cuda.set_stream(benchmark_stream)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
        },
        "measurement_notes": {
            "precision": "float32 compute; FP16 casts included for every implementation",
            "timing": "warmup excluded; synchronized block wall latency; per-call event includes host idle",
            "graph": "captured single iteration; event interval per replay largely excludes Python dispatch",
            "stream": "one dedicated nondefault stream shared by input creation, oracle, qualification, eager, graph capture/replay and profiling",
            "profile": "profiling collected after latency measurements; profiler affects kernel durations",
            "correctness": "independent FP64 grid_sample output and all3 gradients; feature-pixel location gradients",
            "int32_safety": "every input numel/address product <= INT32_MAX/2; contiguous valid positive metadata",
            "no_fill_defaults": "process-local bypass only for fully supplied canonical forward fixed schema; restored after each qualification/measurement/profile",
        },
        "cpu_scheduling_start": cpu_scheduling(),
        "setup_errors": {},
        "correctness": [],
        "measurements": [],
        "profiles": [],
        "python_profiles": [],
        "graph_errors": [],
        "errors": [],
    }
    for package in ("kernels", "huggingface-hub", "torch"):
        try:
            report["environment"][package + "_package"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        report["environment"]["nvidia_smi"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,clocks.sm,clocks.mem,power.limit",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:
        pass

    def save():
        path = args.output_dir / "report.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)

    backends = load_backends(args, report)
    save()
    for case in args.cases:
        for dtype_name in args.dtypes:
            dtype = getattr(torch, dtype_name)
            data, grad, sizes, scale = inputs(case, dtype)
            truth = oracle(data, grad, sizes)
            qualified = {}
            for name, raw in backends.items():
                fn = (
                    raw
                    if dtype == torch.float32 or name.startswith("canonical")
                    else common_precision(raw)
                )
                label = {"case": case, "dtype": dtype_name, "backend": name}
                try:
                    with backend_context(name):
                        check = comparison(evaluate(fn, data, grad), truth, dtype, scale)
                    report["correctness"].append({**label, **check})
                    if check["passed"]:
                        qualified[name] = fn
                    print(
                        json.dumps({"stage": "correctness", **label, "passed": check["passed"]}),
                        flush=True,
                    )
                except Exception:
                    report["errors"].append(
                        {**label, "stage": "correctness", "error": traceback.format_exc()}
                    )
                save()
            for mode in args.modes:
                runs = {name: runner(fn, data, grad, mode) for name, fn in qualified.items()}
                graphs = {}
                if not args.skip_graph:
                    for name, run in runs.items():
                        label = {"case": case, "dtype": dtype_name, "backend": name, "mode": mode}
                        try:
                            with backend_context(name):
                                graph, outputs = capture(run)
                            if mode == "forward_backward":
                                check = comparison(outputs, truth, dtype, scale)
                                if not check["passed"]:
                                    raise RuntimeError(
                                        "CUDA graph correctness failed: " + json.dumps(check)
                                    )
                            else:
                                tolerance = 2e-4 if dtype == torch.float32 else 2e-3
                                torch.testing.assert_close(
                                    outputs.double(), truth[0], rtol=tolerance, atol=tolerance
                                )
                            graphs[name] = (graph, outputs)
                        except Exception:
                            report["graph_errors"].append(
                                {**label, "error": traceback.format_exc()}
                            )
                        save()
                names = list(runs)
                for repeat in range(args.repeats):
                    rng.shuffle(names)
                    for name in names:
                        label = {
                            "case": case,
                            "dtype": dtype_name,
                            "backend": name,
                            "mode": mode,
                            "repeat": repeat,
                        }
                        with backend_context(name):
                            measurement = {
                                **label,
                                "eager": measure_eager(
                                    runs[name], args.min_run_time, args.latency_samples
                                ),
                            }
                        if name in graphs:
                            measurement["graph"] = measure_graph(graphs[name][0])
                        report["measurements"].append(measurement)
                        print(
                            json.dumps(
                                {
                                    "stage": "timing",
                                    **label,
                                    "wall_ms": measurement["eager"]["wall_ms"]["median"],
                                    "graph_ms": measurement.get("graph", {})
                                    .get("event_ms_per_replay", {})
                                    .get("median"),
                                }
                            ),
                            flush=True,
                        )
                        save()
                if not args.skip_profile and dtype_name in args.profile_dtypes:
                    for name, run in runs.items():
                        label = {"case": case, "dtype": dtype_name, "backend": name, "mode": mode}
                        stem = f"{case}-{dtype_name}-{mode}-{name}"
                        try:
                            with backend_context(name):
                                report["profiles"].append(
                                    {
                                        **label,
                                        **profile(
                                            run,
                                            args.output_dir / f"{stem}.trace.json",
                                            args.profile_iterations,
                                            mode=mode,
                                        ),
                                    }
                                )
                                if case == "decoder" and mode == "forward_backward":
                                    report["python_profiles"].append(
                                        {
                                            **label,
                                            **profile_python(
                                                run, args.output_dir / f"{stem}.pstats"
                                            ),
                                        }
                                    )
                        except Exception:
                            report["errors"].append(
                                {**label, "stage": "profile", "error": traceback.format_exc()}
                            )
                        save()
                graphs.clear()
                gc.collect()
                torch.cuda.empty_cache()
            del data, grad, truth, scale, qualified, runs
            gc.collect()
            torch.cuda.empty_cache()
    report["cpu_scheduling_end"] = cpu_scheduling()
    report["completed"] = True
    save()
    print(
        json.dumps({"stage": "complete", "output": str(args.output_dir / "report.json")}),
        flush=True,
    )


if __name__ == "__main__":
    main()

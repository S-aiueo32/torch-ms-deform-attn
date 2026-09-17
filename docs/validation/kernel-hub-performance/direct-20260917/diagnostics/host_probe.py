#!/usr/bin/env python3
"""Small CUDA workload for separating host/autograd overhead from kernel work."""

import argparse
import json
import random
import statistics
import sys
import traceback
from pathlib import Path

import torch
from probe import (
    CASES,
    backend_context,
    capture,
    comparison,
    cpu_scheduling,
    evaluate,
    inputs,
    load_backends,
    measure_eager,
    measure_graph,
    oracle,
    profile_python,
    runner,
)


def summary(values):
    values = sorted(values)
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "median": statistics.median(values),
        "iqr": quartiles[2] - quartiles[0],
        "p25": quartiles[0],
        "p75": quartiles[2],
        "min": values[0],
        "max": values[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--min-run-time", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if args.repeats < 2 or args.min_run_time <= 0:
        parser.error("repeats must be >=2 and min-run-time positive")
    # One stream owns all leaf creation, oracle, autograd, captures and eager work.
    benchmark_stream = torch.cuda.Stream()
    torch.cuda.set_stream(benchmark_stream)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    CASES["tiny_host"] = (1, 1, 1, 8, ((2, 2),))
    report = {
        "config": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "shape": {
            "batch": 1,
            "queries": 1,
            "heads": 1,
            "channels": 8,
            "spatial_shapes": [[2, 2]],
            "points": 4,
            "dtype": "float32",
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": str(torch.cuda.get_device_properties(0)),
            "threads": torch.get_num_threads(),
        },
        "cpu_scheduling_start": cpu_scheduling(),
        "notes": {
            "measurement": "7 shuffled backend repetitions by default; synchronized wall blocks of10 calls; synchronized single-call/event samples also retained",
            "qualification": "independent FP64 grid_sample output and all3 gradients",
            "coordinates": "interior quarter-pixel locations exactly representable at 2x2",
            "precision": "FP32 inputs and computation throughout",
            "stream": "one dedicated nondefault stream held alive throughout",
            "python_profile": "cProfile overhead affects absolute time; use function counts/times to attribute host path, not as latency result",
            "scope": "fixed tiny CUDA workload diagnoses host overhead; do not extrapolate its absolute latency to decoder/encoder",
        },
        "setup_errors": {},
        "correctness": [],
        "measurements": [],
        "summary_us": {},
        "python_profiles": [],
        "graph_results": [],
        "errors": [],
    }

    def save():
        destination = args.output_dir / "host-probe.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)

    load_args = argparse.Namespace(
        variants_dir=None,
        skip_hf=False,
        backends=["canonical", "canonical_no_fill_defaults", "native_control", "hf"],
    )
    backends = load_backends(load_args, report)
    data, grad, sizes, scale = inputs("tiny_host", torch.float32)
    truth = oracle(data, grad, sizes)
    runs = {}
    for name, fn in backends.items():
        try:
            with backend_context(name):
                check = comparison(evaluate(fn, data, grad), truth, torch.float32, scale)
            report["correctness"].append({"backend": name, **check})
            if check["passed"]:
                runs[name] = runner(fn, data, grad, "forward_backward")
            print(
                json.dumps({"stage": "correctness", "backend": name, "passed": check["passed"]}),
                flush=True,
            )
        except Exception:
            report["errors"].append(
                {"stage": "correctness", "backend": name, "error": traceback.format_exc()}
            )
        save()
    names = list(runs)
    for repeat in range(args.repeats):
        rng.shuffle(names)
        for name in names:
            with backend_context(name):
                measured = measure_eager(runs[name], args.min_run_time, latency_samples=10)
            report["measurements"].append({"backend": name, "repeat": repeat, **measured})
            print(
                json.dumps(
                    {
                        "stage": "timing",
                        "backend": name,
                        "repeat": repeat,
                        "wall_us": measured["wall_ms"]["median"] * 1000,
                    }
                ),
                flush=True,
            )
            save()
    for name, run in runs.items():
        rows = [row for row in report["measurements"] if row["backend"] == name]
        report["summary_us"][name] = {}
        for field in (
            "wall_ms",
            "synchronized_single_call_wall_ms",
            "single_call_event_interval_ms",
        ):
            values = [row[field]["median"] * 1000 for row in rows]
            report["summary_us"][name][field.removesuffix("_ms")] = {
                **summary(values),
                "repeat_medians_us": values,
                "pooled_samples": summary(
                    [v * 1000 for row in rows for v in row[field]["samples"]]
                ),
            }
        try:
            with backend_context(name):
                python_profile = profile_python(
                    run, args.output_dir / f"{name}.pstats", iterations=300
                )
                matching = [
                    row
                    for row in python_profile["functions"]
                    if row["name"] in ("fill_defaults", "fixed_schema")
                ]
                python_profile["fill_defaults_contribution"] = {
                    "functions": matching,
                    "self_us_per_iteration": sum(row["self_us_per_iteration"] for row in matching),
                    "cumulative_us_per_iteration": sum(
                        row["cumulative_us_per_iteration"] for row in matching
                    ),
                    "note": "Cumulative parent/child times can overlap if a fallback calls original fill_defaults",
                }
                report["python_profiles"].append({"backend": name, **python_profile})
            if args.graph:
                with backend_context(name):
                    graph, outputs = capture(run)
                check = comparison(outputs, truth, torch.float32, scale)
                if not check["passed"]:
                    raise RuntimeError(
                        "Graph did not pass FP64 qualification: " + json.dumps(check)
                    )
                report["graph_results"].append(
                    {
                        "backend": name,
                        "correctness": check,
                        **measure_graph(graph, samples=5, replays=50),
                    }
                )
                del graph, outputs
        except Exception:
            report["errors"].append(
                {"stage": "profile_or_graph", "backend": name, "error": traceback.format_exc()}
            )
        save()
    canonical = report["summary_us"].get("canonical", {}).get("wall", {}).get("median")
    if canonical is not None:
        report["canonical_wall_difference_us"] = {
            name: canonical - row["wall"]["median"] for name, row in report["summary_us"].items()
        }
    report["cpu_scheduling_end"] = cpu_scheduling()
    report["completed"] = True
    save()
    print(
        json.dumps(
            {
                "stage": "complete",
                "output": str(args.output_dir / "host-probe.json"),
                "summary_us": report["summary_us"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

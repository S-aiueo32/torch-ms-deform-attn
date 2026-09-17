#!/usr/bin/env python3
"""Summarize probe.py results and diagnostic CUDA build resource usage.

Usage: python summarize.py results/report.json --variants-dir variants
Writes summary.json and summary.md beside report.json, and prints summary.md.
Can be rerun against an incomplete report while the experiment is running.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

LABELS = ("case", "dtype", "mode", "backend")
METRICS = {
    "wall_us": ("eager", "wall_ms"),
    "graph_us": ("graph", "event_ms_per_replay"),
    "single_call_wall_us": ("eager", "synchronized_single_call_wall_ms"),
    "single_call_event_us": ("eager", "single_call_event_interval_ms"),
    "graph_wall_us": ("graph", "wall_ms_per_replay"),
}
BACKEND_ORDER = {
    name: i
    for i, name in enumerate(
        (
            "hf",
            "canonical",
            "canonical_no_fill_defaults",
            "native_control",
            "baseline",
            "int32",
            "no_metadata",
            "int32_no_metadata",
            "no_bounds",
            "fwd256",
            "fwd128",
            "int32_no_metadata_fwd256",
        )
    )
}


def order(row):
    return (
        row.get("case", ""),
        row.get("dtype", ""),
        row.get("mode", ""),
        BACKEND_ORDER.get(row.get("backend"), 100),
        row.get("backend", ""),
    )


def safe_ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def summarize_measurements(report, baseline):
    grouped = defaultdict(list)
    for measurement in report.get("measurements", []):
        grouped[tuple(measurement.get(label) for label in LABELS)].append(measurement)
    rows = []
    for key, measurements in grouped.items():
        row = dict(zip(LABELS, key))
        row["repeats"] = len(measurements)
        for metric, path in METRICS.items():
            values = []
            for measurement in measurements:
                value = measurement
                for component in path:
                    value = value.get(component, {})
                if isinstance(value, dict) and isinstance(value.get("median"), (int, float)):
                    values.append(value["median"] * 1000)
            row[metric] = statistics.median(values) if values else None
            row[metric + "_repeat_medians"] = values
            row[metric + "_repeat_min"] = min(values) if values else None
            row[metric + "_repeat_max"] = max(values) if values else None
        rows.append(row)
    lookup = {tuple(row[label] for label in LABELS): row for row in rows}
    for row in rows:
        group = tuple(row[label] for label in LABELS[:-1])
        for reference in ("hf", baseline):
            other = lookup.get((*group, reference), {})
            for metric in ("wall_us", "graph_us"):
                row[metric.removesuffix("_us") + "_ratio_to_" + reference] = safe_ratio(
                    row[metric], other.get(metric)
                )
    return sorted(rows, key=order)


def kernel_kind(name):
    if "ms_deformable_im2col_gpu_kernel" in name:
        return "forward"
    if "ms_deformable_col2im_gpu_kernel" in name:
        return "backward"
    return "auxiliary"


def summarize_profiles(report):
    rows = []
    for profile in report.get("profiles", []):
        row = {label: profile.get(label) for label in LABELS}
        row.update({"iterations": profile.get("iterations"), "trace": profile.get("trace")})
        totals = {"forward": 0.0, "backward": 0.0, "auxiliary": 0.0}
        kernels = []
        iterations = profile.get("iterations", 1)
        for name, kernel in profile.get("cuda", {}).items():
            kind = kernel_kind(name)
            us_per_iteration = kernel.get(
                "us_per_iteration", kernel.get("total_us", 0) / iterations
            )
            totals[kind] += us_per_iteration
            kernels.append(
                {
                    "name": name,
                    "kind": kind,
                    **kernel,
                    "us_per_iteration": us_per_iteration,
                    "calls_per_iteration": kernel.get("count", 0) / iterations,
                }
            )
        row.update({kind + "_us_per_iteration": value for kind, value in totals.items()})
        row["total_us_per_iteration"] = sum(totals.values()) if kernels else None
        row["kernels"] = sorted(kernels, key=lambda item: item["us_per_iteration"], reverse=True)
        if "warning" in profile:
            row["warning"] = profile["warning"]
        rows.append(row)
    return sorted(rows, key=order)


def demangle(names):
    executable = shutil.which("c++filt") or shutil.which("llvm-cxxfilt")
    if not executable:
        return dict(zip(names, names))
    result = subprocess.run(
        [executable, "-n"],
        input="\n".join(names) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout.splitlines()
    if result.returncode or len(output) != len(names):
        return dict(zip(names, names))
    return dict(zip(names, output))


def parse_ptxas(text, variant, path):
    entries = []
    current = None
    arch = None
    for line in text.splitlines():
        compiled = re.search(
            r"Compiling entry function ['\"]([^'\"]+)['\"] for ['\"]([^'\"]+)['\"]", line
        )
        properties = re.search(r"Function properties for\s+(\S+)", line)
        if compiled:
            if current:
                entries.append(current)
            name, arch = compiled.groups()
            current = {
                "variant": variant,
                "symbol": name,
                "architecture": arch,
                "build_log": str(path),
            }
        elif properties:
            name = properties.group(1).strip("'\"")
            if current is None or name != current["symbol"]:
                if current:
                    entries.append(current)
                current = {
                    "variant": variant,
                    "symbol": name,
                    "architecture": arch,
                    "build_log": str(path),
                }
        if current is None:
            continue
        patterns = {
            "registers": r"Used\s+(\d+)\s+registers",
            "stack_bytes": r"(\d+)\s+bytes stack frame",
            "spill_store_bytes": r"(\d+)\s+bytes spill stores",
            "spill_load_bytes": r"(\d+)\s+bytes spill loads",
            "shared_bytes": r"(\d+)\s+bytes smem",
        }
        for field, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                current[field] = int(match.group(1))
    if current:
        entries.append(current)
    names = list(dict.fromkeys(item["symbol"] for item in entries))
    demangled = demangle(names)
    for item in entries:
        symbol = item["symbol"]
        name = demangled[symbol]
        item["demangled"] = name
        item["kind"] = kernel_kind(name)
        item["scalar"] = (
            "float"
            if re.search(r"<float\b|_kernel[^I]*If", name)
            else ("double" if re.search(r"<double\b|_kernel[^I]*Id", name) else None)
        )
        if item["kind"] == "forward":
            match = re.search(r"im2col_gpu_kernel<(?:float|double),\s*([^>]+)>", name)
            if match:
                item["index_type"] = match.group(1)
            else:
                match = re.search(r"im2col_gpu_kernelI[fd]([ilx])E", symbol)
                item["index_type"] = (
                    {"i": "int", "l": "long", "x": "long long"}.get(match.group(1))
                    if match
                    else None
                )
        if item["kind"] == "backward":
            match = re.search(
                r"blocksize_aware_reduce_v\d<(?:float|double),\s*(\d+)(?:u|ul)?>", name
            )
            if not match:
                match = re.search(r"blocksize_aware_reduce_v\dI[fd]Lj(\d+)E", symbol)
            item["block_size"] = int(match.group(1)) if match else None
    return entries


def summarize_resources(report, report_path, variants_dir):
    manifest = report.get("variants_manifest", {})
    if variants_dir and (variants_dir / "manifest.json").is_file():
        manifest = json.loads((variants_dir / "manifest.json").read_text())
    variants = manifest.get("variants", []) if isinstance(manifest, dict) else manifest
    if isinstance(variants, dict):
        variants = [
            {"name": name, **item} for name, item in variants.items() if isinstance(item, dict)
        ]
    rows, missing = [], []
    for variant in variants:
        name = variant["name"]
        candidates = []
        if variants_dir:
            candidates.extend(
                (variants_dir / name / "build.log", variants_dir / name / "ptxas_summary.txt")
            )
        if variant.get("build_log"):
            candidates.append(Path(variant["build_log"]))
            candidates.append(report_path.parent / variant["build_log"])
        candidates.extend(
            (
                report_path.parent / "variants" / name / "build.log",
                report_path.parent.parent / "variants" / name / "build.log",
            )
        )
        log = next((path for path in candidates if path.is_file()), None)
        if log is None:
            missing.append(name)
            continue
        rows.extend(parse_ptxas(log.read_text(errors="replace"), name, log))
    rows.sort(
        key=lambda item: (
            BACKEND_ORDER.get(item["variant"], 100),
            item["variant"],
            item["kind"],
            item.get("scalar") or "",
            item.get("block_size") or 0,
            item.get("index_type") or "",
            item["symbol"],
        )
    )
    return rows, missing


def fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown(summary):
    report = summary
    baseline = report["baseline_backend"]
    env = report.get("environment", {})
    lines = [
        "# PR #16 CUDA profiling summary",
        "",
        f"GPU: {env.get('gpu', '?')}; PyTorch: {env.get('torch', '?')}; CUDA: {env.get('cuda', '?')}.",
        "",
        "Status: "
        + ("complete" if report["completed"] else "partial; measurements may still be running")
        + ".",
        f"Wall and graph times are microseconds. Values are medians of repeat medians; ratios above 1 are slower than the named reference. Baseline reference: `{baseline}`.",
        "",
        "Graph timing largely removes Python/dispatcher enqueue overhead, but includes GPU casts, allocations captured as operations, fills and kernels. Wall minus graph is not a direct measurement of host overhead.",
        "",
    ]
    hf = report.get("hf", {})
    if hf:
        lines += [
            f"HF exact requested revision loaded: `{hf.get('exact_requested_revision_loaded')}`; loader arguments: `{json.dumps(hf.get('loaded_with', {}))}`.",
            "",
        ]
    grouped = defaultdict(list)
    for row in report["measurements"]:
        grouped[(row["case"], row["dtype"], row["mode"])].append(row)
    for label, rows in grouped.items():
        lines += [
            "## " + " / ".join(label),
            "",
            "| Backend | Reps | Wall µs [repeat range] | Graph µs | Wall/HF | Graph/HF | Wall/base | Graph/base |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            wall = f"{fmt(row['wall_us'])} [{fmt(row['wall_us_repeat_min'])}, {fmt(row['wall_us_repeat_max'])}]"
            values = [
                row["backend"],
                str(row["repeats"]),
                wall,
                fmt(row["graph_us"]),
                fmt(row.get("wall_ratio_to_hf"), 3),
                fmt(row.get("graph_ratio_to_hf"), 3),
                fmt(row.get("wall_ratio_to_" + baseline), 3),
                fmt(row.get("graph_ratio_to_" + baseline), 3),
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    if report["profiles"]:
        lines += [
            "## CUDA profiler time per iteration",
            "",
            "Profiler durations are collected separately and can be perturbed by profiling; auxiliary work includes casts and zero fills.",
            "",
            "| Case / dtype / mode | Backend | Forward µs | Backward µs | Auxiliary µs | Total µs |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in report["profiles"]:
            values = [
                " / ".join(row[label] for label in LABELS[:-1]),
                row["backend"],
                *(
                    fmt(row.get(field + "_us_per_iteration"))
                    for field in ("forward", "backward", "auxiliary", "total")
                ),
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines += [
            "",
            "### Individual CUDA kernel durations",
            "",
            "| Case / dtype / mode | Backend | Kernel | Calls/iteration | µs/iteration |",
            "|---|---|---|---:|---:|",
        ]
        for row in report["profiles"]:
            for kernel in row["kernels"]:
                values = [
                    " / ".join(row[label] for label in LABELS[:-1]),
                    row["backend"],
                    "`" + kernel["name"].replace("|", "\\|").replace("`", "'") + "`",
                    fmt(kernel["calls_per_iteration"]),
                    fmt(kernel["us_per_iteration"]),
                ]
                lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    relevant = [
        item
        for item in report["ptxas"]
        if item["scalar"] == "float"
        and (
            item["kind"] == "forward" or item["kind"] == "backward" and item.get("block_size") == 32
        )
    ]
    if relevant:
        lines += [
            "## ptxas FP32 forward and C=32 backward resources",
            "",
            "Spill byte counts are static compiler reports; they do not measure dynamic spill traffic. Rebuilt baseline includes both int32 and int64 forward decomposition specializations; the small benchmark selects int32.",
            "",
            "| Variant | Kernel | Architecture | Registers/thread | Stack bytes | Spill stores | Spill loads |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for item in relevant:
            kernel = (
                "forward, index=" + str(item.get("index_type"))
                if item["kind"] == "forward"
                else "backward, C=32"
            )
            values = [
                item["variant"],
                kernel,
                item.get("architecture") or "?",
                *(
                    str(item.get(field, "?"))
                    for field in (
                        "registers",
                        "stack_bytes",
                        "spill_store_bytes",
                        "spill_load_bytes",
                    )
                ),
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    if report["missing_build_logs"]:
        lines += ["Build logs not found for: " + ", ".join(report["missing_build_logs"]) + ".", ""]
    qualification = report["correctness"]
    if qualification:
        failed = [item for item in qualification if not item.get("passed")]
        lines += [
            f"Correctness: {len(qualification) - len(failed)}/{len(qualification)} backend/case/dtype qualifications passed.",
            "",
        ]
        for item in failed:
            lines += ["Failed qualification: `" + json.dumps(item, ensure_ascii=False) + "`", ""]
    for key in ("setup_errors", "graph_errors", "errors"):
        errors = report.get(key)
        if errors:
            lines += [
                f"## {key}",
                "",
                "```json",
                json.dumps(errors, indent=2, ensure_ascii=False),
                "```",
                "",
            ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--variants-dir", type=Path)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.report.read_text())
    baseline = args.baseline
    available = {row["backend"] for row in source.get("measurements", [])}
    if baseline not in available and baseline == "baseline":
        baseline = "native_control" if "native_control" in available else "canonical"
    ptxas, missing = summarize_resources(source, args.report, args.variants_dir)
    summary = {
        "source_report": str(args.report.resolve()),
        "completed": source.get("completed", False),
        "baseline_backend": baseline,
        "environment": source.get("environment", {}),
        "hf": source.get("hf", {}),
        "measurements": summarize_measurements(source, baseline),
        "profiles": summarize_profiles(source),
        "ptxas": ptxas,
        "missing_build_logs": missing,
        "correctness": source.get("correctness", []),
        "python_profiles": source.get("python_profiles", []),
        **{key: source.get(key, []) for key in ("setup_errors", "graph_errors", "errors")},
    }
    output_dir = args.output_dir or args.report.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    text = markdown(summary)
    (output_dir / "summary.md").write_text(text)
    if not args.quiet:
        print(text)


if __name__ == "__main__":
    main()

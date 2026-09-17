#!/usr/bin/env python3
"""Correct CUDA profiler totals from actual Chrome GPU activity records.

Only complete (ph='X') events with category kernel, gpu_memcpy or gpu_memset
are summed. CPU/GPU user annotations are excluded, avoiding double-counting
synthetic MSDA_DIAGNOSTIC_ITERATION events returned by prof.events().

Usage:
  python reprocess_profiles.py report.json --trace-dir downloaded-traces
  python reprocess_profiles.py fresh/one-case.json

Both probe.py's profiles[] and one_case.py's profile object are supported.
Input files remain unchanged. Output defaults to NAME-corrected.json.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

GPU_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
RAW_FIELDS = ("cuda", "cuda_total_us_per_iteration", "warning")


def read_json(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def resolve_trace(recorded, report_path, trace_dirs, path_maps):
    original = Path(recorded)
    candidates = [original]
    for old, new in path_maps:
        try:
            relative = original.relative_to(old)
        except ValueError:
            continue
        candidates.append(new / relative)
    candidates.append(report_path.parent / original.name)
    for root in trace_dirs:
        candidates.extend((root / original.name, root / original.parent.name / original.name))
    for path in candidates:
        if path.is_file():
            return path
    recursive = {
        path.resolve()
        for root in trace_dirs
        if root.is_dir()
        for path in root.rglob(original.name)
        if path.is_file()
    }
    if len(recursive) == 1:
        return recursive.pop()
    if len(recursive) > 1:
        raise ValueError(
            f"Ambiguous downloaded trace {original.name}: {sorted(map(str, recursive))}; provide --path-map"
        )
    raise FileNotFoundError(f"Cannot find trace {recorded}; use --trace-dir or --path-map OLD=NEW")


def summarize_trace(document, iterations, mode):
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError(f"Invalid profile iteration count: {iterations!r}")
    events = document.get("traceEvents", []) if isinstance(document, dict) else document
    if not isinstance(events, list):
        raise ValueError("Chrome trace has no traceEvents array")
    category_counts = Counter()
    accepted_counts = Counter()
    excluded_counts = Counter()
    phase_counts = Counter()
    grouped = defaultdict(lambda: {"count": 0, "total_us": 0.0, "categories": set()})
    invalid_events = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        phase = str(event.get("ph", "<missing>"))
        phase_counts[phase] += 1
        if phase != "X":
            continue
        category = str(event.get("cat", "<missing>"))
        category_counts[category] += 1
        if category not in GPU_CATEGORIES:
            excluded_counts[category] += 1
            continue
        duration = event.get("dur")
        name = event.get("name")
        if (
            not isinstance(name, str)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration < 0
        ):
            invalid_events.append(
                {"event_index": index, "name": name, "duration": duration, "category": category}
            )
            continue
        accepted_counts[category] += 1
        item = grouped[name]
        item["count"] += 1
        item["total_us"] += duration
        item["categories"].add(category)
    cuda = {}
    for name, item in grouped.items():
        cuda[name] = {
            "count": item["count"],
            "total_us": item["total_us"],
            "us_per_iteration": item["total_us"] / iterations,
            "us_per_call": item["total_us"] / item["count"],
            "categories": sorted(item["categories"]),
        }
    main_counts = {
        "forward": sum(
            item["count"]
            for name, item in cuda.items()
            if "ms_deformable_im2col_gpu_kernel" in name
        ),
        "backward": sum(
            item["count"]
            for name, item in cuda.items()
            if "ms_deformable_col2im_gpu_kernel" in name
        ),
    }
    expected_counts = {"forward": iterations}
    if mode == "forward_backward":
        expected_counts["backward"] = iterations
    elif mode == "forward":
        expected_counts["backward"] = 0
    elif main_counts["backward"]:
        expected_counts["backward"] = iterations
    warnings = []
    for kind, expected in expected_counts.items():
        observed = main_counts[kind]
        if observed != expected:
            warnings.append(
                f"Main {kind} kernel count {observed} != expected {expected} for {iterations} profile iterations; "
                "check missing/duplicate activities or intentional im2col chunking before using durations."
            )
    if invalid_events:
        warnings.append(f"Discarded {len(invalid_events)} malformed GPU activity events.")
    if not cuda:
        warnings.append(
            "No actual GPU kernel/memcpy/memset complete events were found; GPU timing is unavailable."
        )
    audit = {
        "source": "Chrome traceEvents",
        "accepted_categories": sorted(GPU_CATEGORIES),
        "required_phase": "X",
        "duration_unit": "microseconds (Chrome trace dur; displayTimeUnit does not alter storage units)",
        "phase_counts": dict(sorted(phase_counts.items())),
        "complete_event_category_counts": dict(sorted(category_counts.items())),
        "accepted_complete_event_category_counts": dict(sorted(accepted_counts.items())),
        "excluded_complete_event_category_counts": dict(sorted(excluded_counts.items())),
        "invalid_events": invalid_events,
        "main_kernel_counts": main_counts,
        "expected_main_kernel_counts": expected_counts,
        "main_kernel_counts_match": all(
            main_counts[kind] == expected for kind, expected in expected_counts.items()
        ),
        "warnings": warnings,
    }
    return cuda, audit


def reprocess_profile(profile, report_path, config, trace_dirs, path_maps):
    result = copy.deepcopy(profile)
    if "prof_events_original" not in result:
        result["prof_events_original"] = {
            field: copy.deepcopy(profile[field]) for field in RAW_FIELDS if field in profile
        }
    mode = profile.get("mode", config.get("mode"))
    iterations = profile.get("iterations", config.get("profile_iterations"))
    try:
        trace = resolve_trace(profile["trace"], report_path, trace_dirs, path_maps)
        cuda, audit = summarize_trace(read_json(trace), iterations, mode)
        audit["resolved_trace"] = str(trace.resolve())
        audit["trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
        result["cuda"] = cuda
        result["cuda_total_us_per_iteration"] = (
            sum(item["total_us"] for item in cuda.values()) / iterations if cuda else None
        )
        result["trace_reprocessing"] = audit
        result.pop("warning", None)
        if audit["warnings"]:
            result["warning"] = " ".join(audit["warnings"])
        status = {
            "trace": str(trace),
            "case": profile.get("case", config.get("case")),
            "dtype": profile.get("dtype", config.get("dtype")),
            "backend": profile.get("backend", config.get("backend")),
            "mode": mode,
            "accepted_categories": audit["accepted_complete_event_category_counts"],
            "excluded_or_unknown_categories": audit["excluded_complete_event_category_counts"],
            "main_kernel_counts": audit["main_kernel_counts"],
            "main_kernel_counts_match": audit["main_kernel_counts_match"],
            "old_cuda_us_per_iteration": result["prof_events_original"].get(
                "cuda_total_us_per_iteration"
            ),
            "corrected_cuda_us_per_iteration": result["cuda_total_us_per_iteration"],
            "warnings": audit["warnings"],
        }
        return result, status
    except Exception as exc:
        error = repr(exc)
        result["cuda"] = {}
        result["cuda_total_us_per_iteration"] = None
        result["warning"] = (
            "Trace reprocessing failed; original possibly double-counted timing is preserved only in prof_events_original. "
            + error
        )
        result["trace_reprocessing"] = {"error": error}
        return result, {"trace": profile.get("trace"), "error": error}


def reprocess_report(source_path, destination, trace_dirs, path_maps):
    report = read_json(source_path)
    config = report.get("config", {})
    statuses = []
    if isinstance(report.get("profiles"), list):
        corrected = []
        for profile in report["profiles"]:
            result, status = reprocess_profile(profile, source_path, config, trace_dirs, path_maps)
            corrected.append(result)
            statuses.append(status)
        report["profiles"] = corrected
    if isinstance(report.get("profile"), dict):
        report["profile"], status = reprocess_profile(
            report["profile"], source_path, config, trace_dirs, path_maps
        )
        statuses.append(status)
    report["profile_trace_reprocessing"] = {
        "source_report": str(source_path.resolve()),
        "source_report_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "profile_count": len(statuses),
        "errors": sum("error" in status for status in statuses),
        "kernel_count_mismatches": sum(
            status.get("main_kernel_counts_match") is False for status in statuses
        ),
        "method": "Sum ph=X events in the exact category whitelist kernel/gpu_memcpy/gpu_memset; exclude all user annotations.",
    }
    if isinstance(report.get("measurement_notes"), dict):
        report["measurement_notes"]["profile"] = (
            "Corrected from Chrome trace ph=X GPU kernel/memcpy/memset activities only; user annotations excluded. "
            "Profiling may perturb durations. Original prof.events() aggregation retained in each profile.prof_events_original."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for status in statuses:
        print(json.dumps(status, ensure_ascii=False, allow_nan=False), flush=True)
    print(
        json.dumps({"output": str(destination), **report["profile_trace_reprocessing"]}), flush=True
    )
    return report["profile_trace_reprocessing"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--trace-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--path-map", action="append", default=[], metavar="OLD_PREFIX=LOCAL_DIRECTORY"
    )
    parser.add_argument("--output", type=Path, help="Output filename, only valid with one report")
    args = parser.parse_args()
    if args.output and len(args.reports) != 1:
        parser.error("--output requires exactly one input report")
    mappings = []
    for value in args.path_map:
        if "=" not in value:
            parser.error("--path-map must be OLD_PREFIX=LOCAL_DIRECTORY")
        old, new = value.split("=", 1)
        mappings.append((Path(old), Path(new)))
    errors = 0
    for source in args.reports:
        destination = args.output or source.with_name(source.stem + "-corrected.json")
        if source.resolve() == destination.resolve():
            parser.error("Output must differ from input; raw report must be preserved")
        summary = reprocess_report(source, destination, args.trace_dir, mappings)
        errors += summary["errors"]
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

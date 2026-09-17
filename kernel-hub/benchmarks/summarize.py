"""Summarize matching configurations; flag investigations, not confirmed regressions."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

BACKENDS = (
    "torch-ms-deform-attn",
    "kernel-hub-adapter",
    "hf-native",
    "mmcv-source",
    "msda-triton-rziga",
    "pytorch-reference",
)


def summarize(report):
    grouped = defaultdict(list)
    excluded = []
    for row in report["results"]:
        if row["status"] != "passed":
            excluded.append(row)
            continue
        if not row["correctness"]["passed"]:
            raise ValueError("Timing row failed correctness")
        grouped[tuple(row[key] for key in ("case", "dtype", "policy", "mode", "backend"))].append(
            row
        )
    results = []
    for key, rows in grouped.items():
        if len({row["repeat"] for row in rows}) != report["config"]["repeats"]:
            raise ValueError(f"Incomplete repetitions for {key}")
        results.append(
            {
                **dict(zip(("case", "dtype", "policy", "mode", "backend"), key)),
                **{
                    metric: statistics.median(row[metric] for row in rows)
                    for metric in (
                        "wall_ms",
                        "wall_iqr_ms",
                        "event_interval_ms",
                        "incremental_peak_bytes",
                    )
                },
            }
        )
    lookup = {
        tuple(row[key] for key in ("case", "dtype", "policy", "mode", "backend")): row
        for row in results
    }
    comparisons = []
    for row in results:
        if row["backend"] not in BACKENDS[:2]:
            continue
        baseline = lookup.get((row["case"], row["dtype"], row["policy"], row["mode"], "hf-native"))
        if baseline is None:
            continue
        delta = row["wall_ms"] - baseline["wall_ms"]
        comparisons.append(
            {
                **row,
                "hf_wall_ms": baseline["wall_ms"],
                "latency_ratio_to_hf": row["wall_ms"] / baseline["wall_ms"],
                "investigate": delta
                > max(
                    0.05 * baseline["wall_ms"], 2 * max(row["wall_iqr_ms"], baseline["wall_iqr_ms"])
                ),
            }
        )
    return {
        "results": results,
        "hf_comparisons": comparisons,
        "excluded": excluded,
        "setup_errors": report.get("setup_errors", {}),
        "note": "Flags require a fresh-process repeat before confirming regression.",
    }


def markdown(summary):
    lines = [
        "# Kernel Hub benchmark results",
        "",
        "Median of repetition medians; wall time in milliseconds.",
        "FP32-compute rows include input/output casts. Native rows use different arithmetic.",
        "",
    ]
    for policy in ("fp32-compute", "native"):
        for mode in ("forward", "backward", "forward_backward"):
            lines.extend(
                [
                    f"## {policy} / {mode}",
                    "",
                    "| Case / dtype | " + " | ".join(BACKENDS) + " |",
                    "| --- | " + " | ".join(["---:"] * len(BACKENDS)) + " |",
                ]
            )
            rows = [r for r in summary["results"] if r["policy"] == policy and r["mode"] == mode]
            for case, dtype in sorted({(r["case"], r["dtype"]) for r in rows}):
                match = {r["backend"]: r for r in rows if r["case"] == case and r["dtype"] == dtype}
                values = [f"{match[b]['wall_ms']:.4f}" if b in match else "—" for b in BACKENDS]
                lines.append(
                    f"| {case} / {dtype.removeprefix('torch.')} | " + " | ".join(values) + " |"
                )
            lines.append("")
    lines.extend(
        [
            "## Incremental peak allocation",
            "",
            "MiB, FP32-compute encoder forward+backward; includes outputs and temporary tensors.",
            "",
            "| dtype | " + " | ".join(BACKENDS) + " |",
            "| --- | " + " | ".join(["---:"] * len(BACKENDS)) + " |",
        ]
    )
    for dtype in ("torch.float32", "torch.float16", "torch.bfloat16"):
        rows = {
            r["backend"]: r
            for r in summary["results"]
            if r["case"] == "encoder"
            and r["dtype"] == dtype
            and r["policy"] == "fp32-compute"
            and r["mode"] == "forward_backward"
        }
        values = [
            f"{rows[b]['incremental_peak_bytes'] / 2**20:.2f}" if b in rows else "—"
            for b in BACKENDS
        ]
        lines.append(f"| {dtype.removeprefix('torch.')} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Missing entries are not zero latency. Consult the JSON for incorrect/unsupported cases.",
            "The JSON also retains IQRs, event intervals, HF ratios and investigation flags.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.report.read_text()))
    args.report.with_name("benchmark-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    args.report.with_name("benchmark-tables.md").write_text(markdown(result))

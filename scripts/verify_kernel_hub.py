"""Verify full GPU evidence, recorded Nix artifact identity when present, and cleanup."""

import argparse
import json
import re
from pathlib import Path

PRECISIONS = (("fp32", False), ("fp16", False), ("bf16", False), ("fp16", True), ("bf16", True))


def verify(directory, sha):
    def read(name):
        return json.loads((directory / name).read_text())

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(re.fullmatch(r"[0-9a-f]{40}", sha), "Use the full source SHA")
    summary = read("kernel-hub-summary.json")
    upstream = read("UPSTREAM.json")
    require(summary["source_sha"] == upstream["revision"] == sha, "Source SHA mismatch")
    require(summary["status"] == "passed", "Suite failed")
    require(not summary["focus_compile"] and not summary["phase1_only"], "Partial suite")
    require(summary["gpu"] and summary["cuda"], "Missing CUDA environment")
    require(re.fullmatch(r"[0-9a-f]{40}", summary["baseline_revision"]), "Unpinned baseline")
    names = [f"e2e-{dtype}" + ("-amp" if amp else "") for dtype, amp in PRECISIONS]
    runs = summary["runs"]
    require(
        len(runs) == 6 and {run["name"] for run in runs} == {"phase1", *names},
        "Missing or duplicate suite runs",
    )
    require(all(run["exit_code"] == 0 for run in runs), "Failed subprocess")
    require(read("runpod-state.json")["phase"] == "deleted", "Pod deletion not verified")
    phase1 = (directory / "phase1.log").read_text()
    require(re.search(r"Ran 3 tests in .*\n\s*OK\s*$", phase1), "Incomplete Phase 1 log")
    require(bool(read("phase1-numerics.json")), "Missing Phase 1 numerical evidence")
    for name in ("kernel-hub-checks.log", "python-packages.txt", "nvidia-smi.txt"):
        require(bool((directory / name).read_text().strip()), f"Empty {name}")
    manifests = {name: read(f"{name}-files.json") for name in ("candidate", "baseline")}
    require(
        bool(summary.get("nix_build")) == (directory / "NIX_BUILD.json").exists(),
        "Missing Nix build provenance",
    )
    if summary.get("nix_build"):
        nix_build = read("NIX_BUILD.json")
        require(summary["nix_build"] == nix_build, "Nix build provenance mismatch")
        require(re.fullmatch(r"[0-9]+", nix_build["build_run"]), "Invalid Nix build run")
        record = (
            Path(__file__).resolve().parents[1]
            / "docs/validation/kernel-hub-nix"
            / ("run-" + nix_build["build_run"])
        )
        require(
            nix_build["archive_sha256"] == (record / "distribution.sha256").read_text().split()[0],
            "Nix archive checksum mismatch",
        )
        original = read("nix-UPSTREAM.json")
        require(
            original == json.loads((record / "UPSTREAM.json").read_text())
            and original["revision"] == nix_build["artifact_source_sha"],
            "Nix artifact source mismatch",
        )

        def build_sources(manifest):
            return {
                name: digest
                for name, digest in manifest["source_sha256"].items()
                if not name.startswith("kernel-hub/e2e/")
            }

        require(build_sources(original) == build_sources(upstream), "Nix sources changed")
        files = json.loads((record / "distribution-files.json").read_text())
        expected = {
            Path(name).relative_to(nix_build["variant"]).as_posix(): digest
            for name, digest in files.items()
        }
        require(manifests["candidate"] == nix_build["files"] == expected, "Nix binary mismatch")
    for name, manifest in manifests.items():
        require(any(path.endswith(".so") for path in manifest), f"Missing {name} binary hash")
        require(
            all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in manifest.values()),
            f"Invalid {name} hashes",
        )
    for name, (dtype, amp) in zip(names, PRECISIONS):
        report = read(f"{name}.json")
        require(report["status"] == "passed" and "error" not in report, f"Failed {name}")
        require(report["torch"] == summary["torch"], f"Environment mismatch: {name}")
        require(
            report["harness_sha256"] == upstream["source_sha256"]["kernel-hub/e2e/rt_detr.py"],
            f"Harness mismatch: {name}",
        )
        cases = report["cases"]
        require(
            len(cases) == 4
            and {(case["training"], case["compile_backend"]) for case in cases}
            == {
                (training, backend) for training in (False, True) for backend in (None, "inductor")
            },
            f"Incomplete matrix: {name}",
        )
        for case in cases:
            require("error" not in case, f"Failed case: {name}")
            require(case["dtype"] == dtype and case["autocast"] == amp, f"Wrong precision: {name}")
            require(case["reference"] == "HF kernel", f"Missing HF comparison: {name}")
            for artifact, manifest in manifests.items():
                require(case[f"{artifact}_sha256"] == manifest, f"Artifact mismatch: {name}")
            ops = case["executed_ops"]
            for op in ("forward", "backward") if case["training"] else ("forward",):
                require(
                    any(key.endswith(f"::{op}") and count > 0 for key, count in ops.items()),
                    f"Missing executed {op}: {name}",
                )
            if case["training"]:
                require(case["gradient_tensor_count"] > 0, f"Missing gradients: {name}")
            if case["compile_backend"]:
                if summary.get("nix_build") and amp:
                    require(case["reference_compiled"], f"Uncompiled AMP reference: {name}")
                require(
                    any(graph["msda_calls"] > 0 for graph in case["compiled_graphs"]),
                    f"Missing compiled MSDA: {name}",
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    verify(args.evidence, args.sha)
    print(f"Phase 1 and all 20 RT-DETR cases verified for {args.sha}; Pod deleted")


if __name__ == "__main__":
    main()

"""Execute Phase 1 and Phase 2 against real local and pinned Hub CUDA artifacts."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from kernels import get_kernel, get_local_kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus-compile", action="store_true")
    parser.add_argument("--phase1-only", action="store_true")
    parser.add_argument("--diagnostic-controls", action="store_true")
    args = parser.parse_args()
    output = Path(os.environ["MSDA_OUTPUT_DIR"])
    candidate = Path(os.environ["MSDA_KERNEL_DIR"])
    revision = os.environ["MSDA_HF_BASELINE_REVISION"]
    if os.environ.get("LOCAL_KERNELS"):
        raise RuntimeError("Baseline resolution must not use local overrides")
    baseline_module = get_kernel("kernels-community/deformable-detr", revision=revision)
    baseline = Path(baseline_module.__file__).parent.resolve()
    module = get_local_kernel(candidate)
    if not hasattr(module, "_registrations"):
        raise RuntimeError("Builder output is not the upstream adapter")
    for name, root in (("candidate", Path(module.__file__).parent), ("baseline", baseline)):
        manifest = {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        (output / f"{name}-files.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for metadata in (root / "metadata.json", root.parent / "metadata.json"):
            if metadata.is_file():
                (output / f"{name}-metadata.json").write_bytes(metadata.read_bytes())
                break
    summary = {
        "source_sha": os.environ["CUDA_CHECKS_SOURCE_SHA"],
        "baseline_revision": revision,
        "builder": "hf-kernel-builder 0.16.0 create-pyproject + CMake local_install",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "status": "failed",
        "focus_compile": args.focus_compile,
        "phase1_only": args.phase1_only,
        "runs": [],
    }
    os.environ["LOCAL_KERNELS"] = f"kernels-community/deformable-detr={candidate}"
    try:
        runs = [
            (
                "phase1",
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    str(candidate / "tests"),
                    "-p",
                    "test_kernel.py",
                    "-v",
                ],
            )
        ]
        if args.focus_compile:
            runs = []
        for dtype, amp in (
            ("fp32", False),
            ("fp16", False),
            ("bf16", False),
            ("fp16", True),
            ("bf16", True),
        ):
            if args.phase1_only:
                continue
            if args.focus_compile and not amp:
                continue
            name = f"e2e-{dtype}" + ("-amp" if amp else "")
            command = [
                sys.executable,
                str(Path(__file__).with_name("rt_detr.py")),
                "--kernel-dir",
                str(candidate),
                "--baseline-kernel-dir",
                str(baseline),
                "--dtype",
                dtype,
                "--numerics",
                "controlled" if dtype == "fp32" or amp else "eager",
                "--report",
                str(output / f"{name}.json"),
            ]
            if amp:
                command.append("--autocast")
            if amp and dtype == "bf16":
                command.append("--compile-reference")
            if args.focus_compile:
                command.append("--compiled-training-only" if amp else "--compiled-only")
            runs.append((name, command))
            if args.diagnostic_controls and not amp and not args.focus_compile:
                diagnostic = command.copy()
                diagnostic[diagnostic.index("--numerics") + 1] = "default"
                diagnostic[diagnostic.index("--report") + 1] = str(
                    output / f"diagnostic-{dtype}.json"
                )
                diagnostic.append("--compiled-training-only")
                runs.append((f"diagnostic-{dtype}", diagnostic))
        for name, command in runs:
            print(f"Running {name}", flush=True)
            with (output / f"{name}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            summary["runs"].append({"name": name, "exit_code": result.returncode})
            print(f"{name}: exit {result.returncode}", flush=True)
            if result.returncode:
                print((output / f"{name}.log").read_text()[-6000:], flush=True)
            (output / "kernel-hub-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if any(run["exit_code"] for run in summary["runs"]):
            raise RuntimeError("Kernel Hub validation failed; inspect individual logs")
        summary["status"] = "passed"
    finally:
        (output / "kernel-hub-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

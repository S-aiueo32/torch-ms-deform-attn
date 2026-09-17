"""Run several Python/PyTorch pairs on one disposable matching-toolkit GPU host.

Requires uv on PATH. Results are flattened for the Runpod artifact collector;
each pair still uses fresh environments and the full installed-wheel harness.
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

VERSIONS = (
    "2.4.0",
    "2.5.0",
    "2.5.1",
    "2.7.1",
    "2.8.0",
    "2.9.1",
    "2.10.0",
    "2.11.0",
    "2.12.1",
    "2.13.0",
    "2.14.0",
)


def parse_case(value):
    parts = value.split(":")
    if (
        len(parts) != 2
        or parts[0] not in VERSIONS
        or parts[1] not in ("3.10", "3.11", "3.12", "3.13", "3.14")
    ):
        raise argparse.ArgumentTypeError("Expected supported TORCH:PYTHON pair")
    return tuple(parts)


def toolkit(version):
    return "12.4" if version in VERSIONS[:3] else "12.6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", type=parse_case, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len({toolkit(version) for version, _ in args.cases}) != 1:
        parser.error("All cases must use the same CUDA toolkit")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    source = Path(__file__).resolve().parents[1]
    for version, python in args.cases:
        label = f"torch-{version}-python-{python}"
        print(f"Starting {label}", flush=True)
        with tempfile.TemporaryDirectory(prefix="msda-matrix-") as temporary:
            root = Path(temporary)
            output = root / "results"
            output.mkdir()
            with (args.output / f"{label}-driver.log").open("w") as log:
                result = subprocess.run(
                    ["uv", "venv", "--seed", "--python", python, str(root / "base")],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                if result.returncode == 0:
                    env = dict(
                        os.environ,
                        CUDA_CHECKS_TORCH_VERSION=version,
                        CUDA_CHECKS_PYTHON=str(root / "base/bin/python"),
                        CUDA_CHECKS_CPU_SUITE="1",
                    )
                    result = subprocess.run(
                        ["bash", "scripts/run_cuda_checks.sh", "none", str(output)],
                        cwd=source,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            for artifact in output.iterdir():
                if artifact.is_file():
                    shutil.copy2(artifact, args.output / f"{label}-{artifact.name}")
            rows.append({"torch": version, "python": python, "exit_code": result.returncode})
            (args.output / "matrix-summary.json").write_text(
                json.dumps(
                    {"source_sha": os.environ["CUDA_CHECKS_SOURCE_SHA"], "cases": rows}, indent=2
                )
                + "\n"
            )
            print(f"Finished {label}: exit {result.returncode}", flush=True)
    return int(any(row["exit_code"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())

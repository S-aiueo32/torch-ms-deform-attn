"""Build/load/test backend transitions in one temporary source tree."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="msda-build-validation-") as directory:
        root = Path(directory)
        for name in ("setup.py", "pyproject.toml", "README.md", "LICENSE", "NOTICE"):
            shutil.copy2(source / name, root / name)
        for name in ("csrc", "src", "tests"):
            shutil.copytree(
                source / name, root / name, ignore=shutil.ignore_patterns("*.so", "__pycache__")
            )
        for name, options, backend, success in (
            ("auto-ninja", {"USE_NINJA": "1"}, "openmp", True),
            ("serial-distutils", {"FORCE_OPENMP": "0", "USE_NINJA": "0"}, "serial", True),
            (
                "auto-fallback",
                {"OMP_PREFIX": "/nonexistent/msda", "USE_NINJA": "0"},
                "serial",
                True,
            ),
            (
                "required-failure",
                {"OMP_PREFIX": "/nonexistent/msda", "FORCE_OPENMP": "1"},
                None,
                False,
            ),
            ("restored-openmp", {"FORCE_OPENMP": "1", "USE_NINJA": "1"}, "openmp", True),
        ):
            env = {
                k: v
                for k, v in os.environ.items()
                if k
                not in (
                    "FORCE_CPU",
                    "FORCE_CUDA",
                    "FORCE_OPENMP",
                    "OMP_PREFIX",
                    "USE_NINJA",
                    "PYTHONPATH",
                )
            }
            env.update(
                FORCE_CPU="1", FORCE_CUDA="0", MAX_JOBS="2", PYTHONPATH=str(root / "src"), **options
            )
            with (args.output / f"{name}.log").open("w") as log:
                result = subprocess.run(
                    [sys.executable, "setup.py", "build_ext", "--inplace"],
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                if (result.returncode == 0) != success:
                    raise RuntimeError(f"Unexpected build result: {name}")
                if success:
                    subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            f"from torch_ms_deform_attn import _C; assert _C.cpu_parallel_backend == {backend!r}, _C.cpu_parallel_backend; print(_C.__file__, _C.cpu_parallel_backend)",
                        ],
                        cwd=root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "unittest",
                            "discover",
                            "-s",
                            "tests",
                            "-p",
                            "test_cpu.py",
                            "-v",
                        ],
                        cwd=root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                else:
                    log.flush()
                    if (
                        "FORCE_OPENMP=1 could not enable OpenMP"
                        not in (args.output / f"{name}.log").read_text()
                    ):
                        raise RuntimeError("Build failed for an unexpected reason")
            rows.append(
                dict(case=name, expected_backend=backend, build_success=success, result="passed")
            )
            print(rows[-1], flush=True)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

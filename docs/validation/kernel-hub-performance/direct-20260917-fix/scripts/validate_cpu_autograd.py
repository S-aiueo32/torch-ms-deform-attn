"""Record focused native-autograd checks and the exact imported binaries."""

import argparse
import contextlib
import hashlib
import io
import json
import platform
import sys
import unittest
from pathlib import Path

import torch

import torch_ms_deform_attn
from torch_ms_deform_attn import _C

parser = argparse.ArgumentParser()
parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
parser.add_argument("--output-prefix", required=True, type=Path)
parser.add_argument("--expected-package-root", required=True, type=Path)
parser.add_argument("--test-pattern", default="test_integration.py")
args = parser.parse_args()
ROOT = args.repo_root.resolve()
assert Path(torch_ms_deform_attn.__file__).parent.resolve() == args.expected_package_root.resolve()
assert Path(_C.__file__).parent.resolve() == args.expected_package_root.resolve()


def fingerprint(path):
    path = Path(path).resolve()
    return dict(
        path=str(path),
        size=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []

    def addSuccess(self, test):
        self.passed.append(test.id())
        super().addSuccess(test)


sys.path.insert(0, str(ROOT / "tests"))
suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), args.test_pattern)
output = io.StringIO()
with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
    result = unittest.TextTestRunner(stream=output, verbosity=2, resultclass=RecordedResult).run(
        suite
    )
source_paths = [
    "csrc/dispatcher.h",
    "csrc/vision.cpp",
    "src/torch_ms_deform_attn/_ops.py",
    "src/torch_ms_deform_attn/functional.py",
    "tests/test_integration.py",
]
if args.test_pattern == "test_mps.py":
    source_paths.extend(["tests/test_mps.py", "csrc/mps/ms_deform_attn_mps.mm"])
report = dict(
    torch=torch.__version__,
    python=sys.version,
    platform=platform.platform(),
    test_pattern=args.test_pattern,
    mps={
        "built": torch.backends.mps.is_built(),
        "available": torch.backends.mps.is_available(),
        "extension_with_mps": _C.with_mps,
    },
    package=fingerprint(torch_ms_deform_attn.__file__),
    extension=fingerprint(_C.__file__),
    imported_python={
        name: fingerprint(args.expected_package_root / name)
        for name in ("_ops.py", "functional.py")
    },
    source_files={name: fingerprint(ROOT / name) for name in source_paths},
    tests_run=result.testsRun,
    successful=result.wasSuccessful(),
    passed=result.passed,
    failures=[(test.id(), detail) for test, detail in result.failures],
    errors=[(test.id(), detail) for test, detail in result.errors],
    skipped=[(test.id(), reason) for test, reason in result.skipped],
)
args.output_prefix.with_suffix(".log").write_text(output.getvalue())
args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "torch": report["torch"],
            "extension": report["extension"]["path"],
            "tests_run": result.testsRun,
            "successful": result.wasSuccessful(),
            "output": str(args.output_prefix),
        },
        indent=2,
    )
)
raise SystemExit(not result.wasSuccessful())

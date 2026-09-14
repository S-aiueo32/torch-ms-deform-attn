"""Run installed-wheel tests and fail closed on unexpected skips."""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import unittest
from pathlib import Path

ALLOWED_SKIPS = {
    "test_cuda.CUDAAttentionTest.test_noncurrent_device": "Requires two GPUs",
    "test_cuda.CPUOnlyBuildTest.test_cuda_input_error": "Requires GPU and CPU-only build",
}


def unexpected_skips(skipped):
    return [
        (test.id(), reason) for test, reason in skipped if ALLOWED_SKIPS.get(test.id()) != reason
    ]


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_passed = []

    def addSuccess(self, test):
        super().addSuccess(test)
        if test.id().startswith("test_cuda.CUDAAttentionTest."):
            self.cuda_passed.append(test.id())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from torch_ms_deform_attn import _C

    sha = os.environ.get("CUDA_CHECKS_SOURCE_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError("CUDA_CHECKS_SOURCE_SHA must be the full archived source SHA")
    assert torch.cuda.is_available() and _C.with_cuda, "GPU and CUDA wheel are required"
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
    unexpected = unexpected_skips(result.skipped)
    success = result.wasSuccessful() and not unexpected and bool(result.cuda_passed)
    report = {
        "source_sha": sha,
        "success": success,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "toolkit": subprocess.check_output(["nvcc", "--version"], text=True).strip(),
        "build": {
            key: os.environ.get(key, "auto")
            for key in (
                "FORCE_CPU",
                "FORCE_CUDA",
                "FORCE_OPENMP",
                "MAX_JOBS",
                "TORCH_CUDA_ARCH_LIST",
            )
        },
        "cpu_backend": _C.cpu_parallel_backend,
        "cuda_passed": result.cuda_passed,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": [(test.id(), reason) for test, reason in result.skipped],
        "unexpected_skips": unexpected,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

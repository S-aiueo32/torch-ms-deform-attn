"""Record installed CPU-wheel tests, rejecting failures and unexpected skips."""

import argparse
import json
import os
import platform
import sys
import unittest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from torch_ms_deform_attn import _C

    assert not _C.with_cuda
    assert _C.cpu_parallel_backend == "openmp"
    assert Path(_C.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover("tests"))
    unexpected = [
        (test.id(), reason)
        for test, reason in result.skipped
        if reason != "Requires CUDA extension and GPU"
        and not (
            test.id() == "test_cuda.CPUOnlyBuildTest.test_cuda_input_error"
            and reason == "Requires GPU and CPU-only build"
        )
    ]
    success = result.wasSuccessful() and not unexpected and result.testsRun > len(result.skipped)
    args.output.write_text(
        json.dumps(
            {
                "source_sha": os.environ["CUDA_CHECKS_SOURCE_SHA"],
                "success": success,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "torch": torch.__version__,
                "with_cuda": _C.with_cuda,
                "cpu_backend": _C.cpu_parallel_backend,
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": [(test.id(), reason) for test, reason in result.skipped],
                "unexpected_skips": unexpected,
            },
            indent=2,
        )
        + "\n"
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

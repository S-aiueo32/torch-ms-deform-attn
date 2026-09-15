"""Strict mode must preserve diagnostics and fail the process result."""

import contextlib
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
SPEC = importlib.util.spec_from_file_location(
    "benchmark_compile", ROOT / "benchmarks/benchmark_compile.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkFailureTest(unittest.TestCase):
    def test_strict_compile_failure(self):
        for strict in (False, True):
            args = ["benchmark_compile", "--min-run-time", "0.001"] + (
                ["--strict"] if strict else []
            )
            with (
                patch.object(sys, "argv", args),
                patch.object(benchmark, "CASES", [("tiny", 1, 2, 1, 2, [(2, 2)])]),
                patch.object(
                    benchmark.torch, "compile", side_effect=RuntimeError("compile failed")
                ),
                contextlib.redirect_stdout(io.StringIO()) as output,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = benchmark.main()
            report = json.loads(output.getvalue())
            self.assertEqual(code, int(strict))
            failed = [row for row in report["results"] if "error" in row]
            self.assertEqual(len(failed), 4)
            self.assertTrue(all("median_ms" not in row for row in failed))
            self.assertTrue(all("compile failed" in row["error"] for row in failed))

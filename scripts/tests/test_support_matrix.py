"""Validate matrix inputs before any Pod is rented."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_support_matrix as matrix

SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("matrix_controller", SCRIPTS / "runpod_ci.py")
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


class MatrixArgumentsTest(unittest.TestCase):
    def parse(self, *extra):
        return ci.parse_args(
            [
                "run",
                "--repository",
                "owner/repo",
                "--repository-id",
                "1",
                "--run-id",
                "2",
                "--attempt",
                "1",
                "--source",
                ".",
                "--state-dir",
                "/tmp/matrix-state",
                "--output-dir",
                "/tmp/matrix-output",
                *extra,
            ]
        )

    def test_matching_toolkit(self):
        args = self.parse(
            "--torch-version", "2.14.0", "--matrix-cases", "2.9.1:3.10", "2.14.0:3.14"
        )
        self.assertEqual(len(args.matrix_cases), 2)

    def test_reject_mixed_toolkits_and_unsafe_input(self):
        for case in ("2.14.0:3.12", "2.5.1:3.11;id", "2.5.1:3.13t", "2.4.0:3.13"):
            with self.subTest(case=case), self.assertRaises(SystemExit):
                self.parse("--matrix-cases", case)

    def test_reject_duplicate_cases_and_incompatible_options(self):
        for extra in (
            ["--matrix-cases", "2.5.1:3.11", "2.5.1:3.11"],
            ["--sanitizer", "all", "--matrix-cases", "2.5.1:3.11"],
            ["--gpu-count", "2", "--matrix-cases", "2.5.1:3.11"],
        ):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.parse(*extra)


class MatrixExecutionTest(unittest.TestCase):
    def test_failed_case_does_not_drop_later_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence"

            def run(command, **kwargs):
                if command[0] == "uv" and "3.10" in command:
                    return subprocess.CompletedProcess(command, 1)
                if command[0] == "bash":
                    (Path(command[-1]) / "result.json").write_text('{"success":true}')
                return subprocess.CompletedProcess(command, 0)

            argv = [
                "matrix",
                "--cases",
                "2.4.0:3.10",
                "2.4.0:3.11",
                "--output",
                str(output),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(os.environ, CUDA_CHECKS_SOURCE_SHA="a" * 40),
                mock.patch.object(matrix.subprocess, "run", side_effect=run),
            ):
                self.assertEqual(matrix.main(), 1)
            summary = json.loads((output / "matrix-summary.json").read_text())
            self.assertEqual([row["exit_code"] for row in summary["cases"]], [1, 0])
            self.assertTrue((output / "torch-2.4.0-python-3.11-result.json").is_file())


if __name__ == "__main__":
    unittest.main()

"""Validate matrix inputs before any Pod is rented."""

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
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
        for case in ("2.14.0:3.12", "2.5.1:3.11;id", "2.5.1:3.13t"):
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


if __name__ == "__main__":
    unittest.main()

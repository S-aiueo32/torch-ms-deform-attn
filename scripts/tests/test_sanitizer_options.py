"""Sanitizer requests must survive CLI parsing before any Pod is rented."""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runpod_ci


class SanitizerOptionsTest(unittest.TestCase):
    def test_all_supported_tools_and_invalid_name(self):
        argv = [
            "run",
            "--repository",
            "example/repo",
            "--repository-id",
            "1",
            "--run-id",
            "2",
            "--attempt",
            "1",
            "--state-dir",
            "/tmp/ci-state",
            "--output-dir",
            "/tmp/ci-results",
            "--source",
            "/tmp/source",
            "--sanitizer",
        ]
        for tool in ("none", "memcheck", "racecheck", "synccheck", "initcheck", "all"):
            self.assertEqual(runpod_ci.parse_args(argv + [tool]).sanitizer, tool)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runpod_ci.parse_args(argv + ["unknown"])

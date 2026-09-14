"""Release evidence must reject CPU-only, skipped, incomplete and wrong-SHA runs."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuda_test_report import RecordingResult, unexpected_skips
from verify_cuda_release import verify


class EvidenceTest(unittest.TestCase):
    def test_only_explicit_skip_ids_and_reasons_are_allowed(self):
        class Case:
            def __init__(self, name):
                self.name = name

            def id(self):
                return self.name

        allowed = Case("test_cuda.CUDAAttentionTest.test_noncurrent_device")
        self.assertEqual(unexpected_skips([(allowed, "Requires two GPUs")]), [])
        self.assertTrue(unexpected_skips([(allowed, "Requires CUDA extension and GPU")]))
        self.assertTrue(unexpected_skips([(Case("test_cuda.other"), "Requires two GPUs")]))

    def test_cpu_success_is_not_gpu_evidence(self):
        result = RecordingResult(io.StringIO(), True, 0)
        result.addSuccess(unittest.FunctionTestCase(lambda: None))
        self.assertEqual(result.cuda_passed, [])

    def test_evidence_rejections(self):
        sha = "a" * 40
        good = dict(
            source_sha=sha,
            success=True,
            failures=0,
            errors=0,
            unexpected_skips=[],
            gpu=["GPU"],
            cuda_passed=["test"],
            tests_run=3,
            skipped=[],
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "cuda-checks.log").write_text("log")
            (root / "cuda-completion.json").write_text(
                json.dumps(dict(source_sha=sha, success=True))
            )
            for change in (
                {},
                {"source_sha": "b" * 40},
                {"success": False},
                {"gpu": []},
                {"cuda_passed": []},
                {"tests_run": 0},
                {"unexpected_skips": ["test"]},
            ):
                (root / "cuda-tests.json").write_text(json.dumps(good | change))
                if change:
                    with self.assertRaises(ValueError):
                        verify(root, sha)
                else:
                    verify(root, sha)
            (root / "cuda-tests.json").write_text(json.dumps(good))
            (root / "cuda-completion.json").unlink()
            with self.assertRaises(FileNotFoundError):
                verify(root, sha)

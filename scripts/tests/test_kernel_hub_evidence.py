"""Reject incomplete or mixed Kernel Hub evidence even if its summary says passed."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_kernel_hub import PRECISIONS, verify

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / "docs/validation/kernel-hub/run-35197412721"


class KernelHubEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        upstream = json.loads((HISTORICAL / "UPSTREAM.json").read_text())
        self.sha = upstream["revision"]
        self.files = {
            name: json.loads((HISTORICAL / name).read_text())
            for name in (
                "UPSTREAM.json",
                "candidate-files.json",
                "baseline-files.json",
                "runpod-state.json",
            )
        }
        template = json.loads((HISTORICAL / "e2e-bf16-amp.json").read_text())
        runs = [{"name": "phase1", "exit_code": 0}]
        for dtype, amp in PRECISIONS:
            name = f"e2e-{dtype}" + ("-amp" if amp else "")
            report = copy.deepcopy(template)
            report["cases"] = []
            for training in (False, True):
                for backend in (None, "inductor"):
                    case = copy.deepcopy(template["cases"][0])
                    case.update(
                        training=training, compile_backend=backend, dtype=dtype, autocast=amp
                    )
                    report["cases"].append(case)
            self.files[f"{name}.json"] = report
            runs.append({"name": name, "exit_code": 0})
        self.files["kernel-hub-summary.json"] = {
            "source_sha": self.sha,
            "status": "passed",
            "focus_compile": False,
            "phase1_only": False,
            "gpu": "L4",
            "cuda": "12.6",
            "torch": template["torch"],
            "baseline_revision": "a" * 40,
            "runs": runs,
        }
        self.files["phase1-numerics.json"] = {"fixture": True}
        for name in ("kernel-hub-checks.log", "python-packages.txt", "nvidia-smi.txt"):
            (self.directory / name).write_text("fixture\n")
        (self.directory / "phase1.log").write_text("Ran 3 tests in 1.0s\n\nOK\n")

    def write(self, files):
        for name, content in files.items():
            (self.directory / name).write_text(json.dumps(content))

    def test_complete_and_corrupt_evidence(self):
        self.write(self.files)
        verify(self.directory, self.sha)
        mutations = (
            lambda f: f["kernel-hub-summary.json"].update(source_sha="b" * 40),
            lambda f: f["kernel-hub-summary.json"].update(focus_compile=True),
            lambda f: f["kernel-hub-summary.json"]["runs"].pop(),
            lambda f: f["kernel-hub-summary.json"]["runs"][0].update(exit_code=1),
            lambda f: f["runpod-state.json"].update(phase="running"),
            lambda f: f["e2e-fp32.json"]["cases"].pop(),
            lambda f: f["e2e-fp32.json"].update(harness_sha256="b" * 64),
            lambda f: f["e2e-fp32.json"]["cases"][0].update(error="failure"),
            lambda f: f["e2e-fp32.json"]["cases"][0].update(candidate_sha256={}),
            lambda f: f["e2e-fp32.json"]["cases"][0].update(executed_ops={}),
            lambda f: f["e2e-fp32.json"]["cases"][1].update(compiled_graphs=[]),
            lambda f: f["e2e-fp32.json"]["cases"][2].update(gradient_tensor_count=0),
            lambda f: f["e2e-fp16.json"]["cases"][0].update(autocast=True),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                files = copy.deepcopy(self.files)
                mutate(files)
                self.write(files)
                with self.assertRaises(ValueError):
                    verify(self.directory, self.sha)

    def test_missing_and_skipped_phase1(self):
        self.write(self.files)
        (self.directory / "phase1.log").write_text("Ran 3 tests in 1.0s\n\nOK (skipped=3)\n")
        with self.assertRaisesRegex(ValueError, "Phase 1"):
            verify(self.directory, self.sha)
        (self.directory / "phase1.log").unlink()
        with self.assertRaises(FileNotFoundError):
            verify(self.directory, self.sha)

    def test_historical_partial_suite_is_rejected(self):
        with self.assertRaises(ValueError):
            verify(HISTORICAL, self.sha)

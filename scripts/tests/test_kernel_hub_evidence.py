"""Reject incomplete or mixed Kernel Hub evidence even if its summary says passed."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_kernel_hub import PRECISIONS, verify


class KernelHubEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.sha = "1" * 40
        harness_sha256 = "2" * 64
        candidate = {"candidate.so": "3" * 64}
        baseline = {"baseline.so": "4" * 64}
        upstream = {
            "revision": self.sha,
            "source_sha256": {
                "csrc/dispatcher.h": "5" * 64,
                "kernel-hub/e2e/rt_detr.py": harness_sha256,
            },
        }
        self.files = {
            "UPSTREAM.json": upstream,
            "candidate-files.json": candidate,
            "baseline-files.json": baseline,
            "runpod-state.json": {"phase": "deleted"},
        }
        template = {
            "status": "passed",
            "torch": "2.10.0+cu126",
            "harness_sha256": harness_sha256,
            "cases": [
                {
                    "reference": "HF kernel",
                    "candidate_sha256": candidate,
                    "baseline_sha256": baseline,
                    "executed_ops": {"test::forward": 1, "test::backward": 1},
                    "gradient_tensor_count": 1,
                    "reference_compiled": True,
                    "compiled_graphs": [{"msda_calls": 1}],
                }
            ],
        }
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

    def test_nix_artifact_identity(self):
        original = {
            "revision": "6" * 40,
            "source_sha256": {
                "csrc/dispatcher.h": "5" * 64,
                "kernel-hub/e2e/rt_detr.py": "7" * 64,
            },
        }
        variant = "torch11-cu126-x86_64"
        manifest = {"lib/candidate.so": "8" * 64}
        self.files["UPSTREAM.json"] = copy.deepcopy(original)
        self.files["UPSTREAM.json"]["revision"] = self.sha
        # An independently hashed compiler-compatibility harness may change
        # without rebuilding the pinned native implementation.
        harness = "kernel-hub/e2e/rt_detr.py"
        self.files["UPSTREAM.json"]["source_sha256"][harness] = "f" * 64
        self.files["nix-UPSTREAM.json"] = original
        self.files["candidate-files.json"] = manifest
        nix_build = {
            "build_run": "35280414318",
            "artifact_source_sha": original["revision"],
            "archive_sha256": "9" * 64,
            "variant": variant,
            "files": manifest,
        }
        self.files["NIX_BUILD.json"] = nix_build
        self.files["kernel-hub-summary.json"]["nix_build"] = nix_build
        for name, report in self.files.items():
            if name.startswith("e2e-"):
                report["harness_sha256"] = "f" * 64
                for case in report["cases"]:
                    case["candidate_sha256"] = manifest
        self.write(self.files)
        verify(self.directory, self.sha)
        for key, value in (("archive_sha256", "invalid"), ("files", {})):
            with self.subTest(key=key):
                files = copy.deepcopy(self.files)
                files["NIX_BUILD.json"][key] = value
                self.write(files)
                with self.assertRaises(ValueError):
                    verify(self.directory, self.sha)

        files = copy.deepcopy(self.files)
        files["UPSTREAM.json"]["source_sha256"]["csrc/dispatcher.h"] = "0" * 64
        self.write(files)
        with self.assertRaisesRegex(ValueError, "Nix sources changed"):
            verify(self.directory, self.sha)

    def test_historical_partial_suite_is_rejected(self):
        files = copy.deepcopy(self.files)
        files["kernel-hub-summary.json"]["phase1_only"] = True
        self.write(files)
        with self.assertRaises(ValueError):
            verify(self.directory, self.sha)

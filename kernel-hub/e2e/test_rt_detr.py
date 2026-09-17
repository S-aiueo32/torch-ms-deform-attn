"""CPU fixture tests, never evidence of native CUDA or builder compatibility.

Run separately from core tests with the pinned E2E dependencies installed and
the upstream CPU extension importable.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from rt_detr import align_proposals, run_case, run_model, tiny_model

from torch_ms_deform_attn import _C

ROOT = Path(__file__).resolve().parents[2]


class ProposalAlignmentTest(unittest.TestCase):
    def test_only_permutation_is_normalized(self):
        expected = {
            "proposals": torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]),
            "logits": torch.tensor([[[4.0], [5.0]]]),
            "boxes": torch.tensor([[[6.0], [7.0]]]),
            "grad:weight": torch.tensor([8.0]),
        }
        actual = {
            name: value.flip(1) if value.ndim == 3 else value.clone()
            for name, value in expected.items()
        }
        aligned, order = align_proposals(actual, expected, atol=0.02, rtol=0.05)
        self.assertEqual(order, [[1, 0]])
        for name in expected:
            torch.testing.assert_close(aligned[name], expected[name], atol=0, rtol=0)
        self.assertIs(aligned["grad:weight"], actual["grad:weight"])
        # Never match a different proposal or hide a prediction error.
        actual["logits"] += 1
        aligned, _ = align_proposals(actual, expected, atol=0.02, rtol=0.05)
        self.assertFalse(torch.equal(aligned["logits"], expected["logits"]))
        actual["proposals"][0, 0, 0] += 0.01
        with self.assertRaises(AssertionError):
            align_proposals(actual, expected, atol=0.02, rtol=0.05)


class RTDetrCPUFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        spec = importlib.util.spec_from_file_location("e2e_export", ROOT / "kernel-hub/export.py")
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        export_dir = Path(cls.temp.name) / "export"
        exporter.export(export_dir)
        cls.provenance = {
            "export": json.loads((export_dir / "UPSTREAM.json").read_text()),
            "harness_sha256": hashlib.sha256(
                (ROOT / "kernel-hub/e2e/rt_detr.py").read_bytes()
            ).hexdigest(),
            "fixture_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "native_cpu_extension_sha256": hashlib.sha256(
                Path(_C.__file__).read_bytes()
            ).hexdigest(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "torchvision",
                    "transformers",
                    "kernels",
                    "kernels-data",
                    "scipy",
                )
            },
        }
        cls.kernel_dir = Path(cls.temp.name) / "cpu-fixture"
        shutil.copytree(
            export_dir / "torch-ext/deformable_detr", cls.kernel_dir / "deformable_detr"
        )
        (cls.kernel_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "name": "deformable-detr",
                    "id": "msda_cpu_e2e_fixture",
                    "version": 1,
                    "license": "Apache-2.0",
                    "python-depends": [],
                    "backend": {"type": "cpu"},
                }
            )
        )
        (cls.kernel_dir / "deformable_detr/_ops.py").write_text(
            """import torch
from torch_ms_deform_attn import _C as ops
from torch_ms_deform_attn._ops import forward, backward

def add_op_namespace_prefix(name):
    return "msda_cpu_e2e_fixture::" + name

# This CPU fixture delegates autograd to the canonical native operators.
# The real Kernel Hub namespace and CUDA binding need separate GPU validation.
signature = "Tensor value, Tensor shapes, Tensor starts, Tensor locations, Tensor weights"
torch.library.define(add_op_namespace_prefix("forward"), f"({signature}, int step, bool check_cuda_metadata=False) -> Tensor")
torch.library.impl(add_op_namespace_prefix("forward"), "CPU", ops.ms_deform_attn_forward)
torch.library.impl(add_op_namespace_prefix("forward"), "Autograd", forward)
torch.library.define(
    add_op_namespace_prefix("backward"),
    f"({signature}, Tensor grad, int step, bool check_cuda_metadata=False) -> (Tensor, Tensor, Tensor)",
)
torch.library.impl(
    add_op_namespace_prefix("backward"), "CPU",
    lambda *args: tuple(ops.ms_deform_attn_backward(*args)),
)
torch.library.impl(add_op_namespace_prefix("backward"), "Autograd", backward)
"""
        )

    def test_detection_loss_matches_public_forward(self):
        torch.manual_seed(42)
        model = tiny_model().train()
        pixels = torch.randn(2, 3, 64, 64)
        labels = [
            {"class_labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]])}
            for _ in range(2)
        ]
        expected = model(pixel_values=pixels, labels=labels).loss.detach()
        actual = run_model(model, pixels, labels, training=True)["loss"]
        torch.testing.assert_close(actual, expected)

    def test_refuses_same_baseline(self):
        with self.assertRaisesRegex(ValueError, "same kernel module"):
            run_case(self.kernel_dir, baseline_kernel_dir=self.kernel_dir, device="cpu")

    def test_full_detector(self):
        cases = []
        try:
            for dtype, amp in (
                ("fp32", False),
                ("fp16", False),
                ("bf16", False),
                ("fp16", True),
                ("bf16", True),
            ):
                for training in (False, True):
                    for backend in (None, "aot_eager"):
                        with self.subTest(dtype=dtype, amp=amp, training=training, backend=backend):
                            result = run_case(
                                self.kernel_dir,
                                training=training,
                                backend=backend,
                                device="cpu",
                                dtype=dtype,
                                amp=amp,
                            )
                            self.assertEqual(len(result["replaced_layers"]), 2)
                            if training:
                                self.assertGreater(result["gradient_tensor_count"], 0)
                            cases.append(result)
        finally:
            if path := os.environ.get("MSDA_CPU_FIXTURE_REPORT"):
                Path(path).write_text(
                    json.dumps(
                        {
                            "scope": "CPU fixture only; no CUDA build or HF artifact validation",
                            "torch": torch.__version__,
                            "expected_cases": 20,
                            "passed_cases": len(cases),
                            "cases": cases,
                            "provenance": self.provenance,
                        },
                        indent=2,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    unittest.main()

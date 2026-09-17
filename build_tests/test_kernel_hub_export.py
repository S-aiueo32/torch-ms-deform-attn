"""CPU checks of exported Python glue; these do not validate the HF builder/CUDA."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from torch_ms_deform_attn import _C, ms_deform_attn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("msda_export", ROOT / "kernel-hub/export.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportTest(unittest.TestCase):
    def test_independent_namespaces_and_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "kernel"
            exporter.export(target)
            manifest = json.loads((target / "UPSTREAM.json").read_text())
            self.assertIn("csrc/cuda/ms_deform_attn_cuda.cu", manifest["source_sha256"])
            for name in ("msda_hf_test_a", "msda_hf_test_b"):
                with self.subTest(namespace=name):
                    # Emulate only builder-generated _ops using the installed CPU extension.
                    # Loader, C++ binding and CUDA validation live in kernel-hub/tests.
                    ops = types.ModuleType(f"{name}._ops")
                    ops.ops = _C
                    ops.add_op_namespace_prefix = lambda op, ns=name: f"{ns}::{op}"
                    sys.modules[ops.__name__] = ops
                    init = target / "torch-ext/deformable_detr/__init__.py"
                    spec = importlib.util.spec_from_file_location(name, init)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    spec.loader.exec_module(module)
                    layer = module.layers.MultiScaleDeformableAttention()
                    for dtype in (torch.float32, torch.float16, torch.bfloat16):
                        value = torch.randn(1, 4, 2, 3, dtype=dtype, requires_grad=True)
                        loc = torch.rand(1, 5, 2, 1, 2, 2, dtype=dtype, requires_grad=True)
                        weights = torch.rand(1, 5, 2, 1, 2, dtype=dtype, requires_grad=True)
                        shapes, starts = torch.tensor([[2, 2]]), torch.tensor([0])
                        reference = ms_deform_attn(value, shapes, starts, loc, weights)
                        for fn in (
                            layer,
                            torch.compile(layer, backend="aot_eager", fullgraph=True),
                        ):
                            output = fn(value, shapes, [(2, 2)], starts, loc, weights, 64)
                            torch.testing.assert_close(output, reference)
                            args = (value, loc, weights)
                            actual = torch.autograd.grad(output.sum(), args, retain_graph=True)
                            expected = torch.autograd.grad(reference.sum(), args, retain_graph=True)
                            for a, b in zip(actual, expected):
                                torch.testing.assert_close(a, b)
                        explicit = module.ms_deform_attn_forward(
                            value, shapes, starts, loc, weights, 64
                        )
                        torch.testing.assert_close(explicit, reference)
                        gradients = module.ms_deform_attn_backward(
                            value, shapes, starts, loc, weights, torch.ones_like(explicit), 64
                        )
                        self.assertIsInstance(gradients, list)
                        for actual_grad, expected_grad in zip(gradients, expected):
                            torch.testing.assert_close(actual_grad, expected_grad)

    def test_refuse_overwrite_and_changed_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                exporter.export(tmp)
        with self.assertRaises(ValueError):
            exporter.replace_once("changed", "original", "replacement")

    def test_archive_revision_without_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(exporter.subprocess, "check_output") as git:
                destination = Path(tmp) / "kernel"
                exporter.export(destination, revision="a" * 40)
            git.assert_not_called()
            self.assertEqual(
                json.loads((destination / "UPSTREAM.json").read_text())["revision"], "a" * 40
            )
            with self.assertRaisesRegex(ValueError, "full Git commit SHA"):
                exporter.export(Path(tmp) / "invalid", revision="main")


if __name__ == "__main__":
    unittest.main()

"""Run inside a builder testshell with LOCAL_KERNELS pointing at this build."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


class KernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("Kernel Hub validation requires CUDA; a skip is not evidence")
        # Explicit local path prevents accidentally validating the published baseline.
        from kernels import get_kernel

        if not os.environ.get("LOCAL_KERNELS"):
            raise RuntimeError("Run in a builder testshell with LOCAL_KERNELS configured")
        cls.kernel = get_kernel("kernels-community/deformable-detr", version=1)
        if not hasattr(cls.kernel, "_registrations"):
            raise RuntimeError("Loader did not resolve the upstream adapter; check LOCAL_KERNELS")

    def test_pinned_hf_baseline(self):
        revision = os.environ["MSDA_HF_BASELINE_REVISION"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("MSDA_HF_BASELINE_REVISION must be a full Hub commit SHA")
        # A separate process prevents LOCAL_KERNELS and loader caching from
        # accidentally comparing the new implementation with itself.
        script = """
import sys
import torch
from kernels import get_kernel
kernel = get_kernel("kernels-community/deformable-detr", revision=sys.argv[1])
value, shapes, starts, locations, weights, grad = torch.load(sys.argv[2], weights_only=True)
out = kernel.ms_deform_attn_forward(value, shapes, starts, locations, weights, 64)
grads = kernel.ms_deform_attn_backward(value, shapes, starts, locations, weights, grad, 64)
torch.save((out, grads), sys.argv[3])
"""
        env = {key: val for key, val in os.environ.items() if key != "LOCAL_KERNELS"}
        torch.manual_seed(29)
        for dtype in (torch.float32, torch.float64):
            value = torch.randn(2, 20, 2, 8, dtype=dtype, device="cuda")
            shapes = torch.tensor([[4, 4], [2, 2]], device="cuda")
            starts = torch.tensor([0, 16], device="cuda")
            locations = torch.rand(2, 7, 2, 2, 4, 2, dtype=dtype, device="cuda")
            weights = torch.rand(2, 7, 2, 2, 4, dtype=dtype, device="cuda")
            grad = torch.randn(2, 7, 16, dtype=dtype, device="cuda")
            args = (value, shapes, starts, locations, weights)
            with tempfile.TemporaryDirectory() as tmp:
                inputs, outputs = Path(tmp) / "inputs.pt", Path(tmp) / "outputs.pt"
                torch.save((*args, grad), inputs)
                subprocess.run(
                    [sys.executable, "-c", script, revision, str(inputs), str(outputs)],
                    env=env,
                    check=True,
                )
                reference, ref_grads = torch.load(outputs, weights_only=True)
            actual = self.kernel.ms_deform_attn_forward(*args, 64)
            grads = self.kernel.ms_deform_attn_backward(*args, grad, 64)
            torch.testing.assert_close(actual, reference)
            for actual_grad, reference_grad in zip(grads, ref_grads):
                torch.testing.assert_close(actual_grad, reference_grad, atol=1e-5, rtol=1e-5)

    def test_autocast_and_validation(self):
        for dtype in (torch.float16, torch.bfloat16):
            value = torch.randn(1, 4, 2, 3, device="cuda", dtype=dtype, requires_grad=True)
            locations = torch.rand(1, 5, 2, 1, 2, 2, device="cuda", dtype=dtype, requires_grad=True)
            weights = torch.rand(1, 5, 2, 1, 2, device="cuda", dtype=dtype, requires_grad=True)
            shapes, starts = torch.tensor([[2, 2]], device="cuda"), torch.tensor([0], device="cuda")
            fn = self.kernel.ms_deform_attn
            for compiled in (False, True):
                run = torch.compile(fn, fullgraph=True) if compiled else fn
                with torch.autocast("cuda", dtype=dtype):
                    output = run(value, shapes, starts, locations, weights)
                self.assertEqual(output.dtype, torch.float32)
                reference = fn(value.float(), shapes, starts, locations.float(), weights.float())
                torch.testing.assert_close(output, reference)
                inputs = (value, locations, weights)
                actual = torch.autograd.grad(output.sum(), inputs)
                expected = torch.autograd.grad(reference.sum(), inputs)
                for grad, target in zip(actual, expected):
                    torch.testing.assert_close(grad, target)
            with self.assertRaisesRegex(RuntimeError, "dtypes must match"):
                fn(value, shapes, starts, locations.float(), weights)
            with self.assertRaisesRegex(RuntimeError, "same CUDA device"):
                fn(value, shapes.cpu(), starts, locations, weights)

    def test_layer(self):
        from torch.nn.functional import grid_sample

        torch.manual_seed(17)
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            for compiled in (False, True):
                with self.subTest(dtype=dtype, compiled=compiled):
                    value = torch.randn(1, 4, 2, 3, device="cuda", dtype=dtype, requires_grad=True)
                    loc = torch.rand(
                        1, 5, 2, 1, 2, 2, device="cuda", dtype=dtype, requires_grad=True
                    )
                    weights = torch.rand(
                        1, 5, 2, 1, 2, device="cuda", dtype=dtype, requires_grad=True
                    )
                    shapes = torch.tensor([[2, 2]], device="cuda")
                    starts = torch.tensor([0], device="cuda")
                    layer = self.kernel.layers.MultiScaleDeformableAttention()
                    if compiled:
                        layer = torch.compile(layer, fullgraph=True)
                    actual = layer(value, shapes, [(2, 2)], starts, loc, weights, 64)
                    v, locations, w = (
                        x.float() if dtype in (torch.float16, torch.bfloat16) else x
                        for x in (value, loc, weights)
                    )
                    samples = grid_sample(
                        v.permute(0, 2, 3, 1).reshape(2, 3, 2, 2),
                        (2 * locations[:, :, :, 0] - 1).permute(0, 2, 1, 3, 4).reshape(2, 5, 2, 2),
                        align_corners=False,
                    )
                    ref = (samples * w.permute(0, 2, 1, 3, 4).reshape(2, 1, 5, 2)).sum(-1)
                    ref = ref.reshape(1, 6, 5).transpose(1, 2).to(dtype)
                    tol = (
                        2e-2
                        if dtype == torch.bfloat16
                        else 2e-3
                        if dtype == torch.float16
                        else 1e-5
                    )
                    torch.testing.assert_close(actual, ref, atol=tol, rtol=tol)
                    inputs = (value, loc, weights)
                    grads = torch.autograd.grad(actual.sum(), inputs, retain_graph=True)
                    expected = torch.autograd.grad(ref.sum(), inputs)
                    for grad, target in zip(grads, expected):
                        torch.testing.assert_close(grad, target, atol=tol, rtol=tol)


if __name__ == "__main__":
    unittest.main()

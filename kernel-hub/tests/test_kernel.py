"""Run inside a builder testshell with LOCAL_KERNELS pointing at this build."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


def double_reference(args, grad):
    """Independent two-level grid-sample oracle at the exact input coordinates."""
    value, shapes, starts, locations, weights = args
    value, locations, weights = (
        tensor.detach().double().requires_grad_() for tensor in (value, locations, weights)
    )
    batch, _, heads, channels = value.shape
    queries, points = locations.shape[1], locations.shape[4]
    sampled = []
    for level, ((height, width), start) in enumerate(zip(shapes.tolist(), starts.tolist())):
        feature = value[:, start : start + height * width]
        feature = feature.permute(0, 2, 3, 1).reshape(batch * heads, channels, height, width)
        grid = (2 * locations[:, :, :, level] - 1).permute(0, 2, 1, 3, 4)
        sample = torch.nn.functional.grid_sample(
            feature, grid.reshape(batch * heads, queries, points, 2), align_corners=False
        )
        sampled.append(sample)
    samples = torch.stack(sampled, dim=-2).flatten(-2)
    attention = weights.permute(0, 2, 1, 3, 4).reshape(batch * heads, 1, queries, -1)
    output = (samples * attention).sum(-1)
    output = output.reshape(batch, heads * channels, queries).transpose(1, 2).contiguous()
    gradients = torch.autograd.grad(output, (value, locations, weights), grad.double())
    return (output.detach(), *gradients)


def comparison(actual, expected, tolerance):
    difference = actual.double() - expected.double()
    return {
        "max_abs": difference.abs().max().item(),
        "rms": difference.square().mean().sqrt().item(),
        "close": torch.allclose(actual.double(), expected.double(), atol=tolerance, rtol=tolerance),
    }


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
if value.dtype in (torch.float16, torch.bfloat16):
    promoted = (value.float(), shapes, starts, locations.float(), weights.float())
    promoted_out = kernel.ms_deform_attn_forward(*promoted, 64).to(value.dtype)
    promoted_grads = [g.to(value.dtype) for g in kernel.ms_deform_attn_backward(*promoted, grad.float(), 64)]
else:
    promoted_out, promoted_grads = out, grads
torch.save((out, grads, promoted_out, promoted_grads), sys.argv[3])
"""
        env = {key: val for key, val in os.environ.items() if key != "LOCAL_KERNELS"}
        report = {"baseline_revision": revision, "cases": []}
        torch.manual_seed(29)
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
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
                reference, ref_grads, promoted, promoted_grads = torch.load(
                    outputs, weights_only=True
                )
            actual = self.kernel.ms_deform_attn_forward(*args, 64)
            grads = self.kernel.ms_deform_attn_backward(*args, grad, 64)
            tolerance = (
                2e-2 if dtype == torch.bfloat16 else 2e-3 if dtype == torch.float16 else 1e-5
            )
            oracle = double_reference(args, grad)
            tensors = {}
            for name, candidate, native, matched, truth in zip(
                ("output", "grad_value", "grad_locations", "grad_weights"),
                (actual, *grads),
                (reference, *ref_grads),
                (promoted, *promoted_grads),
                oracle,
            ):
                tensors[name] = {
                    "candidate_vs_native_hf": comparison(candidate, native, tolerance),
                    "candidate_vs_fp32_hf": comparison(candidate, matched, tolerance),
                    "candidate_vs_fp64_oracle": comparison(candidate, truth, tolerance),
                    "native_hf_vs_fp64_oracle": comparison(native, truth, tolerance),
                }
            report["cases"].append(
                {"dtype": str(dtype), "tolerance": tolerance, "tensors": tensors}
            )
            if os.environ.get("MSDA_OUTPUT_DIR"):
                (Path(os.environ["MSDA_OUTPUT_DIR"]) / "phase1-numerics.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
            for name, checks in tensors.items():
                for contract in (
                    "candidate_vs_native_hf",
                    "candidate_vs_fp32_hf",
                    "candidate_vs_fp64_oracle",
                ):
                    with self.subTest(dtype=dtype, tensor=name, comparison=contract):
                        self.assertTrue(checks[contract]["close"], checks[contract])

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

"""AMP optimizer parity and autograd lifecycle on CPU and available CUDA."""

import copy
import itertools
import unittest

import torch
from test_integration import inputs

from torch_ms_deform_attn import (
    _C,
    MSDeformAttnFunction,
    ms_deform_attn,
    ms_deform_attn_core_pytorch,
)


class TinyAttention(torch.nn.Module):
    def __init__(self, device, reference=False, legacy=False):
        super().__init__()
        value, shapes, starts, loc, weights = inputs(device=device)
        self.value = torch.nn.Parameter(value.detach())
        self.locations = torch.nn.Parameter(loc.detach())
        self.weights = torch.nn.Parameter(weights.detach())
        self.register_buffer("shapes", shapes)
        self.register_buffer("starts", starts)
        self.reference = reference
        self.legacy = legacy

    def forward(self):
        # Autocast-eligible projections produce low-precision operator inputs
        # from FP32 parameters. Keep coordinates FP32 as common AMP models do.
        v = self.value @ torch.eye(self.value.shape[-1], device=self.value.device)
        w = self.weights @ torch.eye(self.weights.shape[-1], device=self.weights.device)
        if self.reference:
            return ms_deform_attn_core_pytorch(v.float(), self.shapes, self.locations, w.float())
        api = MSDeformAttnFunction.apply if self.legacy else ms_deform_attn
        return api(v, self.shapes, self.starts, self.locations, w, 2)


class AMPTrainingCases:
    device = "cpu"

    def dtypes(self):
        return (torch.float16, torch.bfloat16)

    def test_scaled_training_and_accumulation(self):
        for dtype, compiled, legacy in itertools.product(
            self.dtypes(), (False, True), (False, True)
        ):
            with self.subTest(dtype=dtype, compiled=compiled, legacy=legacy):
                model = TinyAttention(self.device, legacy=legacy)
                reference = copy.deepcopy(model)
                reference.reference = True
                fn = (
                    torch.compile(
                        model,
                        backend="aot_eager" if self.device == "cpu" else "inductor",
                        fullgraph=True,
                    )
                    if compiled
                    else model
                )
                optimizers = [torch.optim.SGD(m.parameters(), lr=0.02) for m in (model, reference)]
                scalers = [torch.amp.GradScaler(self.device, init_scale=8) for _ in range(2)]
                for _ in range(2):
                    for optimizer in optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    losses = []
                    for target, optimizer, scaler in zip((fn, reference), optimizers, scalers):
                        with torch.autocast(self.device, dtype=dtype):
                            output = target()
                            self.assertEqual(output.dtype, torch.float32)
                            loss = output.square().mean()
                        losses.append(loss.detach())
                        # Accumulate on a retained graph, then release it.
                        scaler.scale(loss / 2).backward(retain_graph=True)
                        scaler.scale(loss / 2).backward()
                        scaler.unscale_(optimizer)
                    # Low-precision projection gradients can round differently after
                    # compilation; updates propagate that rounding into step two.
                    torch.testing.assert_close(*losses, atol=2e-3, rtol=2e-2)
                    for actual, expected in zip(model.parameters(), reference.parameters()):
                        self.assertEqual(actual.grad.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(actual.grad).all())
                        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-3, rtol=2e-2)
                    before = [p.detach().clone() for p in model.parameters()]
                    for optimizer, scaler in zip(optimizers, scalers):
                        scaler.step(optimizer)
                        scaler.update()
                    self.assertTrue(
                        any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
                    )
                    for actual, expected in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=2e-3)

    def test_amp_dtype_patterns_and_partial_gradients(self):
        for dtype in self.dtypes():
            for low_precision in itertools.product((False, True), repeat=3):
                args = list(inputs(device=self.device))
                tensors = []
                for index, low in zip((0, 3, 4), low_precision):
                    args[index] = (
                        args[index].detach().to(dtype if low else torch.float32).requires_grad_()
                    )
                    tensors.append(args[index])
                with self.subTest(dtype=dtype, low_precision=low_precision):
                    with torch.autocast(self.device, dtype=dtype):
                        actual = MSDeformAttnFunction.apply(*args, 2)
                    self.assertEqual(actual.dtype, torch.float32)
                    expected = ms_deform_attn_core_pytorch(
                        args[0].float(), args[1], args[3].float(), args[4].float()
                    )
                    torch.testing.assert_close(actual, expected)
                    for tensor, a, e in zip(
                        tensors,
                        torch.autograd.grad(actual.sum(), tensors),
                        torch.autograd.grad(expected.sum(), tensors),
                    ):
                        self.assertEqual(a.dtype, tensor.dtype)
                        torch.testing.assert_close(a, e, atol=0.002, rtol=0.02)
        for mask in itertools.product((False, True), repeat=3):
            if not any(mask):
                continue
            args = list(inputs(device=self.device))
            selected = []
            for index, enabled in zip((0, 3, 4), mask):
                args[index] = args[index].detach().requires_grad_(enabled)
                if enabled:
                    selected.append(args[index])
            actual = ms_deform_attn(*args)
            expected = ms_deform_attn_core_pytorch(args[0], args[1], args[3], args[4])
            for a, e in zip(
                torch.autograd.grad(actual.sum(), selected),
                torch.autograd.grad(expected.sum(), selected),
            ):
                torch.testing.assert_close(a, e, atol=2e-5, rtol=1e-4)

    def test_double_and_inference(self):
        args = inputs(device=self.device, dtype=torch.float64)
        with torch.autocast(self.device, dtype=torch.bfloat16):
            actual = ms_deform_attn(*args)
        self.assertEqual(actual.dtype, torch.float64)
        expected = ms_deform_attn_core_pytorch(args[0], args[1], args[3], args[4])
        torch.testing.assert_close(actual, expected)
        for context in (torch.no_grad(), torch.inference_mode()):
            with context:
                result = ms_deform_attn(*args)
                self.assertFalse(result.requires_grad)
                torch.testing.assert_close(result, expected)


class CPUAMPTrainingTest(AMPTrainingCases, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available() and _C.with_cuda, "Requires CUDA extension and GPU")
class CUDAAMPTrainingTest(AMPTrainingCases, unittest.TestCase):
    device = "cuda"

    def dtypes(self):
        # Report unsupported hardware explicitly instead of silently continuing.
        self.assertTrue(
            torch.cuda.is_bf16_supported(), "AMP validation requires a bf16-capable GPU"
        )
        return super().dtypes()

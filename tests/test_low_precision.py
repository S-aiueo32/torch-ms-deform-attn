"""Explicit half/bfloat16 inputs, without autocast, use float32 accumulation."""

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


def reference(args):
    value, shapes, _, locations, weights = args
    return ms_deform_attn_core_pytorch(
        value.float(), shapes, locations.float(), weights.float()
    ).to(value.dtype)


def low_precision_inputs(device, conversion, batch=2, queries=3):
    args = list(inputs(batch=batch, queries=queries, device=device))
    for index in (0, 3, 4):
        tensor = getattr(args[index].detach(), conversion)()
        # Retain the logical shape while exercising a noncontiguous layout.
        args[index] = tensor.transpose(-1, -2).contiguous().transpose(-1, -2).requires_grad_()
    return args


class LowPrecisionCases:
    device = "cpu"

    def test_forward_backward_and_dynamic_compile(self):
        for conversion, api, backend in itertools.product(
            ("half", "bfloat16"),
            (ms_deform_attn, MSDeformAttnFunction.apply),
            (None, "aot_eager", "inductor"),
        ):
            # CPU/CUDA, dtype, API, and backend combinations share Dynamo's
            # per-code cache. Isolate cases, but reuse it across dynamic shapes.
            torch._dynamo.reset()
            fn = (
                torch.compile(api, backend=backend, fullgraph=True, dynamic=True)
                if backend
                else api
            )
            for batch, queries in ((2, 3), (3, 5)):
                with self.subTest(
                    conversion=conversion, api=api.__name__, backend=backend, batch=batch
                ):
                    args = low_precision_inputs(self.device, conversion, batch, queries)
                    actual = fn(*args, 2)
                    expected = reference(args)
                    self.assertEqual(actual.dtype, args[0].dtype)
                    self.assertEqual(actual.shape, (batch, queries, 8))
                    torch.testing.assert_close(actual, expected)
                    grad = torch.randn_like(actual)
                    tensors = (args[0], args[3], args[4])
                    for tensor, a, e in zip(
                        tensors,
                        torch.autograd.grad(actual, tensors, grad),
                        torch.autograd.grad(expected, tensors, grad),
                    ):
                        self.assertEqual(a.dtype, tensor.dtype)
                        self.assertTrue(torch.isfinite(a).all())
                        torch.testing.assert_close(a, e, atol=2e-3, rtol=2e-2)

    def test_optimizer_steps(self):
        for conversion in ("half", "bfloat16"):
            with self.subTest(conversion=conversion):
                args = low_precision_inputs(self.device, conversion)
                expected_args = [t.detach().clone().requires_grad_(t.requires_grad) for t in args]
                parameters = [args[i] for i in (0, 3, 4)]
                expected_parameters = [expected_args[i] for i in (0, 3, 4)]
                optimizer = torch.optim.SGD(parameters, lr=0.02)
                expected_optimizer = torch.optim.SGD(expected_parameters, lr=0.02)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    expected_optimizer.zero_grad(set_to_none=True)
                    ms_deform_attn(*args).float().square().mean().backward()
                    reference(expected_args).float().square().mean().backward()
                    optimizer.step()
                    expected_optimizer.step()
                    for actual, expected in zip(parameters, expected_parameters):
                        torch.testing.assert_close(actual, expected)

    def test_inference_and_empty_queries(self):
        for conversion in ("half", "bfloat16"):
            args = low_precision_inputs(self.device, conversion)
            with self.subTest(conversion=conversion), torch.inference_mode():
                actual = ms_deform_attn(*args)
                self.assertFalse(actual.requires_grad)
                self.assertEqual(actual.dtype, args[0].dtype)
                torch.testing.assert_close(actual, reference(args))
            args = low_precision_inputs(self.device, conversion, queries=0)
            if self.device == "cuda":
                with self.assertRaisesRegex(RuntimeError, "empty"):
                    ms_deform_attn(*args)
            else:
                actual = ms_deform_attn(*args)
                self.assertEqual(actual.dtype, args[0].dtype)
                self.assertEqual(actual.shape, (2, 0, 8))
                for grad in torch.autograd.grad(actual.sum(), (args[0], args[3], args[4])):
                    self.assertEqual(grad.count_nonzero().item(), 0)


class CPULowPrecisionTest(LowPrecisionCases, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available() and _C.with_cuda, "Requires CUDA extension and GPU")
class CUDALowPrecisionTest(LowPrecisionCases, unittest.TestCase):
    device = "cuda"

"""Dispatcher, compiled autograd, and mixed-precision integration tests."""
import unittest

import torch

from torch_deform_attn import MSDeformAttnFunction, ms_deform_attn, ms_deform_attn_core_pytorch
from torch_deform_attn._ops import forward, backward


def inputs(batch=2, queries=3, dtype=torch.float32, device="cpu"):
    torch.manual_seed(17)
    shapes = torch.tensor([[2, 3], [1, 2]], device=device)
    starts = torch.tensor([0, 6], device=device)
    value = torch.randn(batch, 8, 2, 4, device=device, dtype=dtype, requires_grad=True)
    loc = torch.rand(batch, queries, 2, 2, 2, 2, device=device, dtype=dtype, requires_grad=True)
    weights = torch.rand(batch, queries, 2, 2, 2, device=device, dtype=dtype, requires_grad=True)
    return value, shapes, starts, loc, weights


class IntegrationTest(unittest.TestCase):
    def test_opcheck(self):
        args = (*inputs(), 2)
        for status in torch.library.opcheck(forward, args).values():
            self.assertEqual(status, "SUCCESS")
        value, shapes, starts, loc, weights, step = args
        grad = torch.randn(value.shape[0], loc.shape[1], 8)
        backward_args = tuple(t.detach() if isinstance(t, torch.Tensor) else t
                              for t in (value, shapes, starts, loc, weights, grad, step))
        for status in torch.library.opcheck(backward, backward_args).values():
            self.assertEqual(status, "SUCCESS")

    def test_compile_dynamic_forward_backward(self):
        for backend in ("aot_eager", "inductor"):
            compiled = torch.compile(ms_deform_attn, backend=backend, fullgraph=True, dynamic=True)
            for batch, queries in ((2, 3), (3, 5)):
                with self.subTest(backend=backend, batch=batch, queries=queries):
                    args = inputs(batch, queries)
                    actual = compiled(*args, 2)
                    expected = ms_deform_attn_core_pytorch(args[0], args[1], args[3], args[4])
                    torch.testing.assert_close(actual, expected)
                    grad = torch.randn_like(actual)
                    differentiable = (args[0], args[3], args[4])
                    for a, e in zip(torch.autograd.grad(actual, differentiable, grad),
                                    torch.autograd.grad(expected, differentiable, grad)):
                        torch.testing.assert_close(a, e)

    def test_amp_and_compiled_amp(self):
        for dtype in (torch.float16, torch.bfloat16):
            for compile in (False, True):
                with self.subTest(dtype=dtype, compile=compile):
                    value, shapes, starts, loc, weights = inputs(dtype=dtype)
                    # Mixed value/coordinate dtypes occur in real AMP models.
                    loc = loc.detach().float().requires_grad_()
                    def run(v, s, i, l, w):
                        with torch.autocast("cpu", dtype=dtype):
                            return ms_deform_attn(v, s, i, l, w)
                    fn = torch.compile(run, backend="aot_eager", fullgraph=True) if compile else run
                    actual = fn(value, shapes, starts, loc, weights)
                    expected = ms_deform_attn_core_pytorch(value.float(), shapes, loc, weights.float())
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected)
                    for a, e in zip(torch.autograd.grad(actual.sum(), (value, loc, weights)),
                                    torch.autograd.grad(expected.sum(), (value, loc, weights))):
                        torch.testing.assert_close(a, e)

    def test_double_preserved_under_autocast(self):
        args = inputs(dtype=torch.double)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = ms_deform_attn(*args)
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual, ms_deform_attn_core_pytorch(args[0], args[1], args[3], args[4]))

    def test_legacy_apply_compile(self):
        args = inputs()
        compiled = torch.compile(MSDeformAttnFunction.apply, backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(compiled(*args, 64), ms_deform_attn(*args))

    def test_higher_order_gradients_rejected(self):
        args = inputs()
        grad = torch.autograd.grad(ms_deform_attn(*args).sum(), args[0], create_graph=True)[0]
        with self.assertRaisesRegex(RuntimeError, "autograd formula"):
            torch.autograd.grad(grad.sum(), args[3])

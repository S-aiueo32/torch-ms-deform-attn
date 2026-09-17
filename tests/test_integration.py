"""Dispatcher, compiled autograd, and mixed-precision integration tests."""

import unittest

import torch

from torch_ms_deform_attn import MSDeformAttnFunction, ms_deform_attn, ms_deform_attn_core_pytorch
from torch_ms_deform_attn._ops import backward, forward


def inputs(batch=2, queries=3, dtype=torch.float32, device="cpu"):
    torch.manual_seed(17)
    shapes = torch.tensor([[2, 3], [1, 2]], device=device)
    starts = torch.tensor([0, 6], device=device)
    value = torch.randn(batch, 8, 2, 4, device=device, dtype=dtype, requires_grad=True)
    loc = torch.rand(batch, queries, 2, 2, 2, 2, device=device, dtype=dtype, requires_grad=True)
    weights = torch.rand(batch, queries, 2, 2, 2, device=device, dtype=dtype, requires_grad=True)
    return value, shapes, starts, loc, weights


class IntegrationTest(unittest.TestCase):
    def test_metadata_check_saved_per_call(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        seen = []

        class ObserveBackward(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func == backward:
                    seen.append(args[7] if len(args) > 7 else False)
                return func(*args, **(kwargs or {}))

        args = inputs()
        checked = ms_deform_attn(*args, check_cuda_metadata=True)
        unchecked = ms_deform_attn(*args)
        with ObserveBackward():
            checked.sum().backward()
            unchecked.sum().backward()
        self.assertEqual(seen, [True, False])

    def test_opcheck(self):
        args = (*inputs(), 2, True)
        for status in torch.library.opcheck(forward, args).values():
            self.assertEqual(status, "SUCCESS")
        value, shapes, starts, loc, weights, step, check = args
        grad = torch.randn(value.shape[0], loc.shape[1], 8)
        backward_args = tuple(
            t.detach() if isinstance(t, torch.Tensor) else t
            for t in (value, shapes, starts, loc, weights, grad, step, check)
        )
        for status in torch.library.opcheck(backward, backward_args).values():
            self.assertEqual(status, "SUCCESS")

    def test_compile_dynamic_forward_backward(self):
        for backend in ("aot_eager", "inductor"):
            compiled = torch.compile(ms_deform_attn, backend=backend, fullgraph=True, dynamic=True)
            for batch, queries, check in ((2, 3, False), (2, 3, True), (3, 5, True)):
                with self.subTest(backend=backend, batch=batch, queries=queries):
                    args = inputs(batch, queries)
                    actual = compiled(*args, 2, check_cuda_metadata=check)
                    expected = ms_deform_attn_core_pytorch(args[0], args[1], args[3], args[4])
                    torch.testing.assert_close(actual, expected)
                    grad = torch.randn_like(actual)
                    differentiable = (args[0], args[3], args[4])
                    for a, e in zip(
                        torch.autograd.grad(actual, differentiable, grad),
                        torch.autograd.grad(expected, differentiable, grad),
                    ):
                        torch.testing.assert_close(a, e)

    def test_compiled_autograd(self):
        from torch._dynamo import compiled_autograd

        # PyTorch 2.10 renamed this testing entry point. Keep exercising actual
        # graph capture across supported versions rather than skipping the check.
        enable = getattr(compiled_autograd, "enable", None) or getattr(
            compiled_autograd, "_enable", None
        )
        self.assertTrue(callable(enable), "Compiled autograd capture entry point is unavailable")
        graphs = []

        def compiler(graph):
            graphs.append(graph)
            return torch.compile(graph, backend="eager", fullgraph=True)

        for batch, queries, step in ((2, 3, 2), (3, 5, 3)):
            args = inputs(batch, queries)
            differentiable = (args[0], args[3], args[4])
            expected = ms_deform_attn_core_pytorch(args[0], args[1], args[3], args[4])
            expected_grads = torch.autograd.grad(expected.sum(), differentiable)
            with enable(compiler):
                ms_deform_attn(*args, step, check_cuda_metadata=True).sum().backward()
            for tensor, reference in zip(differentiable, expected_grads):
                torch.testing.assert_close(tensor.grad, reference)
        # C++ autograd nodes have different FX representations across PyTorch
        # versions. Require actual compilation and gradient parity; the regular
        # AOT/export checks cover the opaque operator contract separately.
        self.assertTrue(graphs, "Compiled autograd must invoke the compiler")

    def test_amp_and_compiled_amp(self):
        for dtype in (torch.float16, torch.bfloat16):
            for compile in (False, True):
                with self.subTest(dtype=dtype, compile=compile):
                    value, shapes, starts, loc, weights = inputs(dtype=dtype)
                    # Mixed value/coordinate dtypes occur in real AMP models.
                    loc = loc.detach().float().requires_grad_()

                    def run(v, s, i, locations, w):
                        with torch.autocast("cpu", dtype=dtype):
                            return ms_deform_attn(v, s, i, locations, w)

                    fn = torch.compile(run, backend="aot_eager", fullgraph=True) if compile else run
                    actual = fn(value, shapes, starts, loc, weights)
                    expected = ms_deform_attn_core_pytorch(
                        value.float(), shapes, loc, weights.float()
                    )
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected)
                    for a, e in zip(
                        torch.autograd.grad(actual.sum(), (value, loc, weights)),
                        torch.autograd.grad(expected.sum(), (value, loc, weights)),
                    ):
                        torch.testing.assert_close(a, e)

    def test_double_preserved_under_autocast(self):
        args = inputs(dtype=torch.double)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = ms_deform_attn(*args)
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(
            actual, ms_deform_attn_core_pytorch(args[0], args[1], args[3], args[4])
        )

    def test_legacy_apply_compile(self):
        args = inputs()
        compiled = torch.compile(MSDeformAttnFunction.apply, backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(compiled(*args, 64), ms_deform_attn(*args))

    def test_higher_order_gradients_rejected(self):
        args = inputs()
        grad = torch.autograd.grad(ms_deform_attn(*args).sum(), args[0], create_graph=True)[0]
        with self.assertRaisesRegex(RuntimeError, "autograd formula"):
            torch.autograd.grad(grad.sum(), args[3])

    def test_backward_rejects_gradient_through_grad_output(self):
        args = tuple(tensor.detach() for tensor in inputs())
        grad = torch.randn(2, 3, 8, requires_grad=True)
        grads = backward(*args, grad, 2)
        self.assertTrue(all(tensor.requires_grad for tensor in grads))
        with self.assertRaisesRegex(RuntimeError, "higher-order gradients are unsupported"):
            torch.autograd.grad(grads[0].sum(), grad)

    def test_export_keeps_opaque_operator(self):
        class Attention(torch.nn.Module):
            def forward(self, value, shapes, starts, locations, weights):
                return ms_deform_attn(
                    value, shapes, starts, locations, weights, check_cuda_metadata=True
                )

        args = inputs()
        exported = torch.export.export(Attention(), args)
        self.assertIn(forward, [node.target for node in exported.graph.nodes])
        node = next(node for node in exported.graph.nodes if node.target == forward)
        self.assertIs(node.args[-1], True)
        torch.testing.assert_close(exported.module()(*args), ms_deform_attn(*args))

    def test_saved_tensor_hooks_and_mutation(self):
        args = inputs()
        packed = []

        def pack(tensor):
            packed.append(tensor)
            return tensor.detach().clone()

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = ms_deform_attn(*args)
        self.assertEqual(len(packed), 5)
        differentiable = (args[0], args[3], args[4])
        expected = ms_deform_attn_core_pytorch(args[0], args[1], args[3], args[4])
        for actual, reference in zip(
            torch.autograd.grad(output.sum(), differentiable),
            torch.autograd.grad(expected.sum(), differentiable),
        ):
            torch.testing.assert_close(actual, reference)

        output = ms_deform_attn(*args)
        with torch.no_grad():
            args[3].add_(0.1)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            output.sum().backward()

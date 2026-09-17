"""CUDA dispatcher and fullgraph shape/value polymorphism."""

import itertools
import json
import unittest

import torch

from torch_ms_deform_attn import _C, ms_deform_attn, ms_deform_attn_core_pytorch
from torch_ms_deform_attn._ops import backward, forward


def inputs(spec, dtype=torch.float32):
    batch, queries, heads, channels, points, sizes = spec
    shapes = torch.tensor(sizes, device="cuda")
    starts = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
    options = dict(device="cuda", dtype=dtype, requires_grad=True)
    value = torch.randn(batch, sum(h * w for h, w in sizes), heads, channels, **options)
    loc = torch.rand(batch, queries, heads, len(sizes), points, 2, **options)
    weights = torch.rand(batch, queries, heads, len(sizes), points, **options)
    return value, shapes, starts, loc, weights


@unittest.skipUnless(torch.cuda.is_available() and _C.with_cuda, "Requires CUDA extension and GPU")
class CUDAGraphTest(unittest.TestCase):
    def test_metadata_modes_graph_capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for check, dtype in itertools.product(
            (False, True), (torch.float32, torch.float16, torch.bfloat16)
        ):
            with self.subTest(check=check, dtype=dtype), torch.cuda.stream(stream):
                args = inputs((3, 5, 2, 4, 3, ((2, 3), (2, 2))), dtype)
                differentiable = (args[0], args[3], args[4])

                def run():
                    output = ms_deform_attn(*args, 2, check_cuda_metadata=check)
                    gradients = torch.autograd.grad(output.sum(), differentiable)
                    return output, gradients

                for _ in range(3):
                    run()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual, actual_grads = run()
                graph.replay()
                expected = ms_deform_attn_core_pytorch(
                    args[0].float(), ((2, 3), (2, 2)), args[3].float(), args[4].float()
                ).to(dtype)
                expected_grads = torch.autograd.grad(expected.sum(), differentiable)
                tolerance = (
                    dict(atol=2e-4, rtol=1e-4)
                    if dtype == torch.float32
                    else dict(atol=0.02, rtol=0.02)
                )
                torch.testing.assert_close(actual, expected, **tolerance)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad, **tolerance)
        stream.synchronize()

    def test_opcheck(self):
        for dtype, noncontiguous, mask in itertools.product(
            (torch.float32, torch.float64),
            (False, True),
            ((False, False, False), (True, True, True), (True, False, False), (False, True, True)),
        ):
            with self.subTest(dtype=dtype, noncontiguous=noncontiguous, mask=mask):
                args = list(inputs((3, 5, 2, 4, 3, ((2, 3), (2, 2))), dtype))
                if noncontiguous:
                    args = [torch.stack((t, t), -1)[..., 0].detach() for t in args]
                    self.assertTrue(all(not t.is_contiguous() for t in args))
                for index, enabled in zip((0, 3, 4), mask):
                    args[index] = args[index].detach().requires_grad_(enabled)
                for status in torch.library.opcheck(forward, (*args, 2)).values():
                    self.assertEqual(status, "SUCCESS")
                grad = torch.randn(3, 5, 8, device="cuda", dtype=dtype)
                if noncontiguous:
                    grad = torch.stack((grad, grad), -1)[..., 0]
                # Backward itself deliberately has no higher-order autograd formula.
                backward_args = (*[t.detach() for t in args], grad, 2)
                for status in torch.library.opcheck(backward, backward_args).values():
                    self.assertEqual(status, "SUCCESS")

    def test_dynamic_shapes_and_metadata(self):
        # The first two vary every dimension without introducing unit dimensions.
        # The third keeps shapes fixed while changing metadata values/resolution.
        specs = (
            (3, 5, 2, 4, 3, ((2, 3), (2, 2))),
            (5, 7, 3, 8, 4, ((3, 4), (2, 3), (2, 2))),
            (5, 7, 3, 8, 4, ((4, 3), (3, 2), (2, 2))),
            (1, 1, 1, 1, 1, ((2, 3),)),
        )
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                if dtype == torch.bfloat16:
                    self.assertTrue(torch.cuda.is_bf16_supported(), "Requires bf16-capable GPU")
                torch._dynamo.reset()
                graphs = []

                def backend(gm, example_inputs):
                    graphs.append(str(gm.graph))
                    return torch._inductor.compile(gm, example_inputs)

                def run(v, s, i, loc, w):
                    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                        return ms_deform_attn(v, s, i, loc, w, 2)

                fn = torch.compile(run, backend=backend, fullgraph=True, dynamic=True)
                calls = []
                for spec in specs:
                    args = inputs(spec, dtype)
                    # Coordinates commonly stay float32 under AMP.
                    args = (*args[:3], args[3].detach().float().requires_grad_(), args[4])
                    actual = fn(*args)
                    expected = ms_deform_attn_core_pytorch(
                        args[0].float(), spec[-1], args[3], args[4].float()
                    )
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
                    tensors = (args[0], args[3], args[4])
                    grad = torch.randn_like(actual)
                    tolerance = (
                        dict(atol=2e-4, rtol=1e-4)
                        if dtype == torch.float32
                        else dict(atol=0.02, rtol=0.02)
                    )
                    for a, e in zip(
                        torch.autograd.grad(actual, tensors, grad),
                        torch.autograd.grad(expected, tensors, grad),
                    ):
                        torch.testing.assert_close(a, e, **tolerance)
                    calls.append(len(graphs))
                # Metadata is runtime tensor data, and may never specialize a graph.
                self.assertEqual(calls[1], calls[2], calls)
                # Repeat all seen shapes: no continuing cache churn is acceptable.
                count = len(graphs)
                for spec in specs:
                    args = inputs(spec, dtype)
                    args = (*args[:3], args[3].detach().float().requires_grad_(), args[4])
                    fn(*args).sum().backward()
                self.assertEqual(len(graphs), count)
                print(
                    json.dumps(
                        {
                            "dtype": str(dtype),
                            "graphs_after_calls": calls,
                            "unit_dimension_call": 3,
                            "repeat_graphs": len(graphs),
                        }
                    )
                )

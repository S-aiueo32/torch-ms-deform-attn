"""CPU extension regression tests: python -m unittest discover -s tests -v."""
import json
import os
import subprocess
import sys
import unittest

import torch
from torch.autograd import gradcheck

from torch_deform_attn import _C as MSDA
from torch_deform_attn import ms_deform_attn, MSDeformAttnFunction, ms_deform_attn_core_pytorch


class CPUAttentionTest(unittest.TestCase):
    def inputs(self, dtype=torch.double, noncontiguous=False):
        torch.manual_seed(7)
        shapes = torch.tensor([[3, 4], [2, 2]], dtype=torch.long)
        starts = torch.tensor([0, 12], dtype=torch.long)
        value = torch.randn(3, 16, 2, 3, dtype=dtype)
        locations = torch.rand(3, 2, 2, 2, 3, 2, dtype=dtype) * 1.8 - 0.4
        # Include edges and entirely out-of-bounds samples.
        locations[:, 0, :, :, 0, :] = 0
        locations[:, 0, :, :, 1, :] = 1
        locations[:, 0, :, :, 2, :] = -2
        weights = torch.randn(3, 2, 2, 2, 3, dtype=dtype)
        if noncontiguous:
            shapes = shapes.t().contiguous().t()
            starts = torch.tensor([0, -1, 12, -1], dtype=torch.long)[::2]
            value = value.transpose(1, 2).contiguous().transpose(1, 2)
            locations = locations.transpose(1, 2).contiguous().transpose(1, 2)
            weights = weights.transpose(1, 2).contiguous().transpose(1, 2)
        return value.requires_grad_(), shapes, starts, locations.requires_grad_(), weights.requires_grad_()

    def test_cpu_parallel_workers_honor_thread_count(self):
        self.assertIn(MSDA.cpu_parallel_backend, ("openmp", "native", "serial"))
        if MSDA.cpu_parallel_backend == "serial":
            self.skipTest("extension was built without a CPU parallel backend")

        # Fresh processes set the thread count before the native thread pool is
        # initialized. The probe lives in the CPU kernel's C++ translation unit;
        # PyTorch's own parallel configuration alone cannot detect a serial build.
        probe = """
import json
import sys
import torch
from torch_deform_attn import _C
torch.set_num_threads(int(sys.argv[1]))
print(json.dumps({
    "threads": torch.get_num_threads(),
    "workers": [_C._cpu_parallel_worker_count(items) for items in (0, 1, 8)],
}))
"""
        for threads in (1, 4):
            with self.subTest(threads=threads):
                environment = os.environ.copy()
                environment.update(OMP_DYNAMIC="FALSE", OMP_THREAD_LIMIT=str(threads))
                result = subprocess.run(
                    [sys.executable, "-c", probe, str(threads)],
                    capture_output=True, text=True, env=environment, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["threads"], threads)
                empty, single, multiple = report["workers"]
                self.assertEqual(empty, 0)
                self.assertEqual(single, 1)
                if threads == 1:
                    self.assertEqual(multiple, 1)
                else:
                    self.assertGreater(multiple, 1)
                    self.assertLessEqual(multiple, threads)

    def test_reference_forward_and_backward(self):
        for dtype in (torch.float32, torch.float64):
            for noncontiguous in (False, True):
                with self.subTest(dtype=dtype, noncontiguous=noncontiguous):
                    value, shapes, starts, locations, weights = self.inputs(dtype, noncontiguous)
                    # Batch size 3 also checks that CPU accepts a non-dividing step.
                    actual = MSDeformAttnFunction.apply(value, shapes, starts, locations, weights, 2)
                    expected = ms_deform_attn_core_pytorch(value, shapes, locations, weights)
                    torch.testing.assert_close(actual, expected)
                    grad = torch.randn(3, 6, 2, dtype=dtype).transpose(1, 2)
                    inputs = (value, locations, weights)
                    actual_grads = torch.autograd.grad(actual, inputs, grad)
                    expected_grads = torch.autograd.grad(expected, inputs, grad)
                    for a, e in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(a, e)

    def test_padding_support_and_channel_sizes(self):
        # Exercise all neighbor masks near the support boundary without sampling
        # exactly at a nondifferentiable pixel/support boundary.
        shapes = torch.tensor([[2, 3]], dtype=torch.long)
        starts = torch.tensor([0], dtype=torch.long)
        pixels = torch.tensor([
            [-0.75, 0.3], [2.75, 0.3], [0.3, -0.75], [0.3, 1.75],
            [-1.25, 0.3], [3.25, 0.3], [0.3, -1.25], [0.3, 2.25],
            [-0.75, -0.75], [2.75, -0.75], [-0.75, 1.75], [2.75, 1.75],
            [0.3, 0.7],
        ], dtype=torch.double)
        for dtype in (torch.float32, torch.float64):
            for channels in (1, 7, 32):
                with self.subTest(dtype=dtype, channels=channels):
                    torch.manual_seed(19)
                    value = torch.randn(1, 6, 2, channels, dtype=dtype, requires_grad=True)
                    locations = ((pixels + 0.5) / torch.tensor([3, 2])).to(dtype)
                    locations = locations.view(1, -1, 1, 1, 1, 2).repeat(1, 1, 2, 1, 1, 1)
                    locations.requires_grad_()
                    weights = torch.randn(1, len(pixels), 2, 1, 1, dtype=dtype, requires_grad=True)
                    actual = ms_deform_attn(value, shapes, starts, locations, weights)
                    expected = ms_deform_attn_core_pytorch(value, shapes, locations, weights)
                    torch.testing.assert_close(actual, expected)
                    grad = torch.randn_like(actual)
                    inputs = (value, locations, weights)
                    for a, e in zip(torch.autograd.grad(actual, inputs, grad),
                                    torch.autograd.grad(expected, inputs, grad)):
                        torch.testing.assert_close(a, e)

    def test_colliding_samples_and_overlapping_levels(self):
        # Both levels and every query/point contribute to the same value cells.
        # Batch size one requires parallel backward to divide work by head.
        shapes = torch.tensor([[3, 3], [2, 3]], dtype=torch.long)
        starts = torch.tensor([0, 3], dtype=torch.long)
        previous_threads = torch.get_num_threads()
        try:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(dtype=dtype):
                    torch.manual_seed(23)
                    value = torch.randn(1, 9, 8, 7, dtype=dtype, requires_grad=True)
                    locations = torch.tensor([0.43, 0.61], dtype=dtype)
                    locations = locations.repeat(1, 32, 8, 2, 4, 1).requires_grad_()
                    weights = torch.randn(1, 32, 8, 2, 4, dtype=dtype, requires_grad=True)
                    grad = torch.randn(1, 32, 56, dtype=dtype)
                    inputs = (value, locations, weights)
                    torch.set_num_threads(1)
                    reordered = torch.cat((value, value[:, 3:9]), dim=1)
                    expected = ms_deform_attn_core_pytorch(reordered, shapes, locations, weights)
                    expected_grads = torch.autograd.grad(expected, inputs, grad)
                    baseline = None
                    for threads in (1, 4, 4, 4):
                        torch.set_num_threads(threads)
                        actual = ms_deform_attn(value, shapes, starts, locations, weights)
                        actual_grads = torch.autograd.grad(actual, inputs, grad)
                        torch.testing.assert_close(actual, expected)
                        for a, e in zip(actual_grads, expected_grads):
                            torch.testing.assert_close(a, e)
                        result = (actual, *actual_grads)
                        if baseline is None:
                            baseline = result
                        else:
                            for a, b in zip(result, baseline):
                                torch.testing.assert_close(a, b, rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous_threads)

    def test_public_api(self):
        value, shapes, starts, locations, weights = self.inputs()
        actual = ms_deform_attn(value, shapes, starts, locations, weights)
        expected = ms_deform_attn_core_pytorch(value, shapes, locations, weights)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        for tensor in (value, locations, weights):
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_empty_batch(self):
        value, shapes, starts, locations, weights = self.inputs()
        output = ms_deform_attn(value[:0], shapes, starts, locations[:0], weights[:0])
        self.assertEqual(output.shape, (0, 2, 6))
        output.sum().backward()
        self.assertEqual(value.grad.count_nonzero().item(), 0)

    def test_empty_heads_channels_levels_and_points(self):
        for dimension in ("heads", "channels", "levels", "points"):
            with self.subTest(dimension=dimension):
                value, shapes, starts, locations, weights = self.inputs()
                if dimension == "heads":
                    value, locations, weights = value[:, :, :0], locations[:, :, :0], weights[:, :, :0]
                elif dimension == "channels":
                    value = value[..., :0]
                elif dimension == "levels":
                    shapes, starts = shapes[:0], starts[:0]
                    locations, weights = locations[:, :, :, :0], weights[:, :, :, :0]
                else:
                    locations, weights = locations[..., :0, :], weights[..., :0]
                value, locations, weights = (
                    tensor.detach().requires_grad_() for tensor in (value, locations, weights))
                output = ms_deform_attn(value, shapes, starts, locations, weights)
                self.assertEqual(output.shape, (3, 2, value.shape[2] * value.shape[3]))
                self.assertEqual(output.count_nonzero().item(), 0)
                inputs = (value, locations, weights)
                for tensor, grad in zip(inputs, torch.autograd.grad(output.sum(), inputs)):
                    self.assertEqual(grad.shape, tensor.shape)
                    self.assertEqual(grad.count_nonzero().item(), 0)

    def test_reject_unsupported_dtype(self):
        value, shapes, starts, locations, weights = self.inputs(torch.float16)
        with self.assertRaisesRegex(RuntimeError, "float32 and float64"):
            ms_deform_attn(value, shapes, starts, locations, weights)

    def test_reject_invalid_gradient(self):
        value, shapes, starts, locations, weights = self.inputs()
        with self.assertRaisesRegex(RuntimeError, "grad_output shape"):
            MSDA.ms_deform_attn_backward(value, shapes, starts, locations, weights,
                                        torch.zeros(3, 2, 5, dtype=value.dtype), 64)

    def test_gradcheck(self):
        torch.manual_seed(11)
        value = torch.randn(1, 5, 1, 2, dtype=torch.double, requires_grad=True)
        shapes = torch.tensor([[2, 2], [1, 1]])
        starts = torch.tensor([0, 4])
        locations = torch.rand(1, 2, 1, 2, 2, 2, dtype=torch.double, requires_grad=True)
        weights = torch.rand(1, 2, 1, 2, 2, dtype=torch.double, requires_grad=True)
        self.assertTrue(gradcheck(MSDeformAttnFunction.apply,
                                 (value, shapes, starts, locations, weights, 64)))

    def test_level_start_index(self):
        value, shapes, _, locations, weights = self.inputs()
        starts = torch.tensor([4, 0])
        reordered = torch.cat((value[:, 4:16], value[:, :4]), dim=1)
        actual = MSDeformAttnFunction.apply(value, shapes, starts, locations, weights, 64)
        expected = ms_deform_attn_core_pytorch(reordered, shapes, locations, weights)
        torch.testing.assert_close(actual, expected)
        for a, e in zip(torch.autograd.grad(actual.sum(), (value, locations, weights)),
                        torch.autograd.grad(expected.sum(), (value, locations, weights))):
            torch.testing.assert_close(a, e)

    def test_invalid_inputs(self):
        value, shapes, starts, locations, weights = self.inputs()
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            MSDA.ms_deform_attn_forward(value, shapes, starts + 100, locations, weights, 64)
        with self.assertRaisesRegex(RuntimeError, "dtypes must match"):
            MSDA.ms_deform_attn_forward(value, shapes, starts, locations.float(), weights, 64)
        with self.assertRaisesRegex(RuntimeError, "positive"):
            MSDA.ms_deform_attn_forward(value, shapes, starts, locations, weights, 0)

    def test_empty_queries(self):
        value, shapes, starts, locations, weights = self.inputs()
        output = MSDeformAttnFunction.apply(value, shapes, starts, locations[:, :0], weights[:, :0], 64)
        self.assertEqual(output.shape, (3, 0, 6))
        output.sum().backward()
        self.assertEqual(value.grad.count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()

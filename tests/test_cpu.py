"""CPU extension regression tests: python -m unittest discover -s tests -v."""
import unittest

import torch
from torch.autograd import gradcheck

from ms_deform_attn import _C as MSDA
from ms_deform_attn import ms_deform_attn, MSDeformAttnFunction, ms_deform_attn_core_pytorch


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
            value = value.transpose(1, 2).contiguous().transpose(1, 2)
            locations = locations.transpose(1, 2).contiguous().transpose(1, 2)
            weights = weights.transpose(1, 2).contiguous().transpose(1, 2)
        return value.requires_grad_(), shapes, starts, locations.requires_grad_(), weights.requires_grad_()

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

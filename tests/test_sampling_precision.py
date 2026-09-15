"""Near-knot gradients use a double oracle to avoid reference coordinate rounding."""

import unittest

import torch

from torch_ms_deform_attn import _C, ms_deform_attn, ms_deform_attn_core_pytorch


class SamplingPrecisionCases:
    device = "cpu"

    def test_near_knot_matches_double_oracle(self):
        shapes = torch.tensor([[64, 2]], device=self.device)
        starts = torch.tensor([0], device=self.device)
        value = torch.zeros(1, 64, 2, 1, 1, device=self.device)
        value[:, 8] = 1  # A peak: positive y slope before row 8, negative after.
        value = value.flatten(1, 2).requires_grad_()
        # This float32 coordinate maps to pixel 7.999999046325684. The float32
        # grid_sample normalization (2*y-1) rounds it onto pixel 8 instead.
        loc = (
            torch.tensor([0.5, 0.1328124850988388], device=self.device)
            .reshape(1, 1, 1, 1, 1, 2)
            .requires_grad_()
        )
        weights = torch.ones(1, 1, 1, 1, 1, device=self.device, requires_grad=True)
        tensors = (value, loc, weights)
        actual = ms_deform_attn(value, shapes, starts, loc, weights)
        oracle_inputs = [t.detach().double().requires_grad_() for t in tensors]
        expected = ms_deform_attn_core_pytorch(oracle_inputs[0], ((64, 2),), *oracle_inputs[1:])
        torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-6)
        grads = torch.autograd.grad(actual.sum(), tensors)
        oracle_grads = torch.autograd.grad(expected.sum(), oracle_inputs)
        for a, e in zip(grads, oracle_grads):
            torch.testing.assert_close(a.double(), e, atol=1e-6, rtol=1e-6)
        self.assertEqual(grads[1][..., 1].item(), 64.0)


class CPUSamplingPrecisionTest(SamplingPrecisionCases, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available() and _C.with_cuda, "Requires CUDA extension and GPU")
class CUDASamplingPrecisionTest(SamplingPrecisionCases, unittest.TestCase):
    device = "cuda"

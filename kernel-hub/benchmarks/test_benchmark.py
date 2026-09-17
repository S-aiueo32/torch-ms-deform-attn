"""CPU checks of benchmark qualification, independent of CUDA availability."""

import unittest
from unittest.mock import patch

import torch
from benchmark import CASES, inputs, matched_precision, validate

from torch_ms_deform_attn import ms_deform_attn_core_pytorch


class QualificationTest(unittest.TestCase):
    def test_precision_and_gradient_validation(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with (
                self.subTest(dtype=dtype),
                patch.dict(CASES, {"tiny": (1, 3, 2, 4, ((4, 4), (2, 2)))}),
            ):
                torch.manual_seed(29)
                data, grad, sizes, scale = inputs("tiny", dtype, device="cpu")

                def reference(value, shapes, starts, loc, weights):
                    return ms_deform_attn_core_pytorch(value, sizes, loc, weights)

                oracle_inputs = tuple(
                    tensor.detach().double().requires_grad_()
                    for tensor in (data[0], data[3], data[4])
                )
                out = reference(oracle_inputs[0], data[1], data[2], *oracle_inputs[1:])
                truth = (out, *torch.autograd.grad(out, oracle_inputs, grad.double()))
                self.assertTrue(
                    validate(matched_precision(reference), data, grad, truth, scale)["passed"]
                )

                # A forward-correct implementation with a broken coordinate
                # gradient must not qualify for timing.
                def broken(value, shapes, starts, loc, weights):
                    frozen = loc.detach() + loc * 0
                    return matched_precision(reference)(value, shapes, starts, frozen, weights)

                result = validate(broken, data, grad, truth, scale)
                self.assertFalse(result["passed"])
                self.assertTrue(result["errors"]["output"]["close"])
                self.assertFalse(result["errors"]["grad_locations"]["close"])


if __name__ == "__main__":
    unittest.main()

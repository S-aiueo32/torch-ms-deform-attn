"""Native structural rejection table shared by public and legacy APIs."""

import unittest

import torch
from test_integration import inputs

from torch_ms_deform_attn import _C, MSDeformAttnFunction, ms_deform_attn


class InputContractCases:
    device = "cpu"

    def test_forward_contract(self):
        args = inputs(device=self.device)
        cases = []
        for index in range(5):
            bad = list(args)
            bad[index] = bad[index].unsqueeze(0)
            cases.append((f"rank-{index}", bad, 2, "shape|nonempty|must match"))
        for index, axis in (
            (1, 1),
            (2, 0),
            (3, 0),
            (3, 2),
            (3, 3),
            (3, 5),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
        ):
            bad = list(args)
            bad[index] = bad[index].narrow(axis, 0, 1)
            cases.append((f"shape-{index}-{axis}", bad, 2, "shape|match"))
        for index in (1, 2):
            bad = list(args)
            bad[index] = bad[index].int()
            cases.append((f"metadata-{index}", bad, 2, "int64"))
        for index in (3, 4):
            bad = list(args)
            bad[index] = bad[index].double()
            cases.append((f"mixed-dtype-{index}", bad, 2, "dtypes must match"))
        bad = [t.int() if i in (0, 3, 4) else t for i, t in enumerate(args)]
        cases.append(("unsupported-int32", bad, 2, "float32 and float64"))
        for dtype in (torch.float16, torch.bfloat16):
            for index in (0, 3, 4):
                bad = list(args)
                bad[index] = bad[index].to(dtype)
                cases.append(
                    (
                        f"mixed-low-dtype-{dtype}-{index}",
                        bad,
                        2,
                        "dtypes must match|float32 and float64",
                    )
                )
        cases.extend((f"step-{step}", args, step, "positive") for step in (0, -1))
        if torch.cuda.is_available():
            for index in range(5):
                bad = list(args)
                bad[index] = bad[index].to("cpu" if self.device == "cuda" else "cuda")
                cases.append(
                    (f"device-{index}", bad, 2, "CPU tensors|same CUDA device|built without CUDA")
                )
        for api in (ms_deform_attn, MSDeformAttnFunction.apply):
            for name, bad, step, message in cases:
                with self.subTest(api=api.__name__, case=name):
                    with self.assertRaisesRegex(RuntimeError, message):
                        api(*bad, step)

    def test_backward_contract(self):
        args = inputs(device=self.device)
        grads = [
            torch.empty(shape, device=self.device)
            for shape in ((2, 3), (1, 3, 8), (2, 1, 8), (2, 3, 7), (2, 3, 8, 1))
        ]
        grads.append(torch.empty(2, 3, 8, dtype=torch.float64, device=self.device))
        if torch.cuda.is_available():
            grads.append(torch.empty(2, 3, 8, device="cpu" if self.device == "cuda" else "cuda"))
        for grad in grads:
            with self.subTest(shape=grad.shape, dtype=grad.dtype, device=grad.device):
                with self.assertRaisesRegex(RuntimeError, "grad_output"):
                    _C.ms_deform_attn_backward(*args, grad, 2)

    def test_empty_dimension_policy(self):
        args = inputs(device=self.device)
        axes = {
            "batch": ((0, 0), (3, 0), (4, 0)),
            "spatial": ((0, 1), (1, 0), (2, 0), (3, 3), (4, 3)),
            "heads": ((0, 2), (3, 2), (4, 2)),
            "channels": ((0, 3),),
            "queries": ((3, 1), (4, 1)),
            "levels": ((1, 0), (2, 0), (3, 3), (4, 3)),
            "points": ((3, 4), (4, 4)),
        }
        for name, dimensions in axes.items():
            bad = list(args)
            for index, axis in dimensions:
                bad[index] = bad[index].narrow(axis, 0, 0)
            for api in (ms_deform_attn, MSDeformAttnFunction.apply):
                with self.subTest(dimension=name, api=api.__name__):
                    if self.device == "cuda":
                        with self.assertRaisesRegex(RuntimeError, "nonempty|empty"):
                            api(*bad, 2)
                    else:
                        result = api(*bad, 2)
                        self.assertEqual(
                            result.shape,
                            (bad[0].shape[0], bad[3].shape[1], bad[0].shape[2] * bad[0].shape[3]),
                        )
                        self.assertEqual(result.count_nonzero().item(), 0)
                        for grad in torch.autograd.grad(result.sum(), (bad[0], bad[3], bad[4])):
                            self.assertEqual(grad.count_nonzero().item(), 0)


class CPUInputContractTest(InputContractCases, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available() and _C.with_cuda, "Requires CUDA extension and GPU")
class CUDAInputContractTest(InputContractCases, unittest.TestCase):
    device = "cuda"

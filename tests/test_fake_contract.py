"""Metadata validation must work without reading tensor storage."""

import unittest

import torch
from test_integration import inputs
from torch._subclasses.fake_tensor import FakeTensorMode

from torch_ms_deform_attn._ops import backward, forward


class FakeContractTest(unittest.TestCase):
    def test_invalid_forward_metadata(self):
        for fake in (False, True):
            args = inputs()
            cases = []
            for index in range(5):
                bad = list(args)
                bad[index] = bad[index].unsqueeze(0)
                cases.append((f"rank-{index}", bad, 2))
            for index, dtype in (
                (0, torch.float16),
                (0, torch.bfloat16),
                (0, torch.int32),
                (1, torch.int32),
                (2, torch.int32),
                (3, torch.float64),
                (4, torch.float64),
            ):
                bad = list(args)
                bad[index] = bad[index].to(dtype)
                cases.append((f"dtype-{index}-{dtype}", bad, 2))
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
                cases.append((f"shape-{index}-{axis}", bad, 2))
            cases.extend((f"step-{step}", args, step) for step in (0, -1))
            for name, bad, step in cases:
                with self.subTest(fake=fake, case=name):
                    if fake:
                        with FakeTensorMode() as mode:
                            converted = [mode.from_tensor(t) for t in bad]
                            with self.assertRaises(RuntimeError):
                                forward(*converted, step)
                    else:
                        with self.assertRaises(RuntimeError):
                            forward(*bad, step)

    def test_backward_metadata(self):
        args = inputs()
        for fake in (False, True):
            for grad in (
                torch.empty(2, 3),
                torch.empty(2, 3, 7),
                torch.empty(1, 3, 8),
                torch.empty(2, 1, 8),
                torch.empty(2, 3, 8, dtype=torch.float64),
            ):
                with self.subTest(fake=fake, shape=grad.shape, dtype=grad.dtype):
                    if fake:
                        with FakeTensorMode() as mode:
                            converted = [mode.from_tensor(t) for t in (*args, grad)]
                            with self.assertRaises(RuntimeError):
                                backward(*converted, 2)
                    else:
                        with self.assertRaises(RuntimeError):
                            backward(*args, grad, 2)

    def test_fake_devices_and_empty_dimensions(self):
        # Fake CUDA runs even on a CPU-only host; no native CUDA claim is made.
        for device in ("cpu", "cuda"):
            with FakeTensorMode():
                args = [torch.empty_like(t, device=device) for t in inputs()]
                for index in range(1, 5):
                    bad = args.copy()
                    bad[index] = torch.empty_like(
                        bad[index], device="cuda" if device == "cpu" else "cpu"
                    )
                    with self.assertRaises(RuntimeError):
                        forward(*bad, 2)
                with self.assertRaises(RuntimeError):
                    backward(*args, torch.empty(2, 3, 8, device="meta"), 2)
                for dimension in (
                    "batch",
                    "spatial",
                    "heads",
                    "channels",
                    "queries",
                    "levels",
                    "points",
                ):
                    sizes = [list(t.shape) for t in args]
                    axes = {
                        "batch": ((0, 0), (3, 0), (4, 0)),
                        "spatial": ((0, 1), (1, 0), (2, 0), (3, 3), (4, 3)),
                        "heads": ((0, 2), (3, 2), (4, 2)),
                        "channels": ((0, 3),),
                        "queries": ((3, 1), (4, 1)),
                        "levels": ((1, 0), (2, 0), (3, 3), (4, 3)),
                        "points": ((3, 4), (4, 4)),
                    }
                    for index, axis in axes[dimension]:
                        sizes[index][axis] = 0
                    v, s, i, loc, w = [
                        torch.empty(shape, dtype=t.dtype, device=device)
                        for shape, t in zip(sizes, args)
                    ]
                    with self.subTest(device=device, dimension=dimension):
                        if device == "cuda":
                            with self.assertRaisesRegex(RuntimeError, "nonempty"):
                                forward(v, s, i, loc, w, 2)
                        else:
                            result = forward(v, s, i, loc, w, 2)
                            self.assertEqual(
                                result.shape, (v.shape[0], loc.shape[1], v.shape[2] * v.shape[3])
                            )

"""Real Metal execution, CPU-double oracle, and public API contracts."""

import unittest
import warnings

import torch
from test_integration import inputs
from torch._subclasses.fake_tensor import FakeTensorMode

from torch_ms_deform_attn import _C, MSDeformAttnFunction, ms_deform_attn
from torch_ms_deform_attn._ops import forward


@unittest.skipUnless(torch.backends.mps.is_available() and _C.with_mps, "Requires Metal GPU/build")
class MPSTest(unittest.TestCase):
    def compare(self, args, *, low=False):
        oracle = [t.detach().cpu() for t in args]
        for i in (0, 3, 4):
            oracle[i] = oracle[i].double().requires_grad_()
        actual = ms_deform_attn(*args)
        expected = ms_deform_attn(*oracle)
        grad = torch.randn_like(expected).to(actual)
        actual_grads = torch.autograd.grad(actual, (args[0], args[3], args[4]), grad)
        expected_grads = torch.autograd.grad(
            expected, (oracle[0], oracle[3], oracle[4]), grad.cpu().double()
        )
        atol, rtol = (2e-3, 2e-2) if low else (1e-4, 1e-3)
        torch.testing.assert_close(actual.cpu().double(), expected, atol=atol, rtol=rtol)
        for a, e, original in zip(actual_grads, expected_grads, (args[0], args[3], args[4])):
            self.assertEqual(a.dtype, original.dtype)
            torch.testing.assert_close(a.cpu().double(), e, atol=atol, rtol=rtol)

    def test_forward_backward_and_legacy(self):
        args = inputs(device="mps")
        self.compare(args)
        torch.testing.assert_close(ms_deform_attn(*args), MSDeformAttnFunction.apply(*args, 1))

    def test_channels_boundaries_offsets_and_layouts(self):
        for channels in (1, 7, 32, 71):
            for starts in ([2, 12], [12, 2], [3, 3], [0, 3]):
                with self.subTest(channels=channels, starts=starts):
                    torch.manual_seed(23)
                    shapes = torch.tensor([[2, 3], [2, 3]], device="mps").t().contiguous().t()
                    offsets = torch.tensor([starts[0], -1, starts[1], -1], device="mps")[::2]
                    value = torch.randn(2, 20, 4, channels, device="mps")
                    value = value.transpose(1, 2).contiguous().transpose(1, 2).requires_grad_()
                    points = [[-0.25, 0.3], [0, 0], [1, 1], [1.25, 0.7], [0.43, 0.61]]
                    loc = torch.tensor(points, device="mps").view(1, 5, 1, 1, 1, 2)
                    loc = loc.repeat(2, 1, 4, 2, 4, 1).requires_grad_()
                    weights = torch.randn(2, 5, 4, 2, 4, device="mps", requires_grad=True)
                    self.compare((value, shapes, offsets, loc, weights))

    def test_storage_offsets_and_noncontiguous_grad(self):
        args = list(inputs(device="mps"))
        for i, t in enumerate(args):
            backing = torch.empty(t.numel() + 8, device="mps", dtype=t.dtype)
            view = backing[4:-4].view(t.shape)
            view.copy_(t.detach())
            args[i] = view.requires_grad_(i in (0, 3, 4))
            self.assertGreater(args[i].storage_offset(), 0)
        self.compare(args)
        output = ms_deform_attn(*args)
        grad = torch.randn(2, 8, 3, device="mps").transpose(1, 2)
        a = torch.autograd.grad(output, args[0], grad)[0]
        b = torch.autograd.grad(ms_deform_attn(*args), args[0], grad.contiguous())[0]
        torch.testing.assert_close(a, b)

    def test_low_precision(self):
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.backends.mps.is_macos_or_newer(14, 0):
                continue
            with self.subTest(dtype=dtype):
                args = inputs(device="mps", dtype=dtype)
                self.assertEqual(ms_deform_attn(*args).dtype, dtype)
                self.compare(args, low=True)

    @unittest.skipUnless(
        torch.amp.autocast_mode.is_autocast_available("mps"), "PyTorch has no MPS autocast"
    )
    def test_autocast(self):
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.backends.mps.is_macos_or_newer(14, 0):
                continue
            args = list(inputs(device="mps", dtype=dtype))
            args[3] = args[3].detach().float().requires_grad_()
            # Older PyTorch releases disable unsupported MPS autocast dtypes.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with torch.autocast("mps", dtype=dtype):
                    if not torch.is_autocast_enabled("mps"):
                        continue
                    self.assertEqual(ms_deform_attn(*args).dtype, torch.float32)
                    self.compare(args, low=True)

    def test_near_knot(self):
        value = torch.zeros(1, 128, 1, 1, device="mps")
        value[:, 16:18] = 1
        value.requires_grad_()
        loc = torch.tensor([0.5, 0.1328124850988388], device="mps")
        loc = loc.view(1, 1, 1, 1, 1, 2).requires_grad_()
        weights = torch.ones(1, 1, 1, 1, 1, device="mps", requires_grad=True)
        args = (
            value,
            torch.tensor([[64, 2]], device="mps"),
            torch.tensor([0], device="mps"),
            loc,
            weights,
        )
        self.compare(args)
        self.assertEqual(torch.autograd.grad(ms_deform_attn(*args).sum(), loc)[0][..., 1], 64)

    def test_rejections(self):
        args = list(inputs(device="mps"))
        cases = []
        for i in range(5):
            bad = args.copy()
            bad[i] = bad[i].cpu()
            cases.append(bad)
            bad = args.copy()
            bad[i] = bad[i].unsqueeze(0)
            cases.append(bad)
        for shape, start in (
            ([[0, 3], [1, 2]], [0, 6]),
            ([[2, 3], [1, 2]], [-1, 6]),
            ([[2, 3], [1, 2]], [0, 7]),
            ([[2**62, 4], [1, 2]], [0, 6]),
        ):
            bad = args.copy()
            bad[1] = torch.tensor(shape, device="mps")
            bad[2] = torch.tensor(start, device="mps")
            cases.append(bad)
        for i in (1, 2, 3, 4):
            bad = args.copy()
            bad[i] = bad[i].int() if i in (1, 2) else bad[i].half()
            cases.append(bad)
        for bad in cases:
            with self.assertRaises(RuntimeError):
                ms_deform_attn(*bad)
        for step in (0, -1):
            with self.assertRaisesRegex(RuntimeError, "positive"):
                ms_deform_attn(*args, step)
        axes = [
            ((0, 0), (3, 0), (4, 0)),
            ((0, 2), (3, 2), (4, 2)),
            ((0, 3),),
            ((3, 1), (4, 1)),
            ((1, 0), (2, 0), (3, 3), (4, 3)),
            ((3, 4), (4, 4)),
        ]
        for dimensions in axes:
            bad = args.copy()
            for i, axis in dimensions:
                bad[i] = bad[i].narrow(axis, 0, 0)
            with self.assertRaisesRegex(RuntimeError, "nonempty"):
                ms_deform_attn(*bad)

    def test_determinism_and_higher_order(self):
        args = inputs(device="mps")
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            with self.assertRaisesRegex(RuntimeError, "deterministic"):
                ms_deform_attn(*args).sum().backward()
            torch.use_deterministic_algorithms(True, warn_only=True)
            with self.assertWarnsRegex(UserWarning, "deterministic"):
                ms_deform_attn(*args).sum().backward()
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        grad = torch.autograd.grad(ms_deform_attn(*args).sum(), args[0], create_graph=True)[0]
        with self.assertRaisesRegex(RuntimeError, "autograd formula"):
            torch.autograd.grad(grad.sum(), args[3])

    def test_fake_contract(self):
        args = inputs(device="mps")
        actual = forward(*args, 64)
        # opcheck's mutation checks reduce high-rank MPS tensors, which aborts
        # in PyTorch 2.4. Check fake shape/dtype/device without those reductions.
        with FakeTensorMode() as mode:
            fake = forward(*(mode.from_tensor(t) for t in args), 64)
        self.assertEqual(fake.shape, actual.shape)
        self.assertEqual(fake.dtype, actual.dtype)
        self.assertEqual(fake.device, actual.device)


if __name__ == "__main__":
    unittest.main()

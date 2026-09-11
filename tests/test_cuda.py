"""CUDA regression tests; skipped unless a CUDA build and GPU are available."""
import unittest

import torch
from torch.autograd import gradcheck

from torch_deform_attn import _C, ms_deform_attn, ms_deform_attn_core_pytorch


@unittest.skipUnless(torch.cuda.is_available() and _C.with_cuda, "Requires CUDA extension and GPU")
class CUDAAttentionTest(unittest.TestCase):
    def inputs(self, dtype=torch.double, channels=2, device="cuda", batch=2):
        torch.manual_seed(13)
        shapes = torch.tensor([[3, 4], [2, 2]], device=device)
        starts = torch.tensor([0, 12], device=device)
        value = torch.randn(batch, 16, 2, channels, dtype=dtype, device=device, requires_grad=True)
        locations = (torch.rand(batch, 2, 2, 2, 2, 2, dtype=dtype, device=device) * 1.6 - 0.3).requires_grad_()
        weights = torch.rand(batch, 2, 2, 2, 2, dtype=dtype, device=device, requires_grad=True)
        return value, shapes, starts, locations, weights

    def test_reference_forward_backward(self):
        for dtype in (torch.float32, torch.float64):
            # Exercise upstream small, power-of-two, and large-channel reductions.
            for channels in (2, 32, 71, 1025):
                with self.subTest(dtype=dtype, channels=channels):
                    value, shapes, starts, loc, weights = self.inputs(dtype, channels)
                    actual = ms_deform_attn(value, shapes, starts, loc, weights, 1)
                    expected = ms_deform_attn_core_pytorch(value, shapes, loc, weights)
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
                    grad = torch.randn_like(actual)
                    for a, e in zip(torch.autograd.grad(actual, (value, loc, weights), grad),
                                    torch.autograd.grad(expected, (value, loc, weights), grad)):
                        torch.testing.assert_close(a, e, atol=2e-4, rtol=1e-4)

    def test_partial_batch_chunk(self):
        for batch, step in ((3, 2), (5, 2), (2, 64)):
            with self.subTest(batch=batch, step=step):
                value, shapes, starts, loc, weights = self.inputs(batch=batch)
                actual = ms_deform_attn(value, shapes, starts, loc, weights, step)
                expected = ms_deform_attn_core_pytorch(value, shapes, loc, weights)
                torch.testing.assert_close(actual, expected)
                grad = torch.randn_like(actual)
                for a, e in zip(torch.autograd.grad(actual, (value, loc, weights), grad),
                                torch.autograd.grad(expected, (value, loc, weights), grad)):
                    torch.testing.assert_close(a, e)

    def test_amp_and_compile(self):
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            for compile in (False, True):
                with self.subTest(dtype=dtype, compile=compile):
                    value, shapes, starts, loc, weights = self.inputs(dtype=dtype, batch=3)
                    loc = loc.detach().float().requires_grad_()
                    def run(v, s, i, l, w):
                        with torch.autocast("cuda", dtype=dtype):
                            return ms_deform_attn(v, s, i, l, w, 2)
                    fn = torch.compile(run, fullgraph=True) if compile else run
                    actual = fn(value, shapes, starts, loc, weights)
                    expected = ms_deform_attn_core_pytorch(value.float(), shapes, loc, weights.float())
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected)
                    for a, e in zip(torch.autograd.grad(actual.sum(), (value, loc, weights)),
                                    torch.autograd.grad(expected.sum(), (value, loc, weights))):
                        torch.testing.assert_close(a, e, rtol=0.02, atol=0.002)

    def test_compile_dynamic(self):
        fn = torch.compile(ms_deform_attn, fullgraph=True, dynamic=True)
        for batch in (3, 5):
            args = self.inputs(dtype=torch.float32, batch=batch)
            actual = fn(*args, 2)
            expected = ms_deform_attn_core_pytorch(args[0], args[1], args[3], args[4])
            torch.testing.assert_close(actual, expected)
            for a, e in zip(torch.autograd.grad(actual.sum(), (args[0], args[3], args[4])),
                            torch.autograd.grad(expected.sum(), (args[0], args[3], args[4]))):
                torch.testing.assert_close(a, e, atol=2e-5, rtol=1e-4)

    def test_gradcheck(self):
        self.assertTrue(gradcheck(ms_deform_attn, self.inputs(), fast_mode=True))

    def test_noncontiguous_and_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            value, shapes, starts, loc, weights = self.inputs()
            value = value.transpose(1, 2).contiguous().transpose(1, 2)
            loc = loc.transpose(1, 2).contiguous().transpose(1, 2)
            weights = weights.transpose(1, 2).contiguous().transpose(1, 2)
            actual = ms_deform_attn(value, shapes, starts, loc, weights)
            expected = ms_deform_attn_core_pytorch(value, shapes, loc, weights)
            torch.testing.assert_close(actual, expected)
            grad = torch.randn(2, 4, 2, dtype=value.dtype, device=value.device).transpose(1, 2)
            for a, e in zip(torch.autograd.grad(actual, (value, loc, weights), grad),
                            torch.autograd.grad(expected, (value, loc, weights), grad)):
                torch.testing.assert_close(a, e)
        stream.synchronize()

    def test_invalid_inputs(self):
        value, shapes, starts, loc, weights = self.inputs()
        with self.assertRaisesRegex(RuntimeError, "positive"):
            ms_deform_attn(value, shapes, starts, loc, weights, 0)
        with self.assertRaisesRegex(RuntimeError, "same CUDA device"):
            ms_deform_attn(value, shapes.cpu(), starts, loc, weights)
        with self.assertRaisesRegex(RuntimeError, "dtypes must match"):
            ms_deform_attn(value, shapes, starts, loc.float(), weights)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two GPUs")
    def test_noncurrent_device(self):
        with torch.cuda.device(0):
            value, shapes, starts, loc, weights = self.inputs(device="cuda:1")
            actual = ms_deform_attn(value, shapes, starts, loc, weights)
            expected = ms_deform_attn_core_pytorch(value, shapes, loc, weights)
            torch.testing.assert_close(actual, expected)
            actual.sum().backward()
            self.assertEqual(torch.cuda.current_device(), 0)


@unittest.skipUnless(torch.cuda.is_available() and not _C.with_cuda, "Requires GPU and CPU-only build")
class CPUOnlyBuildTest(unittest.TestCase):
    def test_cuda_input_error(self):
        args = CUDAAttentionTest().inputs()
        with self.assertRaisesRegex(RuntimeError, "built without CUDA"):
            ms_deform_attn(*args)

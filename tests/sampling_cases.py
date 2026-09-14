"""Shared CPU/CUDA sampling cases; reference reconstructs arbitrary offsets."""

import torch

from torch_ms_deform_attn import ms_deform_attn, ms_deform_attn_core_pytorch


class SamplingCases:
    def compare_sampling(self, value, shapes, starts, locations, weights):
        # Keep slicing in autograd so duplicate/overlapping levels accumulate
        # reference gradients into their original value cells.
        reordered = torch.cat(
            [
                value[:, start : start + h * w]
                for (h, w), start in zip(shapes.tolist(), starts.tolist())
            ],
            dim=1,
        )
        expected = ms_deform_attn_core_pytorch(reordered, shapes, locations, weights)
        grad = torch.randn_like(expected)
        tensors = (value, locations, weights)
        expected_grads = torch.autograd.grad(expected, tensors, grad)
        # Atomics may reorder additions. Repeated CUDA results need numerical,
        # not bitwise, equivalence, especially for colliding samples.
        atol, rtol = (2e-4, 1e-4) if value.dtype == torch.float32 else (1e-10, 1e-9)
        for _ in range(3):
            actual = ms_deform_attn(value, shapes, starts, locations, weights, 2)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            for a, e in zip(torch.autograd.grad(actual, tensors, grad), expected_grads):
                torch.testing.assert_close(a, e, atol=atol, rtol=rtol)

    def test_padding_support_and_channel_sizes(self):
        for h, w in ((2, 3), (1, 3), (3, 1), (1, 1)):
            # Sides, corners, outside support, and normalized coordinates 0/1.
            # Fractional pixels avoid finite-difference ambiguity; 0/1 follows
            # grid_sample's chosen bilinear gradient convention, no gradcheck.
            pixels = [
                (-0.75, 0.3),
                (w - 0.25, 0.3),
                (0.3, -0.75),
                (0.3, h - 0.25),
                (-1.25, 0.3),
                (w + 0.25, 0.3),
                (0.3, -1.25),
                (0.3, h + 0.25),
                (-0.75, -0.75),
                (w - 0.25, -0.75),
                (-0.75, h - 0.25),
                (w - 0.25, h - 0.25),
                (0.3, 0.7),
                (-0.5, -0.5),
                (w - 0.5, h - 0.5),
            ]
            for dtype in (torch.float32, torch.float64):
                for channels in (1, 7, 32):
                    with self.subTest(h=h, w=w, dtype=dtype, channels=channels):
                        torch.manual_seed(19)
                        device = self.sampling_device
                        shapes = torch.tensor([[h, w]], device=device)
                        starts = torch.tensor([0], device=device)
                        value = torch.randn(
                            1, h * w, 2, channels, dtype=dtype, device=device, requires_grad=True
                        )
                        locations = (
                            torch.tensor(pixels, dtype=dtype, device=device) + 0.5
                        ) / torch.tensor([w, h], device=device)
                        locations = (
                            locations.view(1, -1, 1, 1, 1, 2)
                            .repeat(1, 1, 2, 1, 1, 1)
                            .requires_grad_()
                        )
                        weights = torch.randn(
                            1, len(pixels), 2, 1, 1, dtype=dtype, device=device, requires_grad=True
                        )
                        self.compare_sampling(value, shapes, starts, locations, weights)

    def test_offsets_collisions_and_noncontiguous_metadata(self):
        for offsets in ([2, 12], [12, 2], [3, 3], [0, 3]):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(offsets=offsets, dtype=dtype):
                    torch.manual_seed(23)
                    device = self.sampling_device
                    shapes = torch.tensor([[2, 3], [2, 3]], device=device).t().contiguous().t()
                    starts = torch.tensor([offsets[0], -1, offsets[1], -1], device=device)[::2]
                    value = (
                        torch.randn(1, 20, 4, 7, device=device, dtype=dtype)
                        .transpose(1, 2)
                        .contiguous()
                        .transpose(1, 2)
                        .requires_grad_()
                    )
                    locations = (
                        torch.tensor([0.43, 0.61], dtype=dtype, device=device)
                        .repeat(1, 32, 4, 2, 4, 1)
                        .transpose(1, 2)
                        .contiguous()
                        .transpose(1, 2)
                        .requires_grad_()
                    )
                    weights = (
                        torch.randn(1, 32, 4, 2, 4, dtype=dtype, device=device)
                        .transpose(1, 2)
                        .contiguous()
                        .transpose(1, 2)
                        .requires_grad_()
                    )
                    for tensor in (value, shapes, starts, locations, weights):
                        self.assertFalse(tensor.is_contiguous())
                    self.compare_sampling(value, shapes, starts, locations, weights)

"""Hugging Face distribution of the torch-ms-deform-attn implementation."""

import torch

from . import layers
from ._registrations import backward
from .functional import ms_deform_attn


# Preserve the HF entry points, including explicit FP16/BF16 calls.
def ms_deform_attn_forward(
    value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step
):
    return ms_deform_attn(
        value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step
    )


def ms_deform_attn_backward(
    value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step
):
    dtype = value.dtype
    low_precision = dtype in (torch.float16, torch.bfloat16)
    if low_precision:
        if sampling_loc.dtype != dtype or attn_weight.dtype != dtype:
            raise RuntimeError("Floating input dtypes must match")
        if grad_output.dtype not in (dtype, torch.float32):
            raise RuntimeError("grad_output dtype must match value or FP32 computation")
        value, sampling_loc, attn_weight, grad_output = (
            tensor.float() for tensor in (value, sampling_loc, attn_weight, grad_output)
        )
    gradients = backward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc,
        attn_weight,
        grad_output,
        im2col_step,
    )
    return [gradient.to(dtype) if low_precision else gradient for gradient in gradients]


__all__ = ["layers", "ms_deform_attn", "ms_deform_attn_forward", "ms_deform_attn_backward"]

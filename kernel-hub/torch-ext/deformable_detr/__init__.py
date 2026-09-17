"""Hugging Face distribution of the torch-ms-deform-attn implementation."""

from . import layers
from ._ops import ops
from .functional import ms_deform_attn


# Preserve the existing low-level HF API (FP32/FP64, contiguous tensors).
def ms_deform_attn_forward(
    value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step
):
    return ops.ms_deform_attn_forward(
        value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step
    )


def ms_deform_attn_backward(
    value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step
):
    return ops.ms_deform_attn_backward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc,
        attn_weight,
        grad_output,
        im2col_step,
    )


__all__ = ["layers", "ms_deform_attn", "ms_deform_attn_forward", "ms_deform_attn_backward"]

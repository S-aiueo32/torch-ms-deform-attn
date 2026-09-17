"""Transformers' Kernel Hub layer contract."""

from torch import nn

from .functional import MSDeformAttnFunction, ms_deform_attn

MultiScaleDeformableAttentionFunction = MSDeformAttnFunction


class MultiScaleDeformableAttention(nn.Module):
    def forward(
        self,
        value,
        value_spatial_shapes,
        value_spatial_shapes_list,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        return ms_deform_attn(
            value,
            value_spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )

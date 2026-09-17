# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Modified: packaged extension import and public functional API.

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ._ops import forward as _forward

_MPS_AUTOCAST_AVAILABLE = torch.amp.autocast_mode.is_autocast_available("mps")
if TYPE_CHECKING:
    from jaxtyping import Float, Int64


class MSDeformAttnFunction:
    """Compatibility entry point for upstream ``MSDeformAttnFunction.apply`` calls."""

    @staticmethod
    def apply(
        value: Float[torch.Tensor, "N S M D"],
        shapes: Int64[torch.Tensor, "L 2"],
        starts: Int64[torch.Tensor, "L"],  # noqa: F821 - jaxtyping dimension
        locations: Float[torch.Tensor, "N Q M L P 2"],
        weights: Float[torch.Tensor, "N Q M L P"],
        im2col_step: int,
    ) -> Float[torch.Tensor, "N Q M*D"]:
        return ms_deform_attn(value, shapes, starts, locations, weights, im2col_step)


def ms_deform_attn_core_pytorch(
    value: Float[torch.Tensor, "N S M D"],
    value_spatial_shapes: Int64[torch.Tensor, "L 2"] | Sequence[Sequence[int]],
    sampling_locations: Float[torch.Tensor, "N Q M L P 2"],
    attention_weights: Float[torch.Tensor, "N Q M L P"],
) -> Float[torch.Tensor, "N Q M*D"]:
    # Independent grid_sample reference for correctness checks and benchmarks.
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(N_, M_ * D_, Lq_)
    )
    return output.transpose(1, 2).contiguous()


def ms_deform_attn(
    value: Float[torch.Tensor, "N S M D"],
    spatial_shapes: Int64[torch.Tensor, "L 2"],
    level_start_index: Int64[torch.Tensor, "L"],  # noqa: F821 - jaxtyping dimension
    sampling_locations: Float[torch.Tensor, "N Q M L P 2"],
    attention_weights: Float[torch.Tensor, "N Q M L P"],
    im2col_step: int = 64,
) -> Float[torch.Tensor, "N Q M*D"]:
    """Multi-scale deformable attention on CPU, CUDA, or MPS, with first-order autograd.

    Args:
        value: [N, S, M, D], float16, bfloat16, float32, or float64.
        spatial_shapes: [L, 2] int64, containing (height, width) for each level.
        level_start_index: [L] int64, each level's offset in S.
        sampling_locations: [N, Q, M, L, P, 2], normalized (x, y).
        attention_weights: [N, Q, M, L, P], same dtype as value.
        im2col_step: Maximum CUDA batch chunk size; CPU/MPS do not chunk by it.

    All inputs must be on the same device. MPS excludes float64 and empty dimensions.
    Samples use bilinear interpolation with zero
    padding and align_corners=False. Weights are used as supplied, without
    normalization. Returns [N, Q, M * D]. Explicit low-precision inputs are computed
    in float32 and returned in the input dtype; under autocast the output stays float32.
    Higher-order gradients are unsupported.
    """
    # Keep interpolation and gradient accumulation in float32 under AMP.
    # Casting here (outside the opaque operator) preserves gradients to low-precision inputs.
    device_type = value.device.type
    autocast_supported = device_type in ("cpu", "cuda") or (
        device_type == "mps" and _MPS_AUTOCAST_AVAILABLE
    )
    if autocast_supported and torch.is_autocast_enabled(device_type):
        value, sampling_locations, attention_weights = (
            t.float() if t.dtype in (torch.float16, torch.bfloat16) else t
            for t in (value, sampling_locations, attention_weights)
        )
        with torch.autocast(device_type=device_type, enabled=False):
            return _forward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                im2col_step,
            )
    if device_type in ("cpu", "cuda", "mps") and value.dtype in (torch.float16, torch.bfloat16):
        if sampling_locations.dtype != value.dtype or attention_weights.dtype != value.dtype:
            raise RuntimeError("Floating input dtypes must match outside autocast")
        return _forward(
            value.float(),
            spatial_shapes,
            level_start_index,
            sampling_locations.float(),
            attention_weights.float(),
            im2col_step,
        ).to(value.dtype)
    return _forward(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
    )

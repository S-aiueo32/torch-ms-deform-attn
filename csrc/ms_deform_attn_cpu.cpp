/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/
// Modified: standalone CPU forward/backward implementation and input validation.

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <vector>

namespace {
void check_inputs(const at::Tensor& value, const at::Tensor& shapes,
                  const at::Tensor& starts, const at::Tensor& locations,
                  const at::Tensor& weights, int step) {
    TORCH_CHECK(value.device().is_cpu() && shapes.device().is_cpu() &&
                starts.device().is_cpu() && locations.device().is_cpu() &&
                weights.device().is_cpu(), "All inputs must be CPU tensors");
    TORCH_CHECK(value.scalar_type() == at::kFloat || value.scalar_type() == at::kDouble,
                "CPU deformable attention supports float32 and float64");
    TORCH_CHECK(locations.scalar_type() == value.scalar_type() &&
                weights.scalar_type() == value.scalar_type(), "Floating input dtypes must match");
    TORCH_CHECK(shapes.scalar_type() == at::kLong && starts.scalar_type() == at::kLong,
                "spatial_shapes and level_start_index must be int64");
    TORCH_CHECK(value.dim() == 4, "value must have shape [N, S, M, D]");
    TORCH_CHECK(shapes.dim() == 2 && shapes.size(1) == 2,
                "spatial_shapes must have shape [L, 2]");
    TORCH_CHECK(starts.dim() == 1 && starts.size(0) == shapes.size(0),
                "level_start_index must have shape [L]");
    TORCH_CHECK(locations.dim() == 6 && locations.size(0) == value.size(0) &&
                locations.size(2) == value.size(2) && locations.size(3) == shapes.size(0) &&
                locations.size(5) == 2, "Invalid sampling_locations shape");
    TORCH_CHECK(weights.sizes() == locations.sizes().slice(0, 5),
                "attention_weights must match sampling_locations without the coordinate dimension");
    TORCH_CHECK(step > 0, "im2col_step must be positive");
    for (int64_t l = 0; l < shapes.size(0); ++l) {
        const auto h = shapes[l][0].item<int64_t>();
        const auto w = shapes[l][1].item<int64_t>();
        const auto start = starts[l].item<int64_t>();
        TORCH_CHECK(h > 0 && w > 0 && start >= 0 && start <= value.size(1) &&
                    h <= (value.size(1) - start) / w,
                    "Spatial level exceeds the value tensor");
    }
}

// Each worker owns whole batch elements, so backward accumulation needs no atomics.
// Coordinates follow grid_sample's bilinear, zero-padding, align_corners=False convention.
template <typename scalar_t, bool backward>
void kernel(const at::Tensor& value, const at::Tensor& shapes,
            const at::Tensor& starts, const at::Tensor& locations,
            const at::Tensor& weights, at::Tensor& output,
            const at::Tensor& grad_output, at::Tensor& grad_value,
            at::Tensor& grad_locations, at::Tensor& grad_weights) {
    const auto N = value.size(0), S = value.size(1), M = value.size(2), D = value.size(3);
    const auto Q = locations.size(1), L = locations.size(3), P = locations.size(4);
    const auto* v = value.data_ptr<scalar_t>();
    const auto* shape = shapes.data_ptr<int64_t>();
    const auto* start = starts.data_ptr<int64_t>();
    const auto* loc = locations.data_ptr<scalar_t>();
    const auto* weight = weights.data_ptr<scalar_t>();
    auto* out = backward ? nullptr : output.data_ptr<scalar_t>();
    const auto* go = backward ? grad_output.data_ptr<scalar_t>() : nullptr;
    auto* gv = backward ? grad_value.data_ptr<scalar_t>() : nullptr;
    auto* gl = backward ? grad_locations.data_ptr<scalar_t>() : nullptr;
    auto* gw = backward ? grad_weights.data_ptr<scalar_t>() : nullptr;
    at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t n = begin; n < end; ++n)
        for (int64_t q = 0; q < Q; ++q)
        for (int64_t m = 0; m < M; ++m)
        for (int64_t l = 0; l < L; ++l)
        for (int64_t p = 0; p < P; ++p) {
            const int64_t i = ((((n * Q + q) * M + m) * L + l) * P + p);
            const auto H = shape[2 * l], W = shape[2 * l + 1];
            const scalar_t x = loc[2 * i] * W - scalar_t(0.5);
            const scalar_t y = loc[2 * i + 1] * H - scalar_t(0.5);
            if (!(x > -1 && x < W && y > -1 && y < H)) continue;
            const int64_t x0 = std::floor(x), y0 = std::floor(y);
            const scalar_t dx = x - x0, dy = y - y0;
            for (int64_t c = 0; c < D; ++c) {
                const int64_t oi = ((n * Q + q) * M + m) * D + c;
                for (int iy = 0; iy < 2; ++iy)
                for (int ix = 0; ix < 2; ++ix) {
                    const auto xx = x0 + ix, yy = y0 + iy;
                    if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;
                    const int64_t vi = ((n * S + start[l] + yy * W + xx) * M + m) * D + c;
                    const scalar_t wx = ix ? dx : 1 - dx, wy = iy ? dy : 1 - dy;
                    if (backward) {
                        const scalar_t g = go[oi];
                        gv[vi] += g * weight[i] * wx * wy;
                        gw[i] += g * v[vi] * wx * wy;
                        gl[2 * i] += g * weight[i] * v[vi] * (ix ? W : -W) * wy;
                        gl[2 * i + 1] += g * weight[i] * v[vi] * wx * (iy ? H : -H);
                    } else {
                        out[oi] += v[vi] * wx * wy * weight[i];
                    }
                }
            }
        }
    });
}
} // namespace

at::Tensor ms_deform_attn_cpu_forward(
    const at::Tensor& value, const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index, const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight, const int im2col_step) {
    check_inputs(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
    auto output = at::zeros({value.size(0), sampling_loc.size(1), value.size(2) * value.size(3)}, value.options());
    at::Tensor unused;
    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_cpu_forward", [&] {
        kernel<scalar_t, false>(value.contiguous(), spatial_shapes.contiguous(),
            level_start_index.contiguous(), sampling_loc.contiguous(), attn_weight.contiguous(),
            output, unused, unused, unused, unused);
    });
    return output;
}

std::vector<at::Tensor> ms_deform_attn_cpu_backward(
    const at::Tensor& value, const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index, const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight, const at::Tensor& grad_output, const int im2col_step) {
    check_inputs(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
    TORCH_CHECK(grad_output.device().is_cpu() && grad_output.scalar_type() == value.scalar_type(),
                "grad_output must be on CPU with the same dtype as value");
    TORCH_CHECK(grad_output.dim() == 3 && grad_output.size(0) == value.size(0) &&
                grad_output.size(1) == sampling_loc.size(1) &&
                grad_output.size(2) == value.size(2) * value.size(3), "Invalid grad_output shape");
    auto grad_value = at::zeros(value.sizes(), value.options());
    auto grad_locations = at::zeros(sampling_loc.sizes(), sampling_loc.options());
    auto grad_weights = at::zeros(attn_weight.sizes(), attn_weight.options());
    at::Tensor unused;
    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_cpu_backward", [&] {
        kernel<scalar_t, true>(value.contiguous(), spatial_shapes.contiguous(),
            level_start_index.contiguous(), sampling_loc.contiguous(), attn_weight.contiguous(),
            unused, grad_output.contiguous(), grad_value, grad_locations, grad_weights);
    });
    return {grad_value, grad_locations, grad_weights};
}

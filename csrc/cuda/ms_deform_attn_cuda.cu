/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/
// Modified: current tensor APIs, input validation, and CUDA device guarding.

#include <vector>
#include "ms_deform_im2col_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace {
void check_inputs(const at::Tensor& value, const at::Tensor& shapes,
                  const at::Tensor& starts, const at::Tensor& loc,
                  const at::Tensor& weight, int step) {
    TORCH_CHECK(value.is_cuda(), "value must be CUDA");
    for (const auto& tensor : {value, shapes, starts, loc, weight}) {
        TORCH_CHECK(tensor.device() == value.device(), "All inputs must be on the same CUDA device");
        TORCH_CHECK(tensor.is_contiguous(), "CUDA inputs must be contiguous");
    }
    TORCH_CHECK(value.scalar_type() == at::kFloat || value.scalar_type() == at::kDouble,
                "CUDA deformable attention supports float32 and float64");
    TORCH_CHECK(loc.scalar_type() == value.scalar_type() && weight.scalar_type() == value.scalar_type(),
                "Floating input dtypes must match");
    TORCH_CHECK(shapes.scalar_type() == at::kLong && starts.scalar_type() == at::kLong,
                "spatial_shapes and level_start_index must be int64");
    TORCH_CHECK(value.dim() == 4 && value.size(0) > 0 && value.size(1) > 0 &&
                value.size(2) > 0 && value.size(3) > 0, "CUDA value must be nonempty [N, S, M, D]");
    TORCH_CHECK(shapes.dim() == 2 && shapes.size(1) == 2 && shapes.size(0) > 0,
                "spatial_shapes must be nonempty [L, 2]");
    TORCH_CHECK(starts.dim() == 1 && starts.size(0) == shapes.size(0), "Invalid level_start_index shape");
    TORCH_CHECK(loc.dim() == 6 && loc.size(0) == value.size(0) && loc.size(1) > 0 &&
                loc.size(2) == value.size(2) && loc.size(3) == shapes.size(0) &&
                loc.size(4) > 0 && loc.size(5) == 2, "Invalid or empty sampling_locations shape");
    TORCH_CHECK(weight.sizes() == loc.sizes().slice(0, 5), "Invalid attention_weights shape");
    TORCH_CHECK(step > 0, "im2col_step must be positive");
}
} // namespace

at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    check_inputs(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
    const c10::cuda::CUDAGuard device_guard(value.device());

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    // Every output element is assigned by the forward kernel.
    auto output = at::empty({batch, num_query, num_heads, channels}, value.options());

    auto per_value_size = int64_t(spatial_size) * num_heads * channels;
    auto per_sample_loc_size = int64_t(num_query) * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = int64_t(num_query) * num_heads * num_levels * num_point;
    for (int n = 0; n < batch; n += im2col_step_)
    {
        const int batch_n = std::min(im2col_step_, batch - n);
        auto columns = output.narrow(0, n, batch_n);
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
            ms_deformable_im2col_cuda(at::cuda::getCurrentCUDAStream(),
                value.data_ptr<scalar_t>() + int64_t(n) * per_value_size,
                spatial_shapes.data_ptr<int64_t>(),
                level_start_index.data_ptr<int64_t>(),
                sampling_loc.data_ptr<scalar_t>() + int64_t(n) * per_sample_loc_size,
                attn_weight.data_ptr<scalar_t>() + int64_t(n) * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                columns.data_ptr<scalar_t>());

        }));
    }

    output = output.view({batch, num_query, num_heads*channels});

    return output;
}


std::vector<at::Tensor> ms_deform_attn_cuda_backward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{

    check_inputs(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
    TORCH_CHECK(grad_output.device() == value.device() && grad_output.scalar_type() == value.scalar_type(),
                "grad_output device and dtype must match value");
    TORCH_CHECK(grad_output.is_contiguous(), "CUDA grad_output must be contiguous");
    TORCH_CHECK(grad_output.dim() == 3 && grad_output.size(0) == value.size(0) &&
                grad_output.size(1) == sampling_loc.size(1) &&
                grad_output.size(2) == value.size(2) * value.size(3), "Invalid grad_output shape");
    const c10::cuda::CUDAGuard device_guard(value.device());

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    auto grad_value = at::zeros_like(value);
    auto grad_sampling_loc = at::zeros_like(sampling_loc);
    auto grad_attn_weight = at::zeros_like(attn_weight);

    auto per_value_size = int64_t(spatial_size) * num_heads * channels;
    auto per_sample_loc_size = int64_t(num_query) * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = int64_t(num_query) * num_heads * num_levels * num_point;

    for (int n = 0; n < batch; n += im2col_step_)
    {
        const int batch_n = std::min(im2col_step_, batch - n);
        auto grad_output_g = grad_output.narrow(0, n, batch_n);
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
            ms_deformable_col2im_cuda(at::cuda::getCurrentCUDAStream(),
                                    grad_output_g.data_ptr<scalar_t>(),
                                    value.data_ptr<scalar_t>() + int64_t(n) * per_value_size,
                                    spatial_shapes.data_ptr<int64_t>(),
                                    level_start_index.data_ptr<int64_t>(),
                                    sampling_loc.data_ptr<scalar_t>() + int64_t(n) * per_sample_loc_size,
                                    attn_weight.data_ptr<scalar_t>() + int64_t(n) * per_attn_weight_size,
                                    batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                                    grad_value.data_ptr<scalar_t>() +  int64_t(n) * per_value_size,
                                    grad_sampling_loc.data_ptr<scalar_t>() + int64_t(n) * per_sample_loc_size,
                                    grad_attn_weight.data_ptr<scalar_t>() + int64_t(n) * per_attn_weight_size);

        }));
    }

    return {
        grad_value, grad_sampling_loc, grad_attn_weight
    };
}

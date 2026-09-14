/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from
* https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/
// Modified: standalone CPU forward/backward implementation and input
// validation.

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <atomic>
#include <cmath>
#include <vector>

#include "csrc/ms_deform_attn_cpu.h"

namespace {
void check_inputs(const at::Tensor &value, const at::Tensor &shapes,
                  const at::Tensor &starts, const at::Tensor &locations,
                  const at::Tensor &weights, int step) {
  TORCH_CHECK(value.device().is_cpu() && shapes.device().is_cpu() &&
                  starts.device().is_cpu() && locations.device().is_cpu() &&
                  weights.device().is_cpu(),
              "All inputs must be CPU tensors");
  TORCH_CHECK(value.scalar_type() == at::kFloat ||
                  value.scalar_type() == at::kDouble,
              "CPU deformable attention supports float32 and float64");
  TORCH_CHECK(locations.scalar_type() == value.scalar_type() &&
                  weights.scalar_type() == value.scalar_type(),
              "Floating input dtypes must match");
  TORCH_CHECK(shapes.scalar_type() == at::kLong &&
                  starts.scalar_type() == at::kLong,
              "spatial_shapes and level_start_index must be int64");
  TORCH_CHECK(value.dim() == 4, "value must have shape [N, S, M, D]");
  TORCH_CHECK(shapes.dim() == 2 && shapes.size(1) == 2,
              "spatial_shapes must have shape [L, 2]");
  TORCH_CHECK(starts.dim() == 1 && starts.size(0) == shapes.size(0),
              "level_start_index must have shape [L]");
  TORCH_CHECK(locations.dim() == 6 && locations.size(0) == value.size(0) &&
                  locations.size(2) == value.size(2) &&
                  locations.size(3) == shapes.size(0) && locations.size(5) == 2,
              "Invalid sampling_locations shape");
  TORCH_CHECK(weights.sizes() == locations.sizes().slice(0, 5),
              "attention_weights must match sampling_locations without the "
              "coordinate dimension");
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

// Backward workers own whole (batch, head) pairs, so accumulation needs no
// atomics. Coordinates follow grid_sample's bilinear, zero-padding,
// align_corners=False convention.
template <typename scalar_t, bool backward>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void kernel(const at::Tensor &value, const at::Tensor &shapes,
            // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
            const at::Tensor &starts, const at::Tensor &locations,
            const at::Tensor &weights, at::Tensor &output,
            // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
            const at::Tensor &grad_output, at::Tensor &grad_value,
            at::Tensor &grad_locations, at::Tensor &grad_weights) {
  const auto N = value.size(0), S = value.size(1), M = value.size(2),
             D = value.size(3);
  const auto Q = locations.size(1), L = locations.size(3),
             P = locations.size(4);
  const auto *v = value.data_ptr<scalar_t>();
  const auto *shape = shapes.data_ptr<int64_t>();
  const auto *start = starts.data_ptr<int64_t>();
  const auto *loc = locations.data_ptr<scalar_t>();
  const auto *weight = weights.data_ptr<scalar_t>();
  auto *out = backward ? nullptr : output.data_ptr<scalar_t>();
  const auto *go = backward ? grad_output.data_ptr<scalar_t>() : nullptr;
  auto *gv = backward ? grad_value.data_ptr<scalar_t>() : nullptr;
  auto *gl = backward ? grad_locations.data_ptr<scalar_t>() : nullptr;
  auto *gw = backward ? grad_weights.data_ptr<scalar_t>() : nullptr;
  const int64_t work_items = backward ? N * M : N * Q * M;
  at::parallel_for(0, work_items, 1, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t n = backward ? task / M : task / (Q * M);
      const int64_t m = task % M;
      const int64_t query_begin = backward ? 0 : (task / M) % Q;
      const int64_t query_end = backward ? Q : query_begin + 1;
      for (int64_t q = query_begin; q < query_end; ++q)
        for (int64_t l = 0; l < L; ++l)
          for (int64_t p = 0; p < P; ++p) {
            const int64_t i = ((((n * Q + q) * M + m) * L + l) * P + p);
            const auto H = shape[2 * l], W = shape[2 * l + 1];
            const scalar_t x = loc[2 * i] * W - scalar_t(0.5);
            const scalar_t y = loc[2 * i + 1] * H - scalar_t(0.5);
            if (!(x > -1 && x < W && y > -1 && y < H))
              continue;
            const int64_t x0 = std::floor(x), y0 = std::floor(y);
            const scalar_t dx = x - x0, dy = y - y0;
            const scalar_t lx = 1 - dx, ly = 1 - dy;
            const scalar_t w00 = lx * ly, w01 = dx * ly;
            const scalar_t w10 = lx * dy, w11 = dx * dy;
            // The support check guarantees x0/y0 < W/H and x0+1/y0+1 >= 0.
            const bool valid00 = x0 >= 0 && y0 >= 0;
            const bool valid01 = x0 + 1 < W && y0 >= 0;
            const bool valid10 = x0 >= 0 && y0 + 1 < H;
            const bool valid11 = x0 + 1 < W && y0 + 1 < H;
            const int64_t base = (n * S + start[l]) * M * D + m * D;
            const int64_t offset00 = base + (y0 * W + x0) * M * D;
            const int64_t offset01 = offset00 + M * D;
            const int64_t offset10 = offset00 + W * M * D;
            const int64_t offset11 = offset10 + M * D;
            const int64_t oi = ((n * Q + q) * M + m) * D;
            const scalar_t attention = weight[i];
            scalar_t grad_attention = 0, grad_x = 0, grad_y = 0;
            for (int64_t c = 0; c < D; ++c) {
              const scalar_t v00 = valid00 ? v[offset00 + c] : 0;
              const scalar_t v01 = valid01 ? v[offset01 + c] : 0;
              const scalar_t v10 = valid10 ? v[offset10 + c] : 0;
              const scalar_t v11 = valid11 ? v[offset11 + c] : 0;
              const scalar_t sample =
                  v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
              if (backward) {
                const scalar_t g = go[oi + c];
                const scalar_t grad_sample = g * attention;
                if (valid00)
                  gv[offset00 + c] += grad_sample * w00;
                if (valid01)
                  gv[offset01 + c] += grad_sample * w01;
                if (valid10)
                  gv[offset10 + c] += grad_sample * w10;
                if (valid11)
                  gv[offset11 + c] += grad_sample * w11;
                grad_attention += g * sample;
                grad_x += grad_sample * (ly * (v01 - v00) + dy * (v11 - v10));
                grad_y += grad_sample * (lx * (v10 - v00) + dx * (v11 - v01));
              } else {
                out[oi + c] += sample * attention;
              }
            }
            if (backward) {
              gw[i] = grad_attention;
              gl[2 * i] = grad_x * W;
              gl[2 * i + 1] = grad_y * H;
            }
          }
    }
  });
}
}  // namespace

// Keep build diagnostics in the same translation unit as the attention kernel.
const char *ms_deform_attn_cpu_parallel_backend() {
#if AT_PARALLEL_OPENMP && defined(_OPENMP)
  return "openmp";
#elif AT_PARALLEL_NATIVE
  return "native";
#else
  return "serial";
#endif
}

int64_t ms_deform_attn_cpu_parallel_worker_count(int64_t work_items) {
  TORCH_CHECK(work_items >= 0, "work_items must be nonnegative");
  std::atomic<int64_t> workers{0};
  at::parallel_for(0, work_items, 1, [&](int64_t, int64_t) {
    workers.fetch_add(1, std::memory_order_relaxed);
  });
  return workers.load(std::memory_order_relaxed);
}

at::Tensor ms_deform_attn_cpu_forward(const at::Tensor &value,
                                      const at::Tensor &spatial_shapes,
                                      const at::Tensor &level_start_index,
                                      const at::Tensor &sampling_loc,
                                      const at::Tensor &attn_weight,
                                      const int im2col_step) {
  check_inputs(value, spatial_shapes, level_start_index, sampling_loc,
               attn_weight, im2col_step);
  auto output = at::zeros(
      {value.size(0), sampling_loc.size(1), value.size(2) * value.size(3)},
      value.options());
  at::Tensor unused;
  AT_DISPATCH_FLOATING_TYPES(
      value.scalar_type(), "ms_deform_attn_cpu_forward", [&] {
        kernel<scalar_t, false>(
            value.contiguous(), spatial_shapes.contiguous(),
            level_start_index.contiguous(), sampling_loc.contiguous(),
            attn_weight.contiguous(), output, unused, unused, unused, unused);
      });
  return output;
}

std::vector<at::Tensor> ms_deform_attn_cpu_backward(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight, const at::Tensor &grad_output,
    const int im2col_step) {
  check_inputs(value, spatial_shapes, level_start_index, sampling_loc,
               attn_weight, im2col_step);
  TORCH_CHECK(grad_output.device().is_cpu() &&
                  grad_output.scalar_type() == value.scalar_type(),
              "grad_output must be on CPU with the same dtype as value");
  TORCH_CHECK(grad_output.dim() == 3 && grad_output.size(0) == value.size(0) &&
                  grad_output.size(1) == sampling_loc.size(1) &&
                  grad_output.size(2) == value.size(2) * value.size(3),
              "Invalid grad_output shape");
  auto grad_value = at::zeros(value.sizes(), value.options());
  auto grad_locations = at::zeros(sampling_loc.sizes(), sampling_loc.options());
  auto grad_weights = at::zeros(attn_weight.sizes(), attn_weight.options());
  at::Tensor unused;
  AT_DISPATCH_FLOATING_TYPES(
      value.scalar_type(), "ms_deform_attn_cpu_backward", [&] {
        kernel<scalar_t, true>(
            value.contiguous(), spatial_shapes.contiguous(),
            level_start_index.contiguous(), sampling_loc.contiguous(),
            attn_weight.contiguous(), unused, grad_output.contiguous(),
            grad_value, grad_locations, grad_weights);
      });
  return {grad_value, grad_locations, grad_weights};
}

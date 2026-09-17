// Copyright (c) 2020 SenseTime. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 [see LICENSE for details].
// Modified: Metal dispatch, input validation, and pipeline management.
#include "csrc/mps/ms_deform_attn_mps.h"

#include <ATen/Context.h>
#include <ATen/mps/MPSStream.h>
#include <torch/mps.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <limits>
#include <vector>

#include "csrc/mps/kernels.h"

namespace {
struct Dimensions {
  uint64_t n, s, m, d, q, l, p;
};

uint64_t product(std::initializer_list<uint64_t> sizes) {
  uint64_t result = 1;
  for (auto size : sizes) {
    TORCH_CHECK(size > 0 && result <= static_cast<uint64_t>(
                                          std::numeric_limits<int64_t>::max()) /
                                          size,
                "MPS inputs must be nonempty and fit 64-bit indexing");
    result *= size;
  }
  return result;
}

Dimensions validate(const at::Tensor &value, const at::Tensor &shapes,
                    const at::Tensor &starts, const at::Tensor &loc,
                    const at::Tensor &weights, int step) {
  TORCH_CHECK(value.is_mps(), "All inputs must be on the same MPS device");
  for (const auto &t : {shapes, starts, loc, weights})
    TORCH_CHECK(t.device() == value.device(),
                "All inputs must be on the same MPS device");
  TORCH_CHECK(value.scalar_type() == at::kFloat,
              "MPS requires float32 computation");
  TORCH_CHECK(loc.scalar_type() == value.scalar_type() &&
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
  TORCH_CHECK(loc.dim() == 6 && loc.size(0) == value.size(0) &&
                  loc.size(2) == value.size(2) &&
                  loc.size(3) == shapes.size(0) && loc.size(5) == 2,
              "Invalid sampling_locations shape");
  TORCH_CHECK(weights.sizes() == loc.sizes().slice(0, 5),
              "attention_weights must match sampling_locations shape");
  TORCH_CHECK(step > 0, "im2col_step must be positive");
  Dimensions d{static_cast<uint64_t>(value.size(0)),
               static_cast<uint64_t>(value.size(1)),
               static_cast<uint64_t>(value.size(2)),
               static_cast<uint64_t>(value.size(3)),
               static_cast<uint64_t>(loc.size(1)),
               static_cast<uint64_t>(shapes.size(0)),
               static_cast<uint64_t>(loc.size(4))};
  product({d.n, d.s, d.m, d.d, sizeof(float)});
  product({d.n, d.q, d.m, d.d, sizeof(float)});
  product({d.n, d.q, d.m, d.l, d.p, 2, sizeof(float)});
  // Metadata is small. Validate on the host before any shader can dereference
  // offsets; no feature/coordinate/weight tensor is transferred to the CPU.
  auto host_shapes = shapes.cpu().contiguous();
  auto host_starts = starts.cpu().contiguous();
  const auto *hw = host_shapes.const_data_ptr<int64_t>();
  const auto *offset = host_starts.const_data_ptr<int64_t>();
  for (uint64_t l = 0; l < d.l; ++l) {
    const auto h = hw[2 * l], w = hw[2 * l + 1], start = offset[l];
    TORCH_CHECK(h > 0 && w > 0 && start >= 0 && start <= value.size(1) &&
                    h <= (value.size(1) - start) / w,
                "Spatial level exceeds the value tensor");
  }
  return d;
}

// All access is serialized on PyTorch's MPS queue. MPS currently exposes one
// device; key the cache by its identity rather than creating a separate device.
class Pipelines {
 public:
  id<MTLDevice> device;
  id<MTLComputePipelineState> forward = nil, backward = nil;

  explicit Pipelines(id<MTLDevice> target) : device([target retain]) {
    NSError *error = nil;
    auto options = [[MTLCompileOptions alloc] init];
    options.fastMathEnabled = NO;
    auto library = [device newLibraryWithSource:@(msda_metal_source)
                                        options:options
                                          error:&error];
    [options release];
    TORCH_CHECK(library, "Metal shader compilation failed: ",
                [[error localizedDescription] UTF8String]);
    auto f = [library newFunctionWithName:@"msda_forward"];
    auto b = [library newFunctionWithName:@"msda_backward"];
    forward = [device newComputePipelineStateWithFunction:f error:&error];
    backward = [device newComputePipelineStateWithFunction:b error:&error];
    [f release];
    [b release];
    [library release];
    TORCH_CHECK(forward && backward, "Metal pipeline creation failed: ",
                [[error localizedDescription] UTF8String]);
  }
  ~Pipelines() {
    [forward release];
    [backward release];
    [device release];
  }
};

void bind(id<MTLComputeCommandEncoder> encoder, const at::Tensor &tensor,
          unsigned index, id<MTLDevice> device) {
  auto buffer = (id<MTLBuffer>)tensor.storage().data();
  TORCH_CHECK([buffer length] <= [device maxBufferLength],
              "MPS buffer exceeds device limit");
  [encoder setBuffer:buffer
              offset:tensor.storage_offset() * tensor.element_size()
             atIndex:index];
}

void run(const std::vector<at::Tensor> &tensors, Dimensions d, bool backward) {
  // Share PyTorch's encoder, including its pending fills/copies. Opening an
  // independent encoder can conflict with PyTorch's kernel coalescing.
  __block std::exception_ptr error;
  dispatch_sync(torch::mps::get_dispatch_queue(), ^{
    @autoreleasepool {
      try {
        auto stream = at::mps::getCurrentMPSStream();
        static Pipelines pipelines(stream->device());
        TORCH_CHECK(pipelines.device == stream->device(), "MPS device changed");
        auto pipeline = backward ? pipelines.backward : pipelines.forward;
        auto encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipeline];
        for (unsigned i = 0; i < tensors.size(); ++i)
          bind(encoder, tensors[i], i, pipelines.device);
        const auto dims_index = backward ? 9 : 6;
        [encoder setBytes:&d length:sizeof(d) atIndex:dims_index];
        const uint64_t width = backward ? [pipeline threadExecutionWidth] : 1;
        const uint64_t count = backward ? product({d.n, d.q, d.m, d.l, d.p})
                                        : product({d.n, d.q, d.m, d.d});
        // uint thread_position_in_grid is local to each dispatch. All tensor
        // offsets remain 64-bit, including when several dispatches are needed.
        const uint64_t chunk = (static_cast<uint64_t>(1) << 30) / width;
        const auto group =
            std::min<NSUInteger>(256, [pipeline maxTotalThreadsPerThreadgroup]);
        for (uint64_t begin = 0; begin < count; begin += chunk) {
          [encoder setBytes:&begin length:sizeof(begin) atIndex:dims_index + 1];
          [encoder dispatchThreads:MTLSizeMake(std::min(chunk, count - begin) *
                                                   width,
                                               1, 1)
              threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
        }
      } catch (...) {
        error = std::current_exception();
      }
    }
  });
  if (error)
    std::rethrow_exception(error);
}
}  // namespace

at::Tensor ms_deform_attn_mps_forward(const at::Tensor &value,
                                      const at::Tensor &shapes,
                                      const at::Tensor &starts,
                                      const at::Tensor &locations,
                                      const at::Tensor &weights, int step) {
  const auto d = validate(value, shapes, starts, locations, weights, step);
  auto output = at::empty({static_cast<int64_t>(d.n), static_cast<int64_t>(d.q),
                           static_cast<int64_t>(d.m * d.d)},
                          value.options());
  run({value.contiguous(), shapes.contiguous(), starts.contiguous(),
       locations.contiguous(), weights.contiguous(), output},
      d, false);
  return output;
}

std::vector<at::Tensor> ms_deform_attn_mps_backward(
    const at::Tensor &value, const at::Tensor &shapes, const at::Tensor &starts,
    const at::Tensor &locations, const at::Tensor &weights,
    const at::Tensor &grad, int step) {
  const auto d = validate(value, shapes, starts, locations, weights, step);
  TORCH_CHECK(grad.device() == value.device() &&
                  grad.scalar_type() == value.scalar_type() &&
                  grad.dim() == 3 &&
                  grad.size(0) == static_cast<int64_t>(d.n) &&
                  grad.size(1) == static_cast<int64_t>(d.q) &&
                  grad.size(2) == static_cast<int64_t>(d.m * d.d),
              "Invalid grad_output shape, dtype, or device");
  at::globalContext().alertNotDeterministic("ms_deform_attn_mps_backward");
  auto gv = at::zeros(value.sizes(), value.options());
  auto gl = at::empty(locations.sizes(), locations.options());
  auto gw = at::empty(weights.sizes(), weights.options());
  run({value.contiguous(), shapes.contiguous(), starts.contiguous(),
       locations.contiguous(), weights.contiguous(), grad.contiguous(), gv, gl,
       gw},
      d, true);
  return {gv, gl, gw};
}

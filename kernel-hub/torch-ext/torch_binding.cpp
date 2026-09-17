#include <torch/library.h>
#include <limits>

#include "cuda/ms_deform_attn_cuda.h"
#include "registration.h"

namespace {
int checked_step(int64_t step) {
  TORCH_CHECK(step > 0 && step <= std::numeric_limits<int>::max(),
              "im2col_step must be a positive int32");
  return static_cast<int>(step);
}
} // namespace

// The dispatcher requires int64_t, while upstream kernels take an int step.
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("ms_deform_attn_forward(Tensor value, Tensor shapes, Tensor starts, "
          "Tensor locations, Tensor weights, int step) -> Tensor");
  ops.impl("ms_deform_attn_forward", torch::kCUDA,
           [](const at::Tensor &v, const at::Tensor &s, const at::Tensor &i,
              const at::Tensor &l, const at::Tensor &w, int64_t step) {
             return ms_deform_attn_cuda_forward(v, s, i, l, w, checked_step(step));
           });
  ops.def("ms_deform_attn_backward(Tensor value, Tensor shapes, Tensor starts, "
          "Tensor locations, Tensor weights, Tensor grad, int step) -> Tensor[]");
  ops.impl("ms_deform_attn_backward", torch::kCUDA,
           [](const at::Tensor &v, const at::Tensor &s, const at::Tensor &i,
              const at::Tensor &l, const at::Tensor &w, const at::Tensor &g,
              int64_t step) {
             return ms_deform_attn_cuda_backward(v, s, i, l, w, g, checked_step(step));
           });
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

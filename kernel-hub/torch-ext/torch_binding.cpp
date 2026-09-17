#include <torch/library.h>
#include <limits>

#include "cuda/ms_deform_attn_cuda.h"
#include "dispatcher.h"
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
  ms_deform_attn::register_dispatcher(ops, &ms_deform_attn_cuda_forward,
                                     &ms_deform_attn_cuda_backward);
  ops.def("ms_deform_attn_forward(Tensor value, Tensor shapes, Tensor starts, "
          "Tensor locations, Tensor weights, int step, bool check_cuda_metadata=False) -> Tensor");
  ops.impl("ms_deform_attn_forward", c10::DispatchKey::CUDA,
           [](const at::Tensor &v, const at::Tensor &s, const at::Tensor &i,
              const at::Tensor &l, const at::Tensor &w, int64_t step,
              bool check_cuda_metadata) {
             return ms_deform_attn_cuda_forward(v, s, i, l, w, checked_step(step), check_cuda_metadata);
           });
  ops.def("ms_deform_attn_backward(Tensor value, Tensor shapes, Tensor starts, "
          "Tensor locations, Tensor weights, Tensor grad, int step, bool check_cuda_metadata=False) -> Tensor[]");
  ops.impl("ms_deform_attn_backward", c10::DispatchKey::CUDA,
           [](const at::Tensor &v, const at::Tensor &s, const at::Tensor &i,
              const at::Tensor &l, const at::Tensor &w, const at::Tensor &g,
              int64_t step, bool check_cuda_metadata) {
             return ms_deform_attn_cuda_backward(v, s, i, l, w, g, checked_step(step), check_cuda_metadata);
           });
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

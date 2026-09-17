#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#include <array>
#include <limits>
#include <tuple>

namespace ms_deform_attn {

inline int dispatcher_step(int64_t step) {
  TORCH_CHECK(step > 0 && step <= std::numeric_limits<int>::max(),
              "im2col_step must be a positive int32");
  return static_cast<int>(step);
}

inline std::array<at::Tensor, 5>
dispatcher_inputs(const at::Tensor &value, const at::Tensor &shapes,
                  const at::Tensor &starts, const at::Tensor &locations,
                  const at::Tensor &weights) {
  if (value.is_cuda()) {
    return {value.contiguous(), shapes.contiguous(), starts.contiguous(),
            locations.contiguous(), weights.contiguous()};
  }
  return {value, shapes, starts, locations, weights};
}

// Shared by the canonical extension and Kernel Hub binding. Keep native calls
// and layout normalization below the dispatcher, without a Python trampoline.
template <typename Forward, typename Backward>
void register_dispatcher(torch::Library &ops, Forward forward,
                         Backward backward) {
  ops.def("forward(Tensor value, Tensor shapes, Tensor starts, Tensor locations, "
          "Tensor weights, int step) -> Tensor");
  ops.impl("forward", c10::DispatchKey::CompositeExplicitAutograd,
           [forward](const at::Tensor &v, const at::Tensor &s,
                     const at::Tensor &i, const at::Tensor &l,
                     const at::Tensor &w, int64_t step) {
             const auto args = dispatcher_inputs(v, s, i, l, w);
             return forward(args[0], args[1], args[2], args[3], args[4],
                            dispatcher_step(step));
           });
  ops.def("backward(Tensor value, Tensor shapes, Tensor starts, Tensor locations, "
          "Tensor weights, Tensor grad, int step) -> (Tensor, Tensor, Tensor)");
  ops.impl("backward", c10::DispatchKey::CompositeExplicitAutograd,
           [backward](const at::Tensor &v, const at::Tensor &s,
                      const at::Tensor &i, const at::Tensor &l,
                      const at::Tensor &w, const at::Tensor &g, int64_t step) {
             const auto args = dispatcher_inputs(v, s, i, l, w);
             const auto grads = backward(args[0], args[1], args[2], args[3],
                                         args[4], g.contiguous(),
                                         dispatcher_step(step));
             return std::make_tuple(grads[0], grads[1], grads[2]);
           });
}

} // namespace ms_deform_attn

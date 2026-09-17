#pragma once

#include <ATen/ATen.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include <array>
#include <limits>
#include <tuple>

namespace ms_deform_attn {

using ForwardSignature = at::Tensor(const at::Tensor &, const at::Tensor &,
                                    const at::Tensor &, const at::Tensor &,
                                    const at::Tensor &, int64_t, bool);
using BackwardSignature = std::tuple<at::Tensor, at::Tensor, at::Tensor>(
    const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, bool);

// Keep autograd in the dispatcher so eager callers avoid Python schema binding.
// The forward/backward computations still go through their opaque operators:
// FakeTensor and AOTAutograd must see those operators, not the backend internals.
class ForwardAutograd : public torch::autograd::Function<ForwardAutograd> {
public:
  // All backward state is saved in ctx, so CppNode can capture and restore it
  // for compiled autograd. Backward invokes only the registered opaque operator.
  static constexpr bool is_traceable = true;

  static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                            const at::Tensor &v, const at::Tensor &s,
                            const at::Tensor &i, const at::Tensor &l,
                            const at::Tensor &w, int64_t step, bool check_cuda_metadata,
                            const c10::OperatorHandle &op) {
    ctx->save_for_backward({v, s, i, l, w});
    ctx->saved_data["step"] = step;
    ctx->saved_data["check_cuda_metadata"] = check_cuda_metadata;
    const auto &name = op.schema().name();
    ctx->saved_data["backward"] =
        name.substr(0, name.rfind("::") + 2) + "backward";
    at::AutoDispatchBelowAutograd guard;
    return op.typed<ForwardSignature>().call(v, s, i, l, w, step, check_cuda_metadata);
  }

  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::variable_list grads) {
    const auto args = ctx->get_saved_variables();
    const auto op = c10::Dispatcher::singleton()
                        .findSchemaOrThrow(
                            ctx->saved_data["backward"].toStringRef().c_str(), "")
                        .typed<BackwardSignature>();
    const auto result = op.call(args[0], args[1], args[2], args[3], args[4],
                                grads[0], ctx->saved_data["step"].toInt(),
                                ctx->saved_data["check_cuda_metadata"].toBool());
    return {std::get<0>(result), at::Tensor(), at::Tensor(),
            std::get<1>(result), std::get<2>(result), at::Tensor(), at::Tensor(),
            at::Tensor()};
  }
};

class BackwardAutograd : public torch::autograd::Function<BackwardAutograd> {
public:
  static torch::autograd::variable_list
  forward(torch::autograd::AutogradContext *, const at::Tensor &v,
          const at::Tensor &s, const at::Tensor &i, const at::Tensor &l,
          const at::Tensor &w, const at::Tensor &g, int64_t step,
          bool check_cuda_metadata,
          const c10::OperatorHandle &op) {
    at::AutoDispatchBelowAutograd guard;
    const auto result = op.typed<BackwardSignature>().call(v, s, i, l, w, g, step, check_cuda_metadata);
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }

  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *, torch::autograd::variable_list) {
    TORCH_CHECK(false, "No autograd formula for backward: higher-order gradients "
                       "are unsupported");
  }
};

inline void autograd_forward(const c10::OperatorHandle &op,
                             torch::jit::Stack *stack) {
  at::Tensor v, s, i, l, w;
  int64_t step;
  bool check_cuda_metadata;
  torch::jit::pop(stack, v, s, i, l, w, step, check_cuda_metadata);
  if (!at::GradMode::is_enabled() ||
      !(v.requires_grad() || l.requires_grad() || w.requires_grad())) {
    at::AutoDispatchBelowAutograd guard;
    torch::jit::push(stack, op.typed<ForwardSignature>().call(v, s, i, l, w, step, check_cuda_metadata));
    return;
  }
  torch::jit::push(stack, ForwardAutograd::apply(v, s, i, l, w, step, check_cuda_metadata, op));
}

inline void autograd_backward(const c10::OperatorHandle &op,
                              torch::jit::Stack *stack) {
  at::Tensor v, s, i, l, w, g;
  int64_t step;
  bool check_cuda_metadata;
  torch::jit::pop(stack, v, s, i, l, w, g, step, check_cuda_metadata);
  if (!at::GradMode::is_enabled() ||
      !(v.requires_grad() || l.requires_grad() || w.requires_grad() ||
        g.requires_grad())) {
    at::AutoDispatchBelowAutograd guard;
    const auto result = op.typed<BackwardSignature>().call(v, s, i, l, w, g, step, check_cuda_metadata);
    torch::jit::push(stack, std::get<0>(result), std::get<1>(result),
                     std::get<2>(result));
    return;
  }
  const auto result = BackwardAutograd::apply(v, s, i, l, w, g, step, check_cuda_metadata, op);
  torch::jit::push(stack, result[0], result[1], result[2]);
}

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
          "Tensor weights, int step, bool check_cuda_metadata=False) -> Tensor");
  ops.impl("forward", c10::DispatchKey::CompositeExplicitAutograd,
           [forward](const at::Tensor &v, const at::Tensor &s,
                     const at::Tensor &i, const at::Tensor &l,
                     const at::Tensor &w, int64_t step, bool check_cuda_metadata) {
             const auto args = dispatcher_inputs(v, s, i, l, w);
             return forward(args[0], args[1], args[2], args[3], args[4],
                            dispatcher_step(step), check_cuda_metadata);
           });
  ops.def("backward(Tensor value, Tensor shapes, Tensor starts, Tensor locations, "
          "Tensor weights, Tensor grad, int step, bool check_cuda_metadata=False) -> (Tensor, Tensor, Tensor)");
  ops.impl("backward", c10::DispatchKey::CompositeExplicitAutograd,
           [backward](const at::Tensor &v, const at::Tensor &s,
                      const at::Tensor &i, const at::Tensor &l,
                      const at::Tensor &w, const at::Tensor &g, int64_t step,
                      bool check_cuda_metadata) {
             const auto args = dispatcher_inputs(v, s, i, l, w);
             const auto grads = backward(args[0], args[1], args[2], args[3],
                                         args[4], g.contiguous(),
                                         dispatcher_step(step), check_cuda_metadata);
             return std::make_tuple(grads[0], grads[1], grads[2]);
           });
  ops.impl("forward", c10::DispatchKey::Autograd,
           torch::CppFunction::makeFromBoxedFunction<&autograd_forward>());
  ops.impl("backward", c10::DispatchKey::Autograd,
           torch::CppFunction::makeFromBoxedFunction<&autograd_backward>());
}

} // namespace ms_deform_attn

#include <torch/csrc/utils/pybind.h>
#include "ms_deform_attn_cpu.h"
#ifdef WITH_CUDA
#include "cuda/ms_deform_attn_cuda.h"
#endif

at::Tensor forward(const at::Tensor& value, const at::Tensor& shapes,
                   const at::Tensor& starts, const at::Tensor& loc,
                   const at::Tensor& weights, int step) {
    if (value.is_cuda()) {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_forward(value, shapes, starts, loc, weights, step);
#else
        TORCH_CHECK(false, "Extension built without CUDA support; rebuild with FORCE_CUDA=1");
#endif
    }
    return ms_deform_attn_cpu_forward(value, shapes, starts, loc, weights, step);
}

std::vector<at::Tensor> backward(const at::Tensor& value, const at::Tensor& shapes,
                               const at::Tensor& starts, const at::Tensor& loc,
                               const at::Tensor& weights, const at::Tensor& grad, int step) {
    if (value.is_cuda()) {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_backward(value, shapes, starts, loc, weights, grad, step);
#else
        TORCH_CHECK(false, "Extension built without CUDA support; rebuild with FORCE_CUDA=1");
#endif
    }
    return ms_deform_attn_cpu_backward(value, shapes, starts, loc, weights, grad, step);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ms_deform_attn_forward", &forward);
    m.def("ms_deform_attn_backward", &backward);
#ifdef WITH_CUDA
    m.attr("with_cuda") = true;
#else
    m.attr("with_cuda") = false;
#endif
}

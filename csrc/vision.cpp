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

#include <torch/csrc/utils/pybind.h>

#include <utility>
#include <vector>

#include "csrc/ms_deform_attn_cpu.h"
#include "cuda/index_utils.h"
#include "dispatcher.h"
#ifdef WITH_MPS
#include "mps/ms_deform_attn_mps.h"
#endif
#ifdef WITH_CUDA
#include "cuda/ms_deform_attn_cuda.h"
#endif

at::Tensor forward(const at::Tensor &value, const at::Tensor &shapes,
                   const at::Tensor &starts, const at::Tensor &loc,
                   const at::Tensor &weights, int step, bool check_cuda_metadata) {
  if (value.is_mps()) {
#ifdef WITH_MPS
    return ms_deform_attn_mps_forward(value, shapes, starts, loc, weights,
                                      step);
#else
    TORCH_CHECK(
        false, "Extension built without MPS support; rebuild with FORCE_MPS=1");
#endif
  }
  if (value.is_cuda()) {
#ifdef WITH_CUDA
    return ms_deform_attn_cuda_forward(value, shapes, starts, loc, weights,
                                       step, check_cuda_metadata);
#else
    TORCH_CHECK(
        false,
        "Extension built without CUDA support; rebuild with FORCE_CUDA=1");
#endif
  }
  return ms_deform_attn_cpu_forward(value, shapes, starts, loc, weights, step);
}

std::vector<at::Tensor>
backward(const at::Tensor &value, const at::Tensor &shapes,
         const at::Tensor &starts, const at::Tensor &loc,
         const at::Tensor &weights, const at::Tensor &grad, int step,
         bool check_cuda_metadata) {
  if (value.is_mps()) {
#ifdef WITH_MPS
    return ms_deform_attn_mps_backward(value, shapes, starts, loc, weights,
                                       grad, step);
#else
    TORCH_CHECK(
        false, "Extension built without MPS support; rebuild with FORCE_MPS=1");
#endif
  }
  if (value.is_cuda()) {
#ifdef WITH_CUDA
    return ms_deform_attn_cuda_backward(value, shapes, starts, loc, weights,
                                        grad, step, check_cuda_metadata);
#else
    TORCH_CHECK(
        false,
        "Extension built without CUDA support; rebuild with FORCE_CUDA=1");
#endif
  }
  return ms_deform_attn_cpu_backward(value, shapes, starts, loc, weights, grad,
                                     step);
}

TORCH_LIBRARY(torch_ms_deform_attn, ops) {
  ms_deform_attn::register_dispatcher(ops, &forward, &backward);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward", &forward, pybind11::arg("value"),
        pybind11::arg("shapes"), pybind11::arg("starts"), pybind11::arg("loc"),
        pybind11::arg("weights"), pybind11::arg("step"),
        pybind11::arg("check_cuda_metadata") = false);
  m.def("ms_deform_attn_backward", &backward, pybind11::arg("value"),
        pybind11::arg("shapes"), pybind11::arg("starts"), pybind11::arg("loc"),
        pybind11::arg("weights"), pybind11::arg("grad"), pybind11::arg("step"),
        pybind11::arg("check_cuda_metadata") = false);
  m.attr("cpu_parallel_backend") = ms_deform_attn_cpu_parallel_backend();
  m.def("_cpu_parallel_worker_count",
        &ms_deform_attn_cpu_parallel_worker_count);
  m.def("_check_cuda_indexing", &ms_deform_attn::check_cuda_indexing);
#ifdef WITH_MPS
  m.attr("with_mps") = true;
#else
  m.attr("with_mps") = false;
#endif
#ifdef WITH_CUDA
  m.attr("with_cuda") = true;
#else
  m.attr("with_cuda") = false;
#endif
}

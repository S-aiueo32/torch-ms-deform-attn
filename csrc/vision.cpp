#include <torch/csrc/utils/pybind.h>
#include "ms_deform_attn_cpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ms_deform_attn_forward", &ms_deform_attn_cpu_forward);
    m.def("ms_deform_attn_backward", &ms_deform_attn_cpu_backward);
}

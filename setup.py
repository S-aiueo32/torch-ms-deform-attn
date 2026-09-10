import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

force_cpu = os.getenv("FORCE_CPU", "0") == "1"
force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
if force_cpu and force_cuda:
    raise RuntimeError("FORCE_CPU and FORCE_CUDA cannot both be enabled")
if force_cuda and (CUDA_HOME is None or torch.version.cuda is None):
    raise RuntimeError("FORCE_CUDA requires a CUDA-enabled PyTorch and CUDA toolkit")
with_cuda = not force_cpu and CUDA_HOME is not None and torch.version.cuda is not None and (
    force_cuda or torch.cuda.is_available()
)
sources = ["csrc/vision.cpp", "csrc/ms_deform_attn_cpu.cpp"]
compile_args = {"cxx": ["-O3"]}
if with_cuda:
    sources.append("csrc/cuda/ms_deform_attn_cuda.cu")
    compile_args["nvcc"] = ["-O3"]

setup(
    ext_modules=[(CUDAExtension if with_cuda else CppExtension)(
        "torch_deform_attn._C",
        sources=sources,
        define_macros=[("WITH_CUDA", None)] if with_cuda else [],
        extra_compile_args=compile_args,
    )],
    cmdclass={"build_ext": BuildExtension},
)

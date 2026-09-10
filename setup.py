from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    ext_modules=[CppExtension(
        "ms_deform_attn._C",
        sources=["csrc/vision.cpp", "csrc/ms_deform_attn_cpu.cpp"],
        extra_compile_args=["-O3"],
    )],
    cmdclass={"build_ext": BuildExtension},
)

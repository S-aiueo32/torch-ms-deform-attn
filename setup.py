import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.errors import LinkError
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
    get_cxx_compiler,
)


def aten_parallel_backend():
    config = (Path(torch.__file__).resolve().parent / "include/ATen/Config.h").read_text()
    for macro, backend in (("OPENMP", "openmp"), ("NATIVE", "native")):
        if re.search(rf"^#define\s+AT_PARALLEL_{macro}\s+1\s*$", config, re.MULTILINE):
            return backend
    return "serial"


def macos_openmp_runtime(torch_dir):
    # Match the runtime already loaded by importing torch. Linking another libomp
    # can create a second thread pool or abort the process at import time.
    import ctypes

    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    loaded = set()
    for index in range(dyld._dyld_image_count()):
        name = dyld._dyld_get_image_name(index)
        if name:
            path = Path(os.fsdecode(name))
            if path.name in ("libomp.dylib", "libiomp5.dylib") and path.is_file():
                loaded.add(path.resolve())
    if len(loaded) > 1:
        raise RuntimeError("multiple OpenMP runtimes are already loaded by PyTorch")
    if loaded:
        return loaded.pop()
    bundled = torch_dir / "lib/libomp.dylib"
    if bundled.is_file():
        return bundled
    raise RuntimeError("cannot locate the OpenMP runtime used by PyTorch")


def without_macos_arch_flags(command):
    result = []
    tokens = iter(command)
    for token in tokens:
        if token == "-arch":
            next(tokens, None)
        else:
            result.append(token)
    return result


def macos_target_flags():
    # Importing torch confirms that its binary supports this process architecture.
    # A universal2 Python can otherwise request slices absent from torch/lib.
    architecture = platform.machine()
    flags = shlex.split(os.getenv("ARCHFLAGS", ""))
    for index, flag in enumerate(flags):
        if flag == "-arch" and (index + 1 == len(flags) or flags[index + 1] != architecture):
            raise RuntimeError(
                f"ARCHFLAGS must target the running Python/PyTorch architecture ({architecture})"
            )
    return ["-arch", architecture]


class TorchBdistWheel(bdist_wheel):
    def get_tag(self):
        python_tag, abi_tag, platform_tag = super().get_tag()
        if sys.platform == "darwin":
            # Retain wheel's binary-derived minimum macOS version, but do not
            # label a thin extension universal2 just because Python is universal2.
            platform_tag = re.sub(r"(?:universal2|arm64|x86_64)$", platform.machine(), platform_tag)
        return python_tag, abi_tag, platform_tag


class OptionalOpenMPBuildExtension(BuildExtension):
    def probe_openmp(self, compile_flags, link_flags):
        # Use the same C++ command and environment flags as the extension build.
        if self.use_ninja:
            # PyTorch's Ninja writer uses CXX with compiler_so's trailing flags.
            compile_command = shlex.split(get_cxx_compiler()) + list(self.compiler.compiler_so[1:])
        else:
            # Recent setuptools selects compiler_so_cxx for .cpp files. Keep the
            # entire command, including wrappers such as "ccache clang++".
            compile_command = list(
                getattr(self.compiler, "compiler_so_cxx", self.compiler.compiler_so)
            )
        target_flags = getattr(self, "target_flags", [])
        if target_flags:
            compile_command = without_macos_arch_flags(compile_command)
        with tempfile.TemporaryDirectory(prefix="torch-ms-deform-attn-openmp-") as directory:
            source = Path(directory) / "probe.cpp"
            obj = Path(directory) / "probe.o"
            executable = Path(directory) / "probe"
            source.write_text(
                "#include <omp.h>\n"
                "#ifndef _OPENMP\n#error OpenMP compiler support is disabled\n#endif\n"
                "int main() { int count = 0;\n"
                "#pragma omp parallel reduction(+:count)\n"
                "{ count += omp_get_thread_num() >= 0; }\n"
                "return count < 1; }\n"
            )
            command = (
                compile_command + target_flags + compile_flags + ["-c", str(source), "-o", str(obj)]
            )
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode:
                detail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-4:])
                raise RuntimeError(detail or f"compiler exited with status {result.returncode}")
            try:
                # The extension also links through setuptools, even with Ninja.
                # Let it select the configured C++ linker and retain architecture,
                # sysroot, wrapper, and environment flags without duplicating them.
                self.compiler.link_executable(
                    [str(obj)],
                    str(executable),
                    extra_postargs=target_flags + link_flags,
                    target_lang="c++",
                )
            except LinkError as error:
                raise RuntimeError(f"OpenMP runtime link check failed: {error}") from error

    def openmp_flags(self):
        if self.compiler.compiler_type != "unix":
            raise RuntimeError("automatic OpenMP configuration supports Unix C++ compilers")
        torch_dir = Path(torch.__file__).resolve().parent
        prefix = os.getenv("OMP_PREFIX")
        explicit_include = Path(prefix).expanduser().resolve() / "include" if prefix else None
        if sys.platform == "darwin":
            runtime = macos_openmp_runtime(torch_dir)
            includes = [
                torch_dir / "include",
                explicit_include,
                Path("/opt/homebrew/opt/libomp/include"),
                Path("/usr/local/opt/libomp/include"),
                None,
            ]
            compile_options = [["-Xpreprocessor", "-fopenmp"], ["-fopenmp"]]
            link_flags = [str(runtime), "-Wl,-rpath,@loader_path/../torch/lib"]
            # A non-wheel installation can use an external runtime. Preserve its
            # existing location rather than introducing a different library.
            if runtime.parent != (torch_dir / "lib").resolve():
                link_flags.append(f"-Wl,-rpath,{runtime.parent}")
        else:
            includes = [explicit_include, None, torch_dir / "include"]
            compile_options = [["-fopenmp"]]
            bundled = sorted((torch_dir / "lib").glob("libgomp*.so*"))
            if bundled:
                link_flags = [str(bundled[0]), "-Wl,-rpath,$ORIGIN/../torch/lib"]
            else:
                link_flags = ["-fopenmp"]

        last_error = "no usable OpenMP compiler/runtime combination"
        attempted = set()
        for include in includes:
            if include is not None and not (include / "omp.h").is_file():
                continue
            for option in compile_options:
                flags = option + ([f"-I{include}"] if include is not None else [])
                if tuple(flags) in attempted:
                    continue
                attempted.add(tuple(flags))
                try:
                    self.probe_openmp(flags, link_flags)
                    return flags, link_flags
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                    last_error = str(error)
        raise RuntimeError(last_error)

    def build_extensions(self):
        self.target_flags = macos_target_flags() if sys.platform == "darwin" else []
        if self.target_flags:
            # Unlike setuptools' compiler/linker, PyTorch's Ninja writer does not
            # strip inherited -arch options when extra arguments select one arch.
            self.compiler.compiler_so = without_macos_arch_flags(self.compiler.compiler_so)
        setting = os.getenv("FORCE_OPENMP")
        if setting not in (None, "0", "1"):
            raise RuntimeError(
                "FORCE_OPENMP must be 0 (disabled) or 1 (required); unset means auto"
            )
        backend = aten_parallel_backend()
        compile_flags, link_flags = [], []
        if backend != "openmp":
            if setting == "1":
                raise RuntimeError(
                    f"FORCE_OPENMP=1 requires an OpenMP PyTorch build; found {backend}"
                )
            detail = "using the installed PyTorch backend"
        elif setting == "0":
            backend = "serial"
            detail = "FORCE_OPENMP=0"
            if self.compiler.compiler_type == "unix":
                compile_flags = ["-fno-openmp"]
        else:
            try:
                compile_flags, link_flags = self.openmp_flags()
                detail = "compiler and matching runtime link check passed"
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                if setting == "1":
                    raise RuntimeError(
                        "FORCE_OPENMP=1 could not enable OpenMP. Provide compatible omp.h via "
                        "OMP_PREFIX/include or install your compiler's OpenMP development files.\n"
                        f"{error}"
                    ) from error
                backend = "serial"
                if self.compiler.compiler_type == "unix":
                    compile_flags = ["-fno-openmp"]
                detail = (
                    f"OpenMP unavailable: {error}. Set OMP_PREFIX to its header prefix; "
                    "FORCE_OPENMP=1 makes this an error"
                )
        print(f"torch-ms-deform-attn CPU parallel backend: {backend} ({detail})", flush=True)
        for extension in self.extensions:
            extension.extra_compile_args["cxx"].extend(self.target_flags + compile_flags)
            extension.extra_link_args.extend(self.target_flags + link_flags)
        # Distutils does not track changed compiler flags. Rebuild when switching
        # auto/on/off so an existing serial object cannot survive an OpenMP build.
        self.force = True
        super().build_extensions()


force_cpu = os.getenv("FORCE_CPU", "0") == "1"
force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
if force_cpu and force_cuda:
    raise RuntimeError("FORCE_CPU and FORCE_CUDA cannot both be enabled")
if force_cuda and (CUDA_HOME is None or torch.version.cuda is None):
    raise RuntimeError("FORCE_CUDA requires a CUDA-enabled PyTorch and CUDA toolkit")
with_cuda = (
    not force_cpu
    and CUDA_HOME is not None
    and torch.version.cuda is not None
    and (force_cuda or torch.cuda.is_available())
)
sources = ["csrc/vision.cpp", "csrc/ms_deform_attn_cpu.cpp"]
compile_args = {"cxx": ["-O3"]}
if with_cuda:
    sources.append("csrc/cuda/ms_deform_attn_cuda.cu")
    compile_args["nvcc"] = ["-O3"]

setup(
    ext_modules=[
        (CUDAExtension if with_cuda else CppExtension)(
            "torch_ms_deform_attn._C",
            sources=sources,
            # Ninja compiles from its build directory, including sdist builds.
            include_dirs=[str(Path(__file__).resolve().parent)],
            define_macros=[("WITH_CUDA", None)] if with_cuda else [],
            extra_compile_args=compile_args,
        )
    ],
    cmdclass={"build_ext": OptionalOpenMPBuildExtension, "bdist_wheel": TorchBdistWheel},
)

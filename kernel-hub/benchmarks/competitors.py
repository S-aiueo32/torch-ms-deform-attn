"""Pinned upstream sources; MMCV builds only its unmodified MSDA CUDA source."""

import hashlib
import importlib
import io
import json
import sys
import types
import urllib.request
import zipfile
from pathlib import Path

PINS = {
    "mmcv": ("open-mmlab/mmcv", "a8073c74bf83d62ec36a103f835faa4837fb6585"),
    "msda-triton": ("rziga/msda-triton", "90cda22eb5bb17db93ddfcc7feb3903d29acf6da"),
    "upstream-before-perf": (
        "S-aiueo32/torch-ms-deform-attn",
        "e07889a886a9e3052ffc10d019ac5d88ba3f40f6",
    ),
}


def prepare(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (repo, revision) in PINS.items():
        url = f"https://codeload.github.com/{repo}/zip/{revision}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        files = {}
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.infolist():
                relative = Path(*Path(member.filename).parts[1:])
                if member.is_dir() or not relative.parts:
                    continue
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe source archive path")
                if not (
                    str(relative).startswith("mmcv/ops/csrc/")
                    or str(relative).startswith("src/msda_triton/")
                    or relative.name in ("LICENSE", "LICENSES", "NOTICE")
                    or (name == "msda-triton" and str(relative) in ("pyproject.toml", "README.md"))
                    or (
                        name == "upstream-before-perf"
                        and str(relative).startswith(("csrc/cuda/", "src/torch_ms_deform_attn/"))
                    )
                ):
                    continue
                content = archive.read(member)
                target = destination / name / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                files[str(relative)] = hashlib.sha256(content).hexdigest()
        manifest[name] = {"repo": repo, "revision": revision, "source_sha256": files}
    (destination / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_mmcv(root):
    from torch.utils.cpp_extension import load

    root = Path(root)
    source = root / "mmcv/mmcv/ops/csrc"
    binding = root / "mmcv_binding.cpp"
    binding.write_text("""#include <torch/extension.h>
at::Tensor ms_deform_attn_cuda_forward(const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, int);
void ms_deform_attn_cuda_backward(const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    at::Tensor&, at::Tensor&, at::Tensor&, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ms_deform_attn_cuda_forward);
    m.def("backward", &ms_deform_attn_cuda_backward);
}
""")
    return load(
        name="mmcv_msda_benchmark",
        sources=[str(binding), str(source / "pytorch/cuda/ms_deform_attn_cuda.cu")],
        extra_include_paths=[str(source / "common"), str(source / "common/cuda")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


def load_previous(root, native=None):
    """Load the pinned original kernels/API under an isolated test namespace."""
    root = Path(root) / "upstream-before-perf"
    if native is None:
        from torch.utils.cpp_extension import load

        binding = root / "binding.cpp"
        binding.write_text("""#include <torch/extension.h>
#include "cuda/ms_deform_attn_cuda.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ms_deform_attn_forward", &ms_deform_attn_cuda_forward);
    m.def("ms_deform_attn_backward", &ms_deform_attn_cuda_backward);
}
""")
        native = load(
            name="msda_before_perf_native",
            sources=[str(binding), str(root / "csrc/cuda/ms_deform_attn_cuda.cu")],
            extra_include_paths=[str(root / "csrc")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=True,
        )
    namespace = "msda_before_perf"
    package_path = root / "src/torch_ms_deform_attn"
    package = types.ModuleType(namespace)
    package.__path__ = [str(package_path)]
    package._C = native
    sys.modules[namespace] = package
    sys.modules[f"{namespace}._C"] = native
    source_path = package_path / "_ops.py"
    source = source_path.read_text()
    if source.count('"torch_ms_deform_attn::') != 2:
        raise ValueError("Pinned registration namespace contract changed")
    source = source.replace('"torch_ms_deform_attn::', f'"{namespace}::')
    registrations = types.ModuleType(f"{namespace}._ops")
    registrations.__package__ = namespace
    registrations.__file__ = str(source_path)
    sys.modules[registrations.__name__] = registrations
    exec(compile(source, str(source_path), "exec"), registrations.__dict__)
    return importlib.import_module(f"{namespace}.functional").ms_deform_attn


if __name__ == "__main__":
    prepare(sys.argv[1])

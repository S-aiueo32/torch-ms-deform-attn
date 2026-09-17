"""Pinned upstream sources; MMCV builds only its unmodified MSDA CUDA source."""

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

PINS = {
    "mmcv": ("open-mmlab/mmcv", "a8073c74bf83d62ec36a103f835faa4837fb6585"),
    "msda-triton": ("rziga/msda-triton", "90cda22eb5bb17db93ddfcc7feb3903d29acf6da"),
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


if __name__ == "__main__":
    prepare(sys.argv[1])

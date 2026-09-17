#!/usr/bin/env python3
"""Generate/build diagnostic CUDA variants without editing the project.

Run with --build on the CUDA host. Each extension retains the original native
entry point and numerical policy; only the header transformations below vary.
These are experiments, not general-purpose safe replacements for the operator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

FILES = (
    "cuda/ms_deform_attn_cuda.cu",
    "cuda/ms_deform_attn_cuda.h",
    "cuda/ms_deform_im2col_cuda.cuh",
    "cuda/index_utils.h",
)
DEFAULT_VARIANTS = "baseline,int32,no_metadata,int32_no_metadata"
SUPPORTED = (
    "baseline",
    "int32",
    "no_metadata",
    "int32_no_metadata",
    "no_bounds",
    "fwd256",
    "fwd128",
    "int32_no_metadata_fwd256",
)
BINDING = r"""
#include <torch/extension.h>
#include "cuda/ms_deform_attn_cuda.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ms_deform_attn_cuda_forward);
  m.def("backward", &ms_deform_attn_cuda_backward);
  m.def("ms_deform_attn_forward", &ms_deform_attn_cuda_forward);
  m.def("ms_deform_attn_backward", &ms_deform_attn_cuda_backward);
}
"""


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence: {old!r}")
    return text.replace(old, new)


def int32_indices(text):
    """Narrow arithmetic while preserving int64 metadata loads and validation."""
    valid_match = re.search(r"__device__ inline bool valid_spatial_level\(.*?\n}\n", text, re.S)
    if valid_match is None:
        raise ValueError("Cannot locate metadata validation function")
    valid_function = valid_match.group(0)
    text = text.replace(valid_function, "/* PROTECTED_VALID_FUNCTION */\n")
    metadata_pattern = re.compile(
        r"      const int64_t level_start_id = data_level_start_index\[l_col\];\n"
        r"      const int64_t spatial_h_ptr = static_cast<int64_t>\(l_col\) \* 2;\n"
        r"      const int64_t height = data_spatial_shapes\[spatial_h_ptr\];\n"
        r"      const int64_t width = data_spatial_shapes\[spatial_h_ptr \+ 1\];\n"
        r"      if \(!valid_spatial_level\(height, width, level_start_id, spatial_size\)\)\n"
        r"        return;\n"
        r"      const int spatial_h = static_cast<int>\(height\);\n"
        r"      const int spatial_w = static_cast<int>\(width\);"
    )
    text, metadata_blocks = metadata_pattern.subn("/* PROTECTED_METADATA_BLOCK */", text)
    if metadata_blocks != 7:
        raise ValueError(f"Expected 7 metadata blocks, got {metadata_blocks}")
    text, metadata_pointers = re.subn(r"const int64_t \*", "const PROTECTED_METADATA_TYPE *", text)
    if metadata_pointers != 18:
        raise ValueError(f"Expected 18 metadata pointers, got {metadata_pointers}")
    replacements = len(re.findall(r"\bint64_t\b", text))
    text = re.sub(r"\bint64_t\b", "int", text)
    text = text.replace("PROTECTED_METADATA_TYPE", "int64_t")
    text = text.replace("/* PROTECTED_VALID_FUNCTION */\n", valid_function)
    metadata_block = """      const int64_t level_start_id_raw = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col * 2;
      const int64_t height_raw = data_spatial_shapes[spatial_h_ptr];
      const int64_t width_raw = data_spatial_shapes[spatial_h_ptr + 1];
      if (!valid_spatial_level(height_raw, width_raw, level_start_id_raw, spatial_size))
        return;
      const int level_start_id = static_cast<int>(level_start_id_raw);
      const int spatial_h = static_cast<int>(height_raw);
      const int spatial_w = static_cast<int>(width_raw);"""
    text = text.replace("/* PROTECTED_METADATA_BLOCK */", metadata_block)
    text = replace_once(
        text,
        "  if (num_kernels <= std::numeric_limits<int>::max()) {\n"
        "    launch(int{});\n  } else {\n    launch(int{});\n  }",
        "  launch(int{});  // Diagnostic: input bounds validated by harness.",
    )
    text = text.replace(
        "    // The launch selects int only when every output index fits. Pointer\n"
        "    // offsets below remain 64-bit even on this faster decomposition path.",
        "    // Diagnostic: all index arithmetic fits int32 for the benchmark inputs.",
    )
    audit_text = text.replace(valid_function, "")
    audit_text = re.sub(r"/\*.*?\*/|//[^\n]*", "", audit_text, flags=re.S)
    remaining = [line.strip() for line in audit_text.splitlines() if "int64_t" in line]
    unexpected = [
        line
        for line in remaining
        if not re.search(r"const int64_t \*data_(?:spatial_shapes|level_start_index)", line)
        and not re.search(r"const int64_t (?:height|width|level_start_id)_raw =", line)
    ]
    if unexpected:
        raise ValueError(f"Unaccounted-for int64 arithmetic: {unexpected}")
    return text, {
        "index_type_occurrences_narrowed": replacements,
        "metadata_blocks_preserved": metadata_blocks,
        "int64_metadata_pointer_parameters_preserved": metadata_pointers,
        "remaining_int64_lines_excluding_metadata_check": remaining,
    }


def transform(text, variant):
    changes = []
    audit = {}
    if variant.startswith("int32"):
        text, audit = int32_indices(text)
        changes.append(
            "Narrow all header loop, work-count, decomposition, stride and bilinear "
            "offset arithmetic to int32; retain int64 metadata loads/check and native host wrapper."
        )
    if "no_metadata" in variant:
        text, count = re.subn(
            r"      if \(!valid_spatial_level\([^\n]*\)\)\n        return;\n",
            "",
            text,
        )
        if count != 7:
            raise ValueError(f"Expected 7 metadata validation calls, got {count}")
        audit["metadata_calls_removed"] = count
        changes.append("Remove per-thread, per-level metadata assertions and return guards.")
    if variant == "no_bounds":
        text, count = re.subn(r"__launch_bounds__\([^)]*\)", "", text)
        if count != 7:
            raise ValueError(f"Expected 7 launch bounds, got {count}")
        audit["launch_bounds_removed"] = count
        changes.append("Remove launch bounds from all kernels; block dimensions unchanged.")
    if "fwd" in variant:
        threads = 128 if variant.endswith("128") else 256
        text = replace_once(
            text,
            "template <typename scalar_t, typename index_t>\n"
            "__global__ void __launch_bounds__(CUDA_NUM_THREADS)",
            "template <typename scalar_t, typename index_t>\n"
            f"__global__ void __launch_bounds__({threads})",
        )
        text = replace_once(
            text,
            "  const int num_threads = CUDA_NUM_THREADS;",
            f"  const int num_threads = {threads};",
        )
        changes.append(
            f"Use {threads} forward threads and matching forward launch bound; backward unchanged."
        )
    if variant == "baseline":
        changes.append(
            "Unmodified project CUDA sources; minimal pybind wrapper instead of package dispatcher."
        )
    return text, changes, audit


def build_one(config_path):
    import torch
    from torch.utils.cpp_extension import load

    config = json.loads(Path(config_path).read_text())
    os.environ.setdefault("MAX_JOBS", "2")
    started = time.time()
    extension = load(
        name=config["module_name"],
        sources=config["sources"],
        extra_include_paths=config["include_paths"],
        extra_cflags=config["cxx_flags"],
        extra_cuda_cflags=config["cuda_flags"],
        build_directory=config["build_directory"],
        verbose=True,
        with_cuda=True,
    )
    result = {
        "path": str(Path(extension.__file__).resolve()),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "build_seconds": time.time() - started,
    }
    Path(config["result_path"]).write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("variants"))
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-one", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.build_one:
        build_one(args.build_one)
        return
    repo = args.repo.resolve()
    output = args.output.resolve()
    variants = args.variants.split(",")
    if not variants or any(name not in SUPPORTED for name in variants):
        parser.error(f"Variants must be chosen from {SUPPORTED}")
    original = {name: (repo / "csrc" / name).read_text() for name in FILES}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "diagnostic_only": True,
        "repository": str(repo),
        "original_source_hashes": {name: digest(text) for name, text in original.items()},
        "safety_assumptions": [
            "Harness verifies valid spatial shapes and offsets before calling any variant.",
            "Harness verifies positive dimensions and each tensor numel and all index products <= INT32_MAX / 2.",
            "Half-range headroom covers out-of-image bilinear neighbor offsets and rounded grid increment.",
            "Inputs, outputs and metadata remain identical; metadata storage stays torch.int64.",
            "Native wrapper is unmodified, including allocations, device guard, host validation and dtype dispatch.",
            "No fast-math flags are used; float32 and float64 compilation policy is preserved.",
            "no_bounds can fail on 1024-thread launches if register allocation exceeds hardware resources.",
        ],
        "variants": [],
    }
    failed = False
    for name in variants:
        root = output / name
        root.mkdir(exist_ok=True)
        build_dir = root / "build"
        build_dir.mkdir(exist_ok=True)
        generated = dict(original)
        header = "cuda/ms_deform_im2col_cuda.cuh"
        generated[header], changes, audit = transform(original[header], name)
        generated["bindings.cpp"] = BINDING
        for relative, source in generated.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
        module_name = "msda_diagnostic_" + name
        config = {
            "module_name": module_name,
            "sources": [str(root / "bindings.cpp"), str(root / "cuda/ms_deform_attn_cuda.cu")],
            "include_paths": [str(root)],
            "cxx_flags": ["-O3"],
            "cuda_flags": ["-O3", "-lineinfo", "--ptxas-options=-v"],
            "build_directory": str(build_dir),
            "result_path": str(root / "build_result.json"),
        }
        config_path = root / "build_config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        entry = {
            "name": name,
            "module": module_name,
            "module_name": module_name,
            "forward": "forward",
            "backward": "backward",
            "changes": changes,
            "source_hashes": {relative: digest(source) for relative, source in generated.items()},
            "transformation_audit": audit,
            "build_config": str(config_path),
            "build_log": str(root / "build.log"),
            "status": "generated",
        }
        if args.build:
            print(f"Building {name}", flush=True)
            with (root / "build.log").open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--build-one",
                        str(config_path),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode == 0:
                entry.update(json.loads(Path(config["result_path"]).read_text()))
                entry["status"] = "built"
            else:
                entry["status"] = "failed"
                entry["returncode"] = result.returncode
                failed = True
                print((root / "build.log").read_text()[-12000:], file=sys.stderr)
            log_text = (root / "build.log").read_text()
            summary = "\n".join(
                line for line in log_text.splitlines() if "ptxas" in line or "spill" in line
            )
            (root / "ptxas_summary.txt").write_text(summary + "\n")
        manifest["variants"].append(entry)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(str(output / "manifest.json"), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Run inside a matching CUDA development container (see installation matrix).
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    printf 'Usage: bash scripts/run_cuda_checks.sh {none|memcheck|racecheck|synccheck|initcheck|all} OUTPUT_DIR [benchmark]\n' >&2
    exit 2
fi

cuda_checks_benchmark=${3:-}
if [[ -n "$cuda_checks_benchmark" && "$cuda_checks_benchmark" != benchmark ]]; then
    printf 'Unsupported workload: %s\n' "$cuda_checks_benchmark" >&2
    exit 2
fi

cuda_checks_sanitizer=$1
case "$cuda_checks_sanitizer" in
    none|memcheck|racecheck|synccheck|initcheck|all) ;;
    *) printf 'Unsupported sanitizer: %s\n' "$cuda_checks_sanitizer" >&2; exit 2 ;;
esac

if [[ ! -f pyproject.toml || ! -d csrc || ! -d tests ]]; then
    printf 'Run this script from the repository root.\n' >&2
    exit 2
fi

cuda_checks_source=$(pwd -P)
mkdir -p -- "$2"
cuda_checks_output=$(cd -- "$2" && pwd -P)
cuda_checks_temp=$(mktemp -d "${TMPDIR:-/tmp}/torch-ms-deform-attn-cuda.XXXXXXXX")
trap 'rm -rf -- "$cuda_checks_temp"' EXIT

run_checks() {
    unset PYTHONPATH PYTHONHOME
    export FORCE_CPU=0 FORCE_CUDA=1 FORCE_OPENMP=1 MAX_JOBS=2
    export PYTHONUNBUFFERED=1

    export CUDA_CHECKS_TORCH_VERSION=${CUDA_CHECKS_TORCH_VERSION:-2.5.1}
    case "$CUDA_CHECKS_TORCH_VERSION" in
        2.5.1) export CUDA_CHECKS_TOOLKIT=12.4; cuda_checks_index=cu124 ;;
        2.7.1) export CUDA_CHECKS_TOOLKIT=12.6; cuda_checks_index=cu126 ;;
        *) echo 'Unsupported validation PyTorch version' >&2; exit 2 ;;
    esac
    local cuda_checks_base_python=${CUDA_CHECKS_PYTHON:-python3.11}
    "$cuda_checks_base_python" --version
    export CUDA_CHECKS_SOURCE_SHA=${CUDA_CHECKS_SOURCE_SHA:-$(git rev-parse HEAD)}
    export CUDA_CHECKS_SANITIZER="$cuda_checks_sanitizer"
    printf 'Source SHA: %s\n' "$CUDA_CHECKS_SOURCE_SHA"
    nvidia-smi > "$cuda_checks_output/nvidia-smi.txt"
    nvcc --version
    if [[ "$cuda_checks_sanitizer" != none ]]; then
        compute-sanitizer --version
    fi

    "$cuda_checks_base_python" -m venv "$cuda_checks_temp/venv"
    export PATH="$cuda_checks_temp/venv/bin:$PATH"
    local cuda_checks_python="$cuda_checks_temp/venv/bin/python"
    "$cuda_checks_python" -m pip install --no-cache-dir \
        numpy 'setuptools>=77' 'packaging>=24.2' wheel ninja build
    "$cuda_checks_python" -m pip install --no-cache-dir "torch==$CUDA_CHECKS_TORCH_VERSION" \
        --index-url "https://download.pytorch.org/whl/$cuda_checks_index"
    "$cuda_checks_python" - <<'PY'
import os
import torch

assert torch.__version__.split("+")[0] == os.environ["CUDA_CHECKS_TORCH_VERSION"], torch.__version__
assert torch.version.cuda == os.environ["CUDA_CHECKS_TOOLKIT"], torch.version.cuda
assert torch.cuda.is_available(), "CUDA validation requires a visible GPU"
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
for device in range(torch.cuda.device_count()):
    print("GPU", device, torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device))
PY

    # Dedicated distribution paths prevent an earlier wheel from being tested.
    "$cuda_checks_python" -m build --sdist --no-isolation \
        --outdir "$cuda_checks_temp/dist" "$cuda_checks_source"
    local -a cuda_checks_sdists=("$cuda_checks_temp"/dist/*.tar.gz)
    [[ ${#cuda_checks_sdists[@]} -eq 1 && -f "${cuda_checks_sdists[0]}" ]]
    cp -- "${cuda_checks_sdists[0]}" "$cuda_checks_output/"
    "$cuda_checks_python" -m pip wheel --no-build-isolation --no-deps \
        "${cuda_checks_sdists[0]}" -w "$cuda_checks_temp/dist"
    local -a cuda_checks_wheels=("$cuda_checks_temp"/dist/*.whl)
    [[ ${#cuda_checks_wheels[@]} -eq 1 && -f "${cuda_checks_wheels[0]}" ]]
    cp -- "${cuda_checks_wheels[0]}" "$cuda_checks_output/"
    "$cuda_checks_python" -m pip install --no-deps "${cuda_checks_wheels[0]}"

    cp -R -- "$cuda_checks_source/tests" "$cuda_checks_temp/tests"
    cd -- "$cuda_checks_temp"
    "$cuda_checks_python" - <<'PY'
from pathlib import Path
import sys
from torch_ms_deform_attn import _C

assert _C.with_cuda, "The installed wheel has no CUDA support"
assert _C.cpu_parallel_backend == "openmp", _C.cpu_parallel_backend
assert Path(_C.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), _C.__file__
print("Installed extension:", _C.__file__)
PY
    cp -- "$cuda_checks_source/scripts/cuda_test_report.py" "$cuda_checks_temp/"
    "$cuda_checks_python" cuda_test_report.py --output "$cuda_checks_output/cuda-tests.json"

    if [[ "$cuda_checks_sanitizer" != none ]]; then
        cd -- "$cuda_checks_temp/tests"
        local -a cuda_checks_tools=("$cuda_checks_sanitizer")
        if [[ "$cuda_checks_sanitizer" == all ]]; then
            cuda_checks_tools=(memcheck racecheck synccheck initcheck)
        fi
        for cuda_checks_tool in "${cuda_checks_tools[@]}"; do
            # Explicit positive cases exclude device-assert subprocess tests.
            # T03 expands reduction coverage; T04 supplies sampling cases.
            compute-sanitizer --tool "$cuda_checks_tool" \
                --error-exitcode 1 --target-processes all \
                "$cuda_checks_python" -m unittest -v \
                test_cuda.CUDAAttentionTest.test_reference_forward_backward \
                test_cuda.CUDAAttentionTest.test_partial_batch_chunk \
                test_cuda.CUDAAttentionTest.test_padding_support_and_channel_sizes \
                test_cuda.CUDAAttentionTest.test_offsets_collisions_and_noncontiguous_metadata \
                2>&1 | tee "$cuda_checks_output/sanitizer-$cuda_checks_tool.log"
        done
    fi
    if [[ "$cuda_checks_benchmark" == benchmark ]]; then
        cp -- "$cuda_checks_source/benchmarks/benchmark_cuda.py" "$cuda_checks_temp/benchmark_cuda.py"
        cd -- "$cuda_checks_temp"
        nvidia-smi > "$cuda_checks_output/nvidia-smi.txt"
        "$cuda_checks_python" benchmark_cuda.py > "$cuda_checks_output/benchmark-cuda.json"
    fi
    "$cuda_checks_python" - "$cuda_checks_output/cuda-completion.json" <<'PY'
import json, os, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "source_sha": os.environ["CUDA_CHECKS_SOURCE_SHA"],
    "sanitizer": os.environ["CUDA_CHECKS_SANITIZER"], "success": True,
}, indent=2) + "\n")
PY
    printf 'CUDA validation completed successfully.\n' 
}

# A bare pipeline preserves errexit inside run_checks and reports tee failures.
run_checks 2>&1 | tee "$cuda_checks_output/cuda-checks.log"

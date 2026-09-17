#!/usr/bin/env bash
# Disposable CUDA 12.6 host workload, run by the ownership-aware Runpod controller.
set -euo pipefail
[[ $# -ge 1 && $# -le 2 && -n "${CUDA_CHECKS_SOURCE_SHA:-}" ]]
msda_suite_args=()
case "${2:-full}" in
    full) ;;
    compile-amp) msda_suite_args=(--focus-compile) ;;
    phase1) msda_suite_args=(--phase1-only) ;;
    benchmark) ;;
    *) echo 'Unknown Kernel Hub suite' >&2; exit 2 ;;
esac
msda_source=$(pwd -P)
mkdir -p "$1"
msda_output=$(cd "$1" && pwd -P)
msda_work=$(mktemp -d /workspace/ci/kernel-hub.XXXXXXXX)
exec > >(tee "$msda_output/kernel-hub-checks.log") 2>&1
export PYTHONUNBUFFERED=1 MAX_JOBS=2 CARGO_BUILD_JOBS=2
export MSDA_HF_BASELINE_REVISION=abfd4042216fa4f84c9c5c4e3e844a3143c70ad5

nvidia-smi > "$msda_output/nvidia-smi.txt"
nvcc --version
python3 -m venv "$msda_work/tools"
"$msda_work/tools/bin/pip" install uv==0.12.5
"$msda_work/tools/bin/uv" venv --python 3.11 --seed "$msda_work/venv"
export PATH="$msda_work/venv/bin:$PATH"
python -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir -r kernel-hub/e2e/requirements.txt \
    'setuptools>=77' wheel ninja 'cmake>=3.26' numpy packaging 'pytest>=8,<10'
python -c 'import torch; assert torch.cuda.is_available(); assert torch.version.cuda == "12.6"; print(torch.__version__, torch.cuda.get_device_name())'

# Configuration generation runs on the CPU controller before GPU rental.
# Only CUDA compilation and execution need the GPU host.
cp -R "$msda_source/kernel-hub-prepared" "$msda_work/candidate"
cp "$msda_work/candidate/UPSTREAM.json" "$msda_output/UPSTREAM.json"
cmake -S "$msda_work/candidate" -B "$msda_work/cmake" -G Ninja \
    -DPython3_EXECUTABLE="$msda_work/venv/bin/python" -DCMAKE_BUILD_TYPE=Release
cmake --build "$msda_work/cmake" --parallel "$MAX_JOBS"
cmake --build "$msda_work/cmake" --target local_install
python -m pip freeze > "$msda_output/python-packages.txt"
export MSDA_KERNEL_DIR="$msda_work/candidate"
export MSDA_OUTPUT_DIR="$msda_output"
if [[ "${2:-full}" == benchmark ]]; then
    FORCE_CUDA=1 python -m pip install --no-build-isolation --no-deps .
    python -m pip install --no-deps "$msda_work/candidate/benchmark-sources/msda-triton"
    cp "$msda_work/candidate/benchmark-sources/sources.json" "$msda_output/benchmark-sources.json"
    python -m pip freeze > "$msda_output/python-packages.txt"
    python "$msda_source/kernel-hub/e2e/gpu_suite.py" --phase1-only
    python -m pytest tests/test_cuda.py -v
    python -m pytest tests/test_cuda_graphs.py -v
    python kernel-hub/benchmarks/benchmark.py \
        --sources "$msda_work/candidate/benchmark-sources" \
        --cases decoder encoder --dtypes float32 float16 --modes forward forward_backward \
        --backends torch-ms-deform-attn kernel-hub-adapter hf-native upstream-before-perf \
        --output "$msda_output/benchmark-kernel-hub.json"
    exit 0
fi
python "$msda_source/kernel-hub/e2e/gpu_suite.py" "${msda_suite_args[@]}"

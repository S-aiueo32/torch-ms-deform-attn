#!/usr/bin/env bash
# Disposable CUDA 12.6 host workload, run by the ownership-aware Runpod controller.
set -euo pipefail
[[ $# -eq 1 && -n "${CUDA_CHECKS_SOURCE_SHA:-}" ]]
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
    'setuptools>=77' wheel ninja 'cmake>=3.26' numpy packaging
python -c 'import torch; assert torch.cuda.is_available(); assert torch.version.cuda == "12.6"; print(torch.__version__, torch.cuda.get_device_name())'

# Official kernel-builder local development route: generate CMake/setup.py,
# then build_kernel. This tests real builder glue without claiming Nix portability.
curl --fail --location --proto '=https' https://sh.rustup.rs -o "$msda_work/rustup.sh"
sh "$msda_work/rustup.sh" -y --profile minimal --default-toolchain 1.94.0 --no-modify-path
export PATH="$HOME/.cargo/bin:$PATH"
cargo install hf-kernel-builder --version 0.16.0 --locked
kernel-builder --version
python kernel-hub/export.py "$msda_work/candidate" --revision "$CUDA_CHECKS_SOURCE_SHA"
cp "$msda_work/candidate/UPSTREAM.json" "$msda_output/UPSTREAM.json"
kernel-builder create-pyproject "$msda_work/candidate" --unique-id "$CUDA_CHECKS_SOURCE_SHA"
export CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=89'
(
    cd "$msda_work/candidate"
    python setup.py build_kernel
)
python -m pip freeze > "$msda_output/python-packages.txt"
export MSDA_KERNEL_DIR="$msda_work/candidate"
export MSDA_OUTPUT_DIR="$msda_output"
python "$msda_source/kernel-hub/e2e/gpu_suite.py"

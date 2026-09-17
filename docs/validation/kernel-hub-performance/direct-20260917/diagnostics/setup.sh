#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ci/source
exec > >(tee /workspace/ci/results/setup.log) 2>&1
export PYTHONUNBUFFERED=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9 CUDA_HOME=/usr/local/cuda
nvidia-smi > /workspace/ci/results/nvidia-smi.txt
nvcc --version
python3 -m venv /workspace/ci/profile-env
/workspace/ci/profile-env/bin/pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
/workspace/ci/profile-env/bin/pip install --no-cache-dir kernels==0.16.0 'setuptools>=77' wheel ninja numpy packaging
export PATH=/workspace/ci/profile-env/bin:$PATH
FORCE_CUDA=1 python -m pip install --no-build-isolation --no-deps .
python -m pip freeze > /workspace/ci/results/python-packages.txt
python -c 'import torch; from torch_ms_deform_attn import _C; print(torch.__version__, torch.cuda.get_device_name(), _C.__file__)'
printf 'SETUP READY\n'

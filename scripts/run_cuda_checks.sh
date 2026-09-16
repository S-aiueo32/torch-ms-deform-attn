#!/usr/bin/env bash
set -euo pipefail
runner_source=$(pwd -P)
mkdir -p "$2"
runner_output=$(cd "$2" && pwd -P)
export GROUNDINGDINO_SHA=a08a5881b994c763925ea734139399e543fa8bf4
export PYTHONUNBUFFERED=1
export FORCE_CUDA=1 MAX_JOBS=2
run_checks() {
  nvidia-smi > "$runner_output/nvidia-smi.txt"
  nvcc --version
  python3.11 -m venv --system-site-packages /workspace/ci/gd-venv
  export PATH=/workspace/ci/gd-venv/bin:$PATH
  python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
  printf 'torch==%s\n' "$CUDA_CHECKS_TORCH_VERSION" > /workspace/ci/constraints.txt
  export PIP_CONSTRAINT=/workspace/ci/constraints.txt
  python -m pip install 'setuptools>=77' 'packaging>=24.2' wheel ninja build
  python -m pip wheel --no-build-isolation --no-deps torch-ms-deform-attn -w "$runner_output"
  python -m pip install --no-deps "$runner_output"/torch_ms_deform_attn-*.whl
  python -c 'from torch_ms_deform_attn import _C; assert _C.with_cuda; print(_C.__file__)'
  git clone https://github.com/S-aiueo32/GroundingDINO.git /workspace/ci/GroundingDINO
  cd /workspace/ci/GroundingDINO
  git checkout --detach "$GROUNDINGDINO_SHA"
  python -m pip wheel --no-build-isolation --no-deps . -w "$runner_output"
  python -m unittest discover -s tests -v
  python "$runner_source/scripts/groundingdino_gpu_checks.py" "$PWD" "$runner_output/groundingdino-cuda.json"
  compute-sanitizer --tool memcheck --error-exitcode 1 python -m unittest discover -s tests -p test_ms_deform_attn.py -v 2>&1 | tee "$runner_output/memcheck.log"
  python -m pip install "$runner_output"/groundingdino-*.whl
  python -m pip freeze > "$runner_output/packages.txt"
  mkdir -p weights
  python - <<'PY'
import urllib.request
urllib.request.urlretrieve('https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth', 'weights/groundingdino_swint_ogc.pth')
PY
  if python "$runner_source/scripts/groundingdino_inference.py" "$PWD" "$runner_output/inference.json" 2>&1 | tee "$runner_output/inference.log"; then
    printf 'Default dependency inference passed.\n'
  else
    printf 'Default dependency inference failed; checking the established Transformers 4.x API.\n'
    python -m pip install 'transformers==4.44.2'
    python -m pip freeze > "$runner_output/packages-transformers4.txt"
    python "$runner_source/scripts/groundingdino_inference.py" "$PWD" "$runner_output/inference-transformers4.json" 2>&1 | tee "$runner_output/inference-transformers4.log"
  fi
  printf 'GroundingDINO CUDA validation complete for %s\n' "$GROUNDINGDINO_SHA"
}
run_checks 2>&1 | tee "$runner_output/groundingdino-validation.log"

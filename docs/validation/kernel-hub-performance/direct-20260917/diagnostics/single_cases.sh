#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ci/source
export PATH=/workspace/ci/profile-env/bin:$PATH
for msda_mode in forward forward_backward; do
  for msda_backend in baseline int32 no_metadata int32_no_metadata hf; do
    python -u diagnostics/one_case.py --backend "$msda_backend" \
      --case encoder --mode "$msda_mode" --variants-dir /workspace/ci/variants \
      --output-dir "/workspace/ci/results/single-$msda_backend-$msda_mode" \
      > "/workspace/ci/results/single-$msda_backend-$msda_mode.log" 2>&1
  done
done

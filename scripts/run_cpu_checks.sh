#!/usr/bin/env bash
# Run in a Linux build environment with Python, a C++ compiler and CPU PyTorch.
set -euo pipefail
source_dir=$(pwd -P)
mkdir -p "$1"
output_dir=$(cd "$1" && pwd -P)
work_dir=$(mktemp -d)
trap 'rm -rf -- "$work_dir"' EXIT
export FORCE_CPU=1 FORCE_CUDA=0 FORCE_OPENMP=1 MAX_JOBS=2
unset PYTHONPATH PYTHONHOME
export CUDA_CHECKS_SOURCE_SHA=${CUDA_CHECKS_SOURCE_SHA:-$(git rev-parse HEAD)}
{
    python -m pip install 'setuptools>=77' 'packaging>=24.2' wheel ninja build numpy 'pytest>=8,<10'
    python -m build --sdist --no-isolation --outdir "$work_dir/dist"
    python -m pip wheel --no-cache-dir --no-build-isolation --no-deps "$work_dir"/dist/*.tar.gz -w "$work_dir/dist"
    python -m pip install --no-deps "$work_dir"/dist/*.whl
    python -m pytest build_tests -v
    cp -R tests "$work_dir/tests"
    cd "$work_dir"
    python "$source_dir/scripts/cpu_test_report.py" --output "$output_dir/cpu-tests.json"
} 2>&1 | tee "$output_dir/cpu-checks.log"

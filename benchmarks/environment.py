"""Provenance shared by benchmark reports (no credentials are collected)."""

import os
import platform
import subprocess

import torch

from torch_ms_deform_attn import _C


def command(args):
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment(seed, warmup):
    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = command(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
    elif os.path.isfile("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as stream:
            cpu = next(
                (line.split(":", 1)[1].strip() for line in stream if line.startswith("model name")),
                cpu,
            )
    return dict(
        source_sha=os.environ.get("CUDA_CHECKS_SOURCE_SHA")
        or command(["git", "rev-parse", "HEAD"]),
        source_dirty=command(["git", "status", "--porcelain"]),
        cpu=cpu,
        torch=torch.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        seed=seed,
        warmup=warmup,
        cpu_backend=_C.cpu_parallel_backend,
        with_cuda=_C.with_cuda,
        build_flags={
            key: os.environ.get(key, "unknown")
            for key in (
                "FORCE_CPU",
                "FORCE_CUDA",
                "FORCE_OPENMP",
                "OMP_PREFIX",
                "USE_NINJA",
                "TORCH_CUDA_ARCH_LIST",
                "CXXFLAGS",
            )
        },
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        driver=command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        cuda=torch.version.cuda,
    )

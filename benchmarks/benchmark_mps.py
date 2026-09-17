"""Metal versus MPS grid_sample on Grounding DINO Swin-T attention shapes."""

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from torch_ms_deform_attn import _C, ms_deform_attn, ms_deform_attn_core_pytorch


class ObservedMemory(TorchDispatchMode):
    """Observe live tensor bytes at operator boundaries, excluding cached memory.

    PyTorch 2.4/2.5 expose no MPS peak allocator counter. This is an observed
    lower bound, not a claim to measure hidden intra-operator workspace peaks.
    """

    def __init__(self):
        super().__init__()
        self.baseline = torch.mps.current_allocated_memory()
        self.peak = self.baseline

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        self.peak = max(self.peak, torch.mps.current_allocated_memory())
        return result


def timed(fn):
    torch.mps.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.mps.synchronize()
    return (time.perf_counter() - start) * 1000, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1:
        parser.error("iterations and warmup must be positive")
    if not (torch.backends.mps.is_available() and _C.with_mps):
        parser.error("requires an MPS device and a Metal-enabled extension")
    torch.manual_seed(37)
    # First call includes lazy library compilation and pipeline creation.
    v = torch.ones(1, 1, 1, 1, device="mps", requires_grad=True)
    s = torch.tensor([[1, 1]], device="mps")
    i = torch.tensor([0], device="mps")
    loc = torch.full((1, 1, 1, 1, 1, 2), 0.5, device="mps", requires_grad=True)
    w = torch.ones(1, 1, 1, 1, 1, device="mps", requires_grad=True)
    cold_ms, _ = timed(lambda: ms_deform_attn(v, s, i, loc, w))
    rows = []
    for height, width in ((800, 800), (800, 1344)):
        sizes = [
            (math.ceil(height / stride), math.ceil(width / stride)) for stride in (8, 16, 32, 64)
        ]
        spatial = sum(h * w for h, w in sizes)
        for batch in (1, 2):
            for stage, queries in (("encoder", spatial), ("decoder", 900)):
                shapes = torch.tensor(sizes, device="mps")
                starts = torch.tensor(
                    [0, *torch.tensor(sizes).prod(1).cumsum(0)[:-1].tolist()], device="mps"
                )
                value = torch.randn(batch, spatial, 8, 32, device="mps", requires_grad=True)
                locations = torch.rand(batch, queries, 8, 4, 4, 2, device="mps", requires_grad=True)
                weights = (torch.rand(batch, queries, 8, 4, 4, device="mps") / 16).requires_grad_()
                grad = torch.randn(batch, queries, 256, device="mps")

                def metal():
                    return ms_deform_attn(value, shapes, starts, locations, weights)

                def reference():
                    # Include metadata transfer in both measured implementations.
                    return ms_deform_attn_core_pytorch(
                        value, shapes.cpu().tolist(), locations, weights
                    )

                for mode in ("forward", "forward_backward"):
                    for name, fn in (("metal", metal), ("grid_sample", reference)):

                        def run():
                            if mode == "forward":
                                with torch.no_grad():
                                    return fn()
                            return torch.autograd.grad(fn(), (value, locations, weights), grad)

                        try:
                            for _ in range(args.warmup):
                                timed(run)
                        except NotImplementedError as error:
                            if name != "grid_sample":
                                raise
                            row = dict(
                                image=[height, width],
                                batch=batch,
                                stage=stage,
                                queries=queries,
                                mode=mode,
                                backend=name,
                                unsupported=str(error),
                            )
                            rows.append(row)
                            print(json.dumps(row), flush=True)
                            continue
                        times = [timed(run)[0] for _ in range(args.iterations)]
                        gc.collect()
                        torch.mps.synchronize()
                        memory = ObservedMemory()
                        with memory:
                            result = run()
                        torch.mps.synchronize()
                        del result
                        row = dict(
                            image=[height, width],
                            batch=batch,
                            stage=stage,
                            queries=queries,
                            mode=mode,
                            backend=name,
                            median_ms=statistics.median(times),
                            min_ms=min(times),
                            max_ms=max(times),
                            observed_peak_extra_tensor_bytes=memory.peak - memory.baseline,
                        )
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                value = locations = weights = grad = None
                gc.collect()
                torch.mps.empty_cache()
    report = dict(
        torch=torch.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        gpu=[
            item.get("sppci_model", "unknown")
            for item in json.loads(
                subprocess.check_output(
                    ["system_profiler", "SPDisplaysDataType", "-json"], text=True
                )
            )["SPDisplaysDataType"]
        ],
        extension_sha256=hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        working_tree_diff=subprocess.check_output(["git", "diff", "--stat"], text=True),
        dtype="float32",
        warmup=args.warmup,
        iterations=args.iterations,
        first_forward_including_shader_compile_ms=cold_ms,
        memory_measurement="Observed live tensor bytes at operator boundaries; excludes allocator "
        "cache and may miss intra-operator workspace. Measured separately from latency.",
        results=rows,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

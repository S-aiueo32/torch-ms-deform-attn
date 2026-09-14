"""Compare CPU extension and grid_sample reference; run after installing package."""

import argparse
import json
import platform

import torch
from torch.utils.benchmark import Timer

from torch_ms_deform_attn import _C, ms_deform_attn, ms_deform_attn_core_pytorch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=0.3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.min_run_time <= 0:
        parser.error("threads and min-run-time must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rows = []
    for name, batch, queries, heads, channels, sizes in [
        ("small", 1, 100, 4, 16, [(16, 16), (8, 8)]),
        ("decoder", 1, 300, 8, 32, [(64, 64), (32, 32), (16, 16), (8, 8)]),
        ("batched", 4, 300, 8, 32, [(64, 64), (32, 32), (16, 16), (8, 8)]),
    ]:
        shapes = torch.tensor(sizes, dtype=torch.long)
        starts = torch.cat((torch.zeros(1, dtype=torch.long), shapes.prod(1).cumsum(0)[:-1]))
        value = torch.randn(
            batch, sum(h * w for h, w in sizes), heads, channels, requires_grad=True
        )
        locations = torch.rand(batch, queries, heads, len(sizes), 4, 2, requires_grad=True)
        weights = torch.rand(batch, queries, heads, len(sizes), 4, requires_grad=True)
        grad = torch.randn(batch, queries, heads * channels)

        def extension():
            return ms_deform_attn(value, shapes, starts, locations, weights)

        def reference():
            return ms_deform_attn_core_pytorch(value, shapes, locations, weights)

        # Check the same forward and backward paths that will be timed.
        actual, expected = extension(), reference()
        torch.testing.assert_close(actual, expected)
        differentiable = (value, locations, weights)
        for a, e in zip(
            torch.autograd.grad(actual, differentiable, grad),
            torch.autograd.grad(expected, differentiable, grad),
        ):
            torch.testing.assert_close(a, e, atol=2e-4, rtol=1e-4)
        for mode in ("forward", "forward_backward"):
            timings = {}
            for backend, fn in (("cpp", extension), ("pytorch", reference)):

                def run():
                    if mode == "forward":
                        with torch.no_grad():
                            return fn()
                    output = fn()
                    return torch.autograd.grad(output, (value, locations, weights), grad)

                measurement = Timer(
                    stmt="run()", globals={"run": run}, num_threads=args.threads
                ).blocked_autorange(min_run_time=args.min_run_time)
                timings[backend + "_ms"] = measurement.median * 1000
                timings[backend + "_iqr_ms"] = measurement.iqr * 1000
            rows.append(
                dict(
                    case=name,
                    mode=mode,
                    **timings,
                    speedup=timings["pytorch_ms"] / timings["cpp_ms"],
                )
            )
    report = dict(
        torch=torch.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        threads=args.threads,
        device="cpu",
        dtype="float32",
        min_run_time_seconds=args.min_run_time,
        cpu_parallel_backend=getattr(_C, "cpu_parallel_backend", "unknown"),
        results=rows,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"PyTorch {report['torch']} | {report['platform']} | threads={args.threads}")
        print(f"{'case':10} {'mode':18} {'C++ ms':>10} {'PyTorch ms':>12} {'speedup':>9}")
        for row in rows:
            print(
                f"{row['case']:10} {row['mode']:18} {row['cpp_ms']:10.3f} "
                f"{row['pytorch_ms']:12.3f} {row['speedup']:8.2f}x"
            )


if __name__ == "__main__":
    main()

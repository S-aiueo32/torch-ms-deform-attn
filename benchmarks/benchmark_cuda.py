"""Compare CUDA extension and grid_sample reference; emit a JSON report."""

import argparse
import json
import math
import platform
import sys

import torch
from torch.utils.benchmark import Timer

from torch_ms_deform_attn import _C, ms_deform_attn, ms_deform_attn_core_pytorch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=1.0)
    args = parser.parse_args()
    if args.threads < 1 or not math.isfinite(args.min_run_time) or args.min_run_time <= 0:
        parser.error("threads and min-run-time must be positive")
    if not torch.cuda.is_available() or not _C.with_cuda:
        raise RuntimeError("Benchmark requires a visible GPU and a CUDA-enabled extension")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rows = []
    validation = []
    for name, batch, queries, heads, channels, sizes in [
        ("small", 1, 100, 4, 16, [(16, 16), (8, 8)]),
        ("decoder", 1, 300, 8, 32, [(64, 64), (32, 32), (16, 16), (8, 8)]),
        ("batched", 4, 300, 8, 32, [(64, 64), (32, 32), (16, 16), (8, 8)]),
    ]:
        shapes = torch.tensor(sizes, dtype=torch.long, device="cuda")
        starts = torch.cat(
            (torch.zeros(1, dtype=torch.long, device="cuda"), shapes.prod(1).cumsum(0)[:-1])
        )
        value = torch.randn(
            batch, sum(h * w for h, w in sizes), heads, channels, device="cuda", requires_grad=True
        )
        locations = torch.rand(
            batch, queries, heads, len(sizes), 4, 2, device="cuda", requires_grad=True
        )
        weights = torch.rand(
            batch, queries, heads, len(sizes), 4, device="cuda", requires_grad=True
        )
        grad = torch.randn(batch, queries, heads * channels, device="cuda")

        def extension():
            return ms_deform_attn(value, shapes, starts, locations, weights)

        def reference():
            return ms_deform_attn_core_pytorch(value, tuple(sizes), locations, weights)

        # Check the same forward and backward paths that will be timed.
        actual, expected = extension(), reference()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
        differentiable = (value, locations, weights)
        # Locations are normalized coordinates: their gradients contain a
        # width/height factor. Compare in feature-pixel units so the absolute
        # tolerance does not become tighter as spatial resolution increases.
        coordinate_scale = shapes.flip(-1).to(value.dtype).view(1, 1, 1, len(sizes), 1, 2)
        errors = {}
        for index, (a, e) in enumerate(
            zip(
                torch.autograd.grad(actual, differentiable, grad),
                torch.autograd.grad(expected, differentiable, grad),
            )
        ):
            label = ("value", "locations", "weights")[index]
            errors[label + "_max_abs_error"] = (a - e).abs().max().item()
            if index == 1:
                a, e = a / coordinate_scale, e / coordinate_scale
                errors["locations_pixel_max_abs_error"] = (a - e).abs().max().item()
            torch.testing.assert_close(a, e, atol=2e-4, rtol=1e-4)

        # Independently verify the algorithm at higher precision outside timing,
        # rather than relying only on the float32 comparison tolerance.
        double_inputs = tuple(t.detach().double().requires_grad_() for t in differentiable)
        double_actual = ms_deform_attn(double_inputs[0], shapes, starts, *double_inputs[1:])
        double_expected = ms_deform_attn_core_pytorch(
            double_inputs[0], tuple(sizes), *double_inputs[1:]
        )
        torch.testing.assert_close(double_actual, double_expected, atol=1e-8, rtol=1e-8)
        for a, e in zip(
            torch.autograd.grad(double_actual, double_inputs, grad.double()),
            torch.autograd.grad(double_expected, double_inputs, grad.double()),
        ):
            torch.testing.assert_close(a, e, atol=1e-8, rtol=1e-8)
        validation.append(dict(case=name, float64_check="passed", **errors))
        del double_inputs, double_actual, double_expected
        for mode in ("forward", "forward_backward"):
            timings = {}
            for backend, fn in (("cuda", extension), ("pytorch", reference)):

                def run():
                    if mode == "forward":
                        with torch.no_grad():
                            return fn()
                    output = fn()
                    return torch.autograd.grad(output, (value, locations, weights), grad)

                print(f"{name} {mode} {backend}", file=sys.stderr, flush=True)
                for _ in range(5):
                    run()
                torch.cuda.synchronize()
                # Timer synchronizes accelerator work at timing boundaries.
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
                    speedup=timings["pytorch_ms"] / timings["cuda_ms"],
                )
            )
    report = dict(
        torch=torch.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        threads=args.threads,
        device="cuda",
        dtype="float32",
        gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,
        compute_capability=torch.cuda.get_device_capability(),
        reference_shape_metadata="static Python tuples",
        timing="wall time with CUDA synchronization; five warmup calls",
        min_run_time_seconds=args.min_run_time,
        gradient_atol=2e-4,
        gradient_rtol=1e-4,
        location_gradient_comparison="feature-pixel units",
        validation=validation,
        results=rows,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Validated CUDA latency, event intervals, and peak allocated memory."""

import argparse
import json
import math
import statistics
import sys

import torch
from environment import environment
from torch.utils.benchmark import Timer

from torch_ms_deform_attn import _C, ms_deform_attn, ms_deform_attn_core_pytorch

CASES = {
    "small": (100, 4, ((16, 16), (8, 8))),
    "decoder": (300, 8, ((64, 64), (32, 32), (16, 16), (8, 8))),
    "encoder": (5440, 8, ((64, 64), (32, 32), (16, 16), (8, 8))),
}


def measure(run, args):
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    wall = Timer(stmt="run()", globals={"run": run}, num_threads=args.threads).blocked_autorange(
        min_run_time=args.min_run_time
    )
    # Event intervals include stream idle gaps caused by host dispatch; they are
    # NOT the sum of kernel durations. Use a profiler for pure kernel time.
    events = []
    for _ in range(20):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        events.append(start.elapsed_time(end))
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = run()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    quartiles = statistics.quantiles(events, n=4)
    kernel = {}
    if args.profile_kernels:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            run()
            torch.cuda.synchronize()
        durations = [
            event.time_range.elapsed_us()
            for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        if not durations:
            raise RuntimeError("Profiler returned no CUDA activities")
        kernel["profiled_cuda_activity_ms"] = sum(durations) / 1000
    return dict(
        wall_ms=wall.median * 1000,
        wall_iqr_ms=wall.iqr * 1000,
        wall_sample_count=len(wall.raw_times),
        wall_calls_per_sample=wall.number_per_run,
        event_interval_ms=statistics.median(events),
        event_interval_iqr_ms=quartiles[2] - quartiles[0],
        baseline_allocated_bytes=baseline,
        peak_allocated_bytes=peak,
        incremental_peak_bytes=peak - baseline,
        **kernel,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile-kernels",
        action="store_true",
        help="Also sum profiler CUDA activities (kernels/copies), outside latency timing",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--cases", nargs="+", choices=CASES, default=["small", "decoder", "encoder"]
    )
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--channels", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--steps", nargs="+", type=int, default=[2, 64])
    parser.add_argument(
        "--dtypes", nargs="+", choices=["float32", "float16", "bfloat16"], default=["float32"]
    )
    parser.add_argument("--execution", nargs="+", choices=["eager", "compiled"], default=["eager"])
    args = parser.parse_args()
    if (
        min(args.threads, args.warmup, args.repeats, *args.batch, *args.channels, *args.steps) < 1
        or not math.isfinite(args.min_run_time)
        or args.min_run_time <= 0
    ):
        parser.error("counts and min-run-time must be positive and finite")
    if not torch.cuda.is_available() or not _C.with_cuda:
        raise RuntimeError("Benchmark requires a visible GPU and a CUDA-enabled extension")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rows = []
    import itertools

    for name, batch, channels, step, dtype_name, execution in itertools.product(
        args.cases, args.batch, args.channels, args.steps, args.dtypes, args.execution
    ):
        label = dict(
            case=name,
            batch=batch,
            channels=channels,
            im2col_step=step,
            dtype=dtype_name,
            execution=execution,
        )
        print(label, file=sys.stderr, flush=True)
        try:
            torch._dynamo.reset()
            dtype = getattr(torch, dtype_name)
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                raise RuntimeError("bf16 is unsupported by this GPU")
            queries, heads, sizes = CASES[name]
            shapes = torch.tensor(sizes, device="cuda")
            starts = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
            value = torch.randn(
                batch,
                sum(h * w for h, w in sizes),
                heads,
                channels,
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )
            loc = torch.rand(
                batch, queries, heads, len(sizes), 4, 2, device="cuda", requires_grad=True
            )
            # Stay away from interpolation knots: grid_sample's normalization can
            # round a near-knot coordinate onto the other side of its derivative jump.
            coordinate_scale = shapes.flip(-1).view(1, 1, 1, len(sizes), 1, 2)
            pixel = loc.detach() * coordinate_scale
            loc = ((pixel.floor() + 0.75 + 0.5 * pixel.frac()) / coordinate_scale).requires_grad_()
            weights = torch.rand(
                batch, queries, heads, len(sizes), 4, device="cuda", dtype=dtype, requires_grad=True
            )
            tensors = (value, loc, weights)
            grad = torch.randn(batch, queries, heads * channels, device="cuda")

            def extension(v, loc, w):
                with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                    return ms_deform_attn(v, shapes, starts, loc, w, step)

            def reference(v, loc, w):
                return ms_deform_attn_core_pytorch(v.float(), sizes, loc.float(), w.float())

            if execution == "compiled":
                extension = torch.compile(extension, fullgraph=True)
                reference_fn = torch.compile(reference, fullgraph=True)
            else:
                reference_fn = reference
            # Validate BOTH timed implementations against an eager reference.
            expected = reference(*tensors)
            expected_grads = torch.autograd.grad(expected, tensors, grad)
            scale = shapes.flip(-1).view(1, 1, 1, len(sizes), 1, 2)
            for fn in (extension, reference_fn):
                actual = fn(*tensors)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
                actual_grads = torch.autograd.grad(actual, tensors, grad)
                for index, (a, e) in enumerate(zip(actual_grads, expected_grads)):
                    if index == 1:
                        a, e = a / scale, e / scale
                    tol = (
                        dict(atol=2e-4, rtol=1e-4)
                        if dtype == torch.float32
                        else dict(atol=0.02, rtol=0.02)
                    )
                    torch.testing.assert_close(a, e, **tol)
            del expected, expected_grads, actual, actual_grads
            for repeat in range(args.repeats):
                for mode in ("forward", "backward", "forward_backward"):
                    for backend, fn in (("extension", extension), ("reference", reference_fn)):
                        output = fn(*tensors) if mode == "backward" else None

                        def run():
                            if mode == "forward":
                                with torch.no_grad():
                                    return fn(*tensors)
                            if mode == "backward":
                                return torch.autograd.grad(output, tensors, grad, retain_graph=True)
                            return torch.autograd.grad(fn(*tensors), tensors, grad)

                        timing = measure(run, args)
                        rows.append(
                            dict(
                                **label,
                                repeat=repeat,
                                mode=mode,
                                backend=backend,
                                correctness="passed",
                                **timing,
                            )
                        )
                        output = None
        except Exception as exc:
            # Drop partial measurements from a failed configuration.
            rows = [r for r in rows if any(r.get(k) != v for k, v in label.items())]
            rows.append(dict(**label, error=f"{type(exc).__name__}: {exc}"))
            print(rows[-1]["error"], file=sys.stderr, flush=True)
    print(
        json.dumps(
            dict(
                provenance=environment(args.seed, args.warmup),
                config=vars(args),
                timing="synchronized host wall time; CUDA event stream interval (not pure kernel time)",
                memory="allocator allocated bytes; backward baseline includes retained graph",
                results=rows,
            ),
            indent=2,
        )
    )
    return int(any("error" in row for row in rows))


if __name__ == "__main__":
    sys.exit(main())

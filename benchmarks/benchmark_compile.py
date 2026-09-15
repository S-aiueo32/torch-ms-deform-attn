"""Compare eager and Inductor execution; compilation is excluded from timings."""

import argparse
import json
import math
import platform
import sys

import torch
from environment import environment
from torch.utils.benchmark import Timer

from torch_ms_deform_attn import ms_deform_attn, ms_deform_attn_core_pytorch

CASES = [
    ("small", 1, 100, 4, 16, [(16, 16), (8, 8)]),
    ("decoder", 1, 300, 8, 32, [(64, 64), (32, 32), (16, 16), (8, 8)]),
    ("batched", 4, 300, 8, 32, [(64, 64), (32, 32), (16, 16), (8, 8)]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any row fails")
    args = parser.parse_args()
    if args.threads < 1 or not math.isfinite(args.min_run_time) or args.min_run_time <= 0:
        parser.error("threads and min-run-time must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rows = []
    for name, batch, queries, heads, channels, sizes in CASES:
        shapes = torch.tensor(sizes, dtype=torch.long)
        starts = torch.cat((torch.zeros(1, dtype=torch.long), shapes.prod(1).cumsum(0)[:-1]))
        value = torch.randn(
            batch, sum(h * w for h, w in sizes), heads, channels, requires_grad=True
        )
        locations = torch.rand(batch, queries, heads, len(sizes), 4, 2, requires_grad=True)
        weights = torch.rand(batch, queries, heads, len(sizes), 4, requires_grad=True)
        grad = torch.randn(batch, queries, heads * channels)
        differentiable = (value, locations, weights)

        def extension(v, loc, w):
            return ms_deform_attn(v, shapes, starts, loc, w)

        def reference(v, loc, w):
            # Static shape metadata avoids tensor-to-Python graph breaks. Use the
            # same tuple in both eager and compiled reference measurements.
            return ms_deform_attn_core_pytorch(v, tuple(sizes), loc, w)

        expected = reference(*differentiable)
        expected_grads = torch.autograd.grad(expected, differentiable, grad)
        for mode in ("forward", "forward_backward"):
            for backend, base_fn in (("cpp", extension), ("pytorch_static_shapes", reference)):
                for compile in (False, True):
                    label = f"{backend}_{'compiled' if compile else 'eager'}"
                    print(f"{name} {mode} {label}", file=sys.stderr, flush=True)
                    row = dict(case=name, mode=mode, backend=label)
                    try:
                        fn = (
                            torch.compile(base_fn, backend="inductor", fullgraph=True)
                            if compile
                            else base_fn
                        )

                        def run():
                            if mode == "forward":
                                with torch.no_grad():
                                    return fn(*differentiable)
                            return torch.autograd.grad(fn(*differentiable), differentiable, grad)

                        # Compile forward and, when applicable, backward, then
                        # validate and warm up before starting the timed region.
                        actual = run()
                        if mode == "forward":
                            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
                        else:
                            for a, e in zip(actual, expected_grads):
                                torch.testing.assert_close(a, e, atol=2e-4, rtol=1e-4)
                        run()
                        measurement = Timer(
                            stmt="run()", globals={"run": run}, num_threads=args.threads
                        ).blocked_autorange(min_run_time=args.min_run_time)
                        row.update(
                            median_ms=measurement.median * 1000, iqr_ms=measurement.iqr * 1000
                        )
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        print(row["error"], file=sys.stderr, flush=True)
                    rows.append(row)
    print(
        json.dumps(
            dict(
                provenance=environment(0, 2),
                torch=torch.__version__,
                python=platform.python_version(),
                platform=platform.platform(),
                device="cpu",
                dtype="float32",
                threads=args.threads,
                min_run_time_seconds=args.min_run_time,
                compiler="inductor",
                fullgraph=True,
                reference_shape_metadata="static Python tuples in eager and compiled",
                results=rows,
            ),
            indent=2,
        )
    )

    return 1 if args.strict and any("error" in row for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())

"""Correctness-qualified MSDA comparisons, including cast and Python overhead."""

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import torch
from competitors import load_mmcv, load_previous
from kernels import get_kernel, get_local_kernel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from benchmark_cuda import measure  # noqa: E402
from environment import cpu_scheduling, environment  # noqa: E402

from torch_ms_deform_attn import _C, ms_deform_attn, ms_deform_attn_core_pytorch  # noqa: E402

CASES = {
    "small": (1, 100, 4, 16, ((16, 16), (8, 8))),
    "decoder": (2, 300, 8, 32, ((64, 64), (32, 32), (16, 16), (8, 8))),
    "encoder": (2, 5440, 8, 32, ((64, 64), (32, 32), (16, 16), (8, 8))),
}


def extension_function(forward, backward, inplace=False):
    class Operator(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, shapes, starts, locations, weights):
            ctx.save_for_backward(value, shapes, starts, locations, weights)
            return forward(value, shapes, starts, locations, weights, 64)

        @staticmethod
        def backward(ctx, grad):
            value, shapes, starts, locations, weights = ctx.saved_tensors
            if inplace:
                gradients = [torch.zeros_like(x) for x in (value, locations, weights)]
                backward(
                    value, shapes, starts, locations, weights, grad.contiguous(), *gradients, 64
                )
            else:
                gradients = backward(
                    value, shapes, starts, locations, weights, grad.contiguous(), 64
                )
            return gradients[0], None, None, gradients[1], gradients[2]

    return Operator.apply


def matched_precision(fn):
    def run(value, shapes, starts, locations, weights):
        return fn(value.float(), shapes, starts, locations.float(), weights.float()).to(value.dtype)

    return run


def inputs(case, dtype, device="cuda"):
    batch, queries, heads, channels, sizes = CASES[case]
    shapes = torch.tensor(sizes, device=device)
    starts = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
    value = torch.randn(
        batch, sum(h * w for h, w in sizes), heads, channels, device=device, dtype=dtype
    )
    loc_shape = (batch, queries, heads, len(sizes), 4, 2)
    scale = shapes.flip(-1).reshape(1, 1, 1, -1, 1, 2)
    # Interior quarter-pixel coordinates are exactly representable for these
    # power-of-two spatial sizes, including BF16. Boundary tests live elsewhere.
    pixels = (torch.rand(loc_shape, device=device) * (scale - 1)).floor() + 0.75
    locations = (pixels / scale).to(dtype)
    weights = torch.rand(loc_shape[:-1], device=device).flatten(-2).softmax(-1)
    weights = weights.reshape(loc_shape[:-1]).to(dtype)
    for tensor in (value, locations, weights):
        tensor.requires_grad_()
    grad = torch.randn(batch, queries, heads * channels, device=device, dtype=dtype)
    return (value, shapes, starts, locations, weights), grad, sizes, scale


def validate(fn, args, grad, truth, scale):
    tensors = (args[0], args[3], args[4])
    output = fn(*args)
    actual = (output, *torch.autograd.grad(output, tensors, grad))
    tolerance = {torch.float32: 2e-4, torch.float16: 2e-3, torch.bfloat16: 2e-2}[args[0].dtype]
    errors = {}
    passed = True
    for index, (name, a, e) in enumerate(
        zip(("output", "grad_value", "grad_locations", "grad_weights"), actual, truth)
    ):
        a, e = a.double(), e.double()
        if index == 2:
            a, e = a / scale, e / scale
        close = torch.allclose(a, e, atol=tolerance, rtol=tolerance)
        errors[name] = {"max_abs": (a - e).abs().max().item(), "close": close}
        passed = passed and close
    return {
        "passed": passed,
        "atol": tolerance,
        "rtol": tolerance,
        "location_gradient_units": "feature pixels",
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--profile-kernels", action="store_true")
    parser.add_argument("--native-control", action="store_true")
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=("float32", "float16", "bfloat16"),
        default=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("forward", "backward", "forward_backward"),
        default=["forward", "backward", "forward_backward"],
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=(
            "torch-ms-deform-attn",
            "kernel-hub-adapter",
            "hf-native",
            "mmcv-source",
            "msda-triton-rziga",
            "pytorch-reference",
            "upstream-native-control",
            "upstream-before-perf",
        ),
    )
    args = parser.parse_args()
    if args.backends and "upstream-native-control" in args.backends:
        args.native_control = True
    if min(args.warmup, args.repeats, args.threads, args.min_run_time) <= 0:
        parser.error("measurement parameters must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    def selected(name):
        return args.backends is None or name in args.backends

    report = {
        "environment": environment(args.seed, args.warmup),
        "sources": json.loads((args.sources / "sources.json").read_text()),
        "config": {**vars(args), "output": str(args.output), "sources": str(args.sources)},
        "cases": CASES,
        "execution": "eager; JIT/warmup excluded",
        "ordering": "qualified backend/policy order reshuffled for each repetition",
        "timing": "synchronized host wall latency; CUDA event interval includes idle gaps",
        "memory": "allocated bytes, excludes allocator reservations; backward retains graph",
        "results": [],
        "setup_errors": {},
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    hf = get_kernel(
        "kernels-community/deformable-detr", revision=os.environ["MSDA_HF_BASELINE_REVISION"]
    )
    local = get_local_kernel(Path(os.environ["MSDA_KERNEL_DIR"]))
    report["hf_revision"] = os.environ["MSDA_HF_BASELINE_REVISION"]
    backends = {
        "torch-ms-deform-attn": ms_deform_attn,
        "kernel-hub-adapter": local.ms_deform_attn,
        "hf-native": extension_function(hf.ms_deform_attn_forward, hf.ms_deform_attn_backward),
    }
    if args.native_control:
        # Diagnostic only: same upstream CUDA kernels through legacy autograd,
        # bypassing the public torch.library registration and functional API.
        backends["upstream-native-control"] = extension_function(
            _C.ms_deform_attn_forward, _C.ms_deform_attn_backward
        )
    if args.backends and "upstream-before-perf" in args.backends:
        try:
            backends["upstream-before-perf"] = load_previous(args.sources)
        except Exception as exc:
            report["setup_errors"]["upstream-before-perf"] = str(exc)
            save()
    if selected("mmcv-source"):
        try:
            mmcv = load_mmcv(args.sources)
            backends["mmcv-source"] = extension_function(mmcv.forward, mmcv.backward, inplace=True)
        except Exception as exc:
            report["setup_errors"]["mmcv-source"] = str(exc)
            save()
    if selected("msda-triton-rziga"):
        try:
            # The public frontend silently falls back to grid_sample on errors.
            # Call its strict Triton frontend so fallback cannot be timed as Triton.
            from msda_triton.frontend import triton_multiscale_deformable_attention

            def triton(value, shapes, starts, locations, weights):
                return triton_multiscale_deformable_attention(
                    value, shapes, locations, weights, "zeros", False
                ).flatten(2)

            backends["msda-triton-rziga"] = triton
        except Exception as exc:
            report["setup_errors"]["msda-triton-rziga"] = str(exc)
            save()
    required_failed = bool(report["setup_errors"])
    for case in args.cases:
        for dtype in (getattr(torch, name) for name in args.dtypes):
            data, grad, sizes, scale = inputs(case, dtype)
            tensors = (data[0], data[3], data[4])

            def reference(value, shapes, starts, locations, weights):
                return ms_deform_attn_core_pytorch(value, sizes, locations, weights)

            oracle_inputs = tuple(x.detach().double().requires_grad_() for x in tensors)
            oracle_out = reference(oracle_inputs[0], data[1], data[2], *oracle_inputs[1:])
            truth = (
                oracle_out.detach(),
                *torch.autograd.grad(oracle_out, oracle_inputs, grad.double()),
            )
            del oracle_out, oracle_inputs
            implementations = {**backends, "pytorch-reference": reference}
            implementations = {name: fn for name, fn in implementations.items() if selected(name)}
            variants = []
            for backend, fn in implementations.items():
                # Public upstream/adapter already implement this policy internally.
                direct = backend in (
                    "torch-ms-deform-attn",
                    "kernel-hub-adapter",
                    "upstream-before-perf",
                )
                variants.append(
                    (
                        backend,
                        "fp32-compute",
                        fn if direct or dtype == torch.float32 else matched_precision(fn),
                    )
                )
                if dtype != torch.float32 and not direct and backend != "pytorch-reference":
                    variants.append((backend, "native", fn))
            rng.shuffle(variants)
            qualified = []
            for backend, policy, fn in variants:
                label = {"case": case, "dtype": str(dtype), "backend": backend, "policy": policy}
                print(label, flush=True)
                try:
                    check = validate(fn, data, grad, truth, scale)
                    if not check["passed"]:
                        report["results"].append(
                            {**label, "status": "incorrect", "correctness": check}
                        )
                        required_failed |= policy == "fp32-compute"
                        save()
                        continue
                    qualified.append((label, fn, check))
                except Exception as exc:
                    report["results"].append(
                        {**label, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                    )
                    required_failed |= policy == "fp32-compute"
                save()
            failed = set()
            for repeat in range(args.repeats):
                rng.shuffle(qualified)
                for label, fn, check in qualified:
                    key = (label["backend"], label["policy"])
                    if key in failed:
                        continue
                    try:
                        gc.collect()
                        for mode in args.modes:
                            output = fn(*data) if mode == "backward" else None

                            def run():
                                if mode == "forward":
                                    with torch.no_grad():
                                        return fn(*data)
                                if mode == "backward":
                                    return torch.autograd.grad(
                                        output, tensors, grad, retain_graph=True
                                    )
                                return torch.autograd.grad(fn(*data), tensors, grad)

                            timing = measure(run, args)
                            report["results"].append(
                                {
                                    **label,
                                    "repeat": repeat,
                                    "mode": mode,
                                    "status": "passed",
                                    "correctness": check,
                                    **timing,
                                }
                            )
                            output = None
                    except Exception as exc:
                        failed.add(key)
                        report["results"] = [
                            row
                            for row in report["results"]
                            if not all(row.get(k) == v for k, v in label.items())
                        ]
                        report["results"].append(
                            {**label, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                        )
                        required_failed |= label["policy"] == "fp32-compute"
                    finally:
                        output = None
                    save()
            del truth
            data = grad = tensors = None
    report["status"] = "failed" if required_failed else "passed"
    report["cpu_scheduling_end"] = cpu_scheduling()
    save()
    return int(required_failed)


if __name__ == "__main__":
    sys.exit(main())

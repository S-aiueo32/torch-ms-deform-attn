"""RT-DETR -> Transformers Kernel Hub integration -> local MSDA build.

Default runs require CUDA and a real Kernel Builder artifact. CPU fixture tests
exercise integration plumbing only, and cannot establish GPU compatibility.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import traceback
from contextlib import nullcontext
from pathlib import Path
from types import MethodType

import torch
from kernels import LocalLayerRepository, Mode, get_local_kernel
from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrResNetConfig
from transformers.integrations.hub_kernels import kernelize
from transformers.utils.kernel_config import KernelConfig


def tiny_model():
    """Complete randomly initialized detector, with small widths for regression tests."""
    backbone = RTDetrResNetConfig(
        embedding_size=16,
        hidden_sizes=[16, 32, 64, 128],
        depths=[1, 1, 1, 1],
        layer_type="basic",
        out_indices=[2, 3, 4],
    )
    config = RTDetrConfig(
        backbone_config=backbone,
        encoder_in_channels=[32, 64, 128],
        encoder_hidden_dim=32,
        encoder_ffn_dim=64,
        encoder_attention_heads=4,
        d_model=32,
        decoder_in_channels=[32, 32, 32],
        decoder_ffn_dim=64,
        decoder_attention_heads=4,
        decoder_layers=2,
        num_queries=10,
        num_labels=3,
        num_denoising=0,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    return RTDetrForObjectDetection(config)


def bind_kernel(model, kernel_dir, *, compiled, require_adapter=True):
    module = get_local_kernel(Path(kernel_dir))
    if require_adapter and not hasattr(module, "_registrations"):
        raise RuntimeError("Selected kernel is not the torch-ms-deform-attn adapter")
    mode = Mode.TRAINING if model.training else Mode.INFERENCE
    if compiled:
        mode |= Mode.TORCH_COMPILE
    model.kernel_config = KernelConfig(
        kernel_mapping={
            "MultiScaleDeformableAttention": {
                model.device.type: {
                    mode: LocalLayerRepository(
                        Path(kernel_dir), layer_name="MultiScaleDeformableAttention"
                    )
                }
            }
        },
        use_local_kernel=True,
        inherit_mapping=False,
    )
    # Use Transformers' integration, including its mode/device selection.
    kernelize(model, mode=mode)
    targets = [
        (name, child)
        for name, child in model.named_modules()
        if getattr(child, "kernel_layer_name", None) == "MultiScaleDeformableAttention"
    ]
    if not targets:
        raise RuntimeError("No Transformers MSDA layers were discovered")
    expected = module.layers.MultiScaleDeformableAttention.forward
    for name, child in targets:
        if child.forward.__func__ is not expected:
            raise RuntimeError(f"Kernel integration fell back at {name}")
    return module, [name for name, _ in targets]


def detection_loss(model, output, labels):
    # Keep the Hungarian matcher outside the compiled model graph. This is the
    # detector's actual supervised criterion, not a surrogate sum of outputs.
    def fp32(value):
        return value.float() if isinstance(value, torch.Tensor) else value

    # Hungarian cdist and box losses run in FP32, including explicit half models.
    return model.loss_function(
        fp32(output.intermediate_logits[:, -1]),
        labels,
        model.device,
        fp32(output.intermediate_reference_points[:, -1]),
        model.config,
        fp32(output.intermediate_logits),
        fp32(output.intermediate_reference_points),
        enc_topk_logits=fp32(output.enc_topk_logits),
        enc_topk_bboxes=fp32(output.enc_topk_bboxes),
        denoising_meta_values=output.denoising_meta_values,
        predicted_corners=fp32(output.intermediate_predicted_corners),
        initial_reference_points=fp32(output.initial_reference_points),
    )[0]


def run_model(model, pixels, labels, *, training, amp_dtype=None, execute=None):
    # The detection heads are attached to model.model.decoder by the outer
    # RTDetrForObjectDetection constructor. Preserve encoder proposals for the
    # real criterion while compiling the complete learned detector core.
    execute = execute if execute is not None else model.model
    context = torch.autocast(model.device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
    model.zero_grad(set_to_none=True)
    inputs = pixels.detach().clone().requires_grad_(training)
    with torch.set_grad_enabled(training), context:
        output = execute(pixel_values=inputs)
        loss = detection_loss(model, output, labels) if training else None
    if loss is not None:
        loss.backward()
    result = {
        "logits": output.intermediate_logits[:, -1].detach(),
        "boxes": output.intermediate_reference_points[:, -1].detach(),
    }
    if training:
        result["loss"] = loss.detach()
        result["input_grad"] = inputs.grad.detach()
        result.update(
            {
                f"grad:{name}": p.grad.detach().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
        )
    for name, tensor in result.items():
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"Non-finite {name}")
    return result


def artifact_hashes(module):
    # HF snapshot files are symlinks into a shared blobs directory. Preserve
    # their containing snapshot directory when finding sibling package files.
    directory = Path(module.__file__).parent.resolve()
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def reference_precision(model):
    """Evaluate reference interpolation in FP32 within the same precision model.

    CPU grid_sample and the existing HF native op do not accept all low dtypes.
    This adaptation tests our FP32-compute policy; it is not a claim that the
    unmodified baseline supports half inputs. Keeping the surrounding model in
    the same precision avoids different top-k proposals caused by quantization.
    """

    def wrap(original):
        def forward(self, value, shapes, shape_list, starts, locations, weights, step):
            output = original(
                value.float(), shapes, shape_list, starts, locations.float(), weights.float(), step
            )
            return (
                output if torch.is_autocast_enabled(value.device.type) else output.to(value.dtype)
            )

        return forward

    for child in model.modules():
        if getattr(child, "kernel_layer_name", None) == "MultiScaleDeformableAttention":
            child.forward = MethodType(wrap(child.forward), child)


def run_case(
    kernel_dir,
    *,
    training=False,
    backend=None,
    device="cuda",
    dtype="fp32",
    amp=False,
    baseline_kernel_dir=None,
):
    torch.manual_seed(123)
    torch.set_num_threads(1)
    candidate = tiny_model().to(device).train(training)
    scalar_type = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    if not amp:
        candidate.to(scalar_type)
    reference = copy.deepcopy(candidate)
    module, replaced = bind_kernel(candidate, kernel_dir, compiled=backend is not None)
    baseline_module = None
    if baseline_kernel_dir is not None:
        baseline_module, _ = bind_kernel(
            reference, baseline_kernel_dir, compiled=False, require_adapter=False
        )
        if baseline_module is module:
            raise ValueError("Baseline and candidate resolved to the same kernel module")
    promoted_reference = dtype != "fp32" and baseline_module is None
    if promoted_reference:
        reference_precision(reference)
    pixels = torch.randn(2, 3, 64, 64, device=device)
    labels = [
        {
            "class_labels": torch.tensor([0, 2], device=device),
            "boxes": torch.tensor([[0.3, 0.4, 0.2, 0.3], [0.7, 0.6, 0.2, 0.2]], device=device),
        }
        for _ in range(2)
    ]
    candidate_pixels = pixels if amp else pixels.to(scalar_type)
    baseline_original_error = None
    try:
        expected = run_model(
            reference,
            candidate_pixels,
            labels,
            training=training,
            amp_dtype=scalar_type if amp else None,
        )
    except RuntimeError as error:
        # Some HF versions reject mixed AMP inputs. Record the original failure
        # before comparing with explicit FP32 interpolation. Never recover from
        # arbitrary CUDA/device errors or a failure of our candidate.
        dtype_error = any(
            message in str(error)
            for message in ("expected scalar type", "not implemented for", "expected dtype")
        )
        if baseline_module is None or dtype == "fp32" or not dtype_error:
            raise
        baseline_original_error = str(error)
        reference_precision(reference)
        promoted_reference = True
        expected = run_model(
            reference,
            candidate_pixels,
            labels,
            training=training,
            amp_dtype=scalar_type if amp else None,
        )
    graphs = []
    execute = None
    if backend:
        torch._dynamo.reset()
        from torch._dynamo.backends.registry import lookup_backend

        compiler = lookup_backend(backend)

        def record_graph(graph, inputs):
            graphs.append(
                {
                    "node_count": len(list(graph.graph.nodes)),
                    "msda_calls": sum(
                        node.target == module._registrations.forward._opoverload
                        for node in graph.graph.nodes
                    ),
                }
            )
            return compiler(graph, inputs)

        # RT-DETR has a data-dependent finite-value check during training.
        # Permit that model graph break, but require MSDA inside captured graphs.
        execute = torch.compile(candidate.model, backend=record_graph, fullgraph=not training)
    # Warm up compile outside the profiler; observed ops must be real execution.
    if backend:
        run_model(
            candidate,
            candidate_pixels,
            labels,
            training=training,
            amp_dtype=scalar_type if amp else None,
            execute=execute,
        )
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = run_model(
            candidate,
            candidate_pixels,
            labels,
            training=training,
            amp_dtype=scalar_type if amp else None,
            execute=execute,
        )
    names = {event.key: event.count for event in profile.key_averages()}
    prefix = module._registrations.forward._qualname.rsplit("::", 1)[0]
    required = [f"{prefix}::forward"] + ([f"{prefix}::backward"] if training else [])
    for name in required:
        if not names.get(name):
            raise AssertionError(f"Profiler did not observe {name}; refusing fallback evidence")
    if backend and not any(graph["msda_calls"] for graph in graphs):
        raise AssertionError("No compiled graph contains the adapter's MSDA operator")
    if actual.keys() != expected.keys():
        raise AssertionError("Gradient coverage differs from the reference")
    atol, rtol = (2e-5, 2e-4) if dtype == "fp32" else (2e-2, 5e-2)
    errors = {}
    for name in actual:
        value, target = actual[name].float(), expected[name].float()
        torch.testing.assert_close(
            value, target, atol=atol, rtol=rtol, msg=lambda msg: f"{name}: {msg}"
        )
        errors[name] = (value - target).abs().max().item()
    return {
        "training": training,
        "compile_backend": backend,
        "dtype": dtype,
        "autocast": amp,
        "replaced_layers": replaced,
        "executed_ops": {name: names[name] for name in required},
        "atol": atol,
        "rtol": rtol,
        "max_absolute_errors": errors,
        "gradient_tensor_count": sum(name.startswith("grad:") for name in actual),
        "compiled_graphs": graphs,
        "fullgraph": not training if backend else None,
        "reference": "HF kernel" if baseline_module else "Transformers grid_sample",
        "reference_msda_fp32_adapter": promoted_reference,
        "baseline_original_error": baseline_original_error,
        "candidate_sha256": artifact_hashes(module),
        "baseline_sha256": artifact_hashes(baseline_module) if baseline_module else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--baseline-kernel-dir",
        type=Path,
        help="Optional downloaded, pinned existing HF artifact for FP32 comparison",
    )
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--autocast", action="store_true")
    args = parser.parse_args()
    report = {
        "status": "failed",
        "model": "RTDetrForObjectDetection",
        "weights": "random seed 123",
        "scope": "CUDA local Kernel Hub artifact; tiny full detector; no dataset accuracy claim",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "versions": {
            name: importlib.metadata.version(name) for name in ("transformers", "kernels")
        },
        "kernel_dir": str(args.kernel_dir.resolve()),
        "cases": [],
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase 2 GPU validation requires CUDA")
        if args.autocast and args.dtype == "fp32":
            raise ValueError("Autocast requires fp16 or bf16")
        report["gpu"] = torch.cuda.get_device_name()
        for training in (False, True):
            for backend in (None, "inductor"):
                try:
                    case = run_case(
                        args.kernel_dir,
                        training=training,
                        backend=backend,
                        dtype=args.dtype,
                        amp=args.autocast,
                        baseline_kernel_dir=args.baseline_kernel_dir,
                    )
                except Exception:
                    case = {
                        "training": training,
                        "compile_backend": backend,
                        "error": traceback.format_exc(),
                    }
                report["cases"].append(case)
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
        if any("error" in case for case in report["cases"]):
            raise RuntimeError("E2E cases failed; inspect per-case errors in the report")
        report["status"] = "passed"
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

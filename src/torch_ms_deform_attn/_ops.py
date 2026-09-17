"""Dispatcher, FakeTensor, and first-order autograd registrations."""

import torch

from . import _C  # noqa: F401 - loads the native dispatcher registrations


def _native_op(name):
    namespace, op = name.split("::")
    return getattr(getattr(torch.ops, namespace), op).default


forward = _native_op("torch_ms_deform_attn::forward")


def _validate_fake_inputs(value, shapes, starts, locations, weights, step, grad=None):
    # Separate checks preserve symbolic dimensions without Python bool/int conversion.
    for tensor, rank, name in (
        (value, 4, "value"),
        (shapes, 2, "spatial_shapes"),
        (starts, 1, "level_start_index"),
        (locations, 6, "sampling_locations"),
        (weights, 5, "attention_weights"),
    ):
        torch._check(tensor.dim() == rank, lambda: f"Invalid {name} rank")
    torch._check(
        value.dtype in (torch.float32, torch.float64),
        lambda: "deformable attention supports float32 and float64",
    )
    if value.device.type == "mps":
        torch._check(value.dtype == torch.float32, lambda: "MPS requires float32 computation")
    for tensor in (locations, weights):
        torch._check(tensor.dtype == value.dtype, lambda: "Floating input dtypes must match")
    for tensor in (shapes, starts):
        torch._check(
            tensor.dtype == torch.int64,
            lambda: "spatial_shapes and level_start_index must be int64",
        )
    for tensor in (shapes, starts, locations, weights):
        torch._check(tensor.device == value.device, lambda: "All inputs must be on the same device")
    torch._check(shapes.shape[1] == 2, lambda: "Invalid spatial_shapes shape")
    torch._check(starts.shape[0] == shapes.shape[0], lambda: "Invalid level_start_index shape")
    for actual, expected in (
        (locations.shape[0], value.shape[0]),
        (locations.shape[2], value.shape[2]),
        (locations.shape[3], shapes.shape[0]),
        (locations.shape[5], 2),
    ):
        torch._check(actual == expected, lambda: "Invalid sampling_locations shape")
    for actual, expected in zip(weights.shape, locations.shape[:5]):
        torch._check(actual == expected, lambda: "Invalid attention_weights shape")
    torch._check(step > 0, lambda: "im2col_step must be positive")
    if value.device.type in ("cuda", "mps"):
        for size in (*value.shape, shapes.shape[0], locations.shape[1], locations.shape[4]):
            torch._check(size > 0, lambda: "CUDA and MPS inputs must be nonempty")
    if grad is not None:
        torch._check(grad.dim() == 3, lambda: "Invalid grad_output rank")
        torch._check(grad.dtype == value.dtype, lambda: "grad_output dtype must match value")
        torch._check(grad.device == value.device, lambda: "grad_output device must match value")
        for actual, expected in zip(
            grad.shape, (value.shape[0], locations.shape[1], value.shape[2] * value.shape[3])
        ):
            torch._check(actual == expected, lambda: "Invalid grad_output shape")


@torch.library.register_fake(forward)
def _forward_fake(value, shapes, starts, locations, weights, step):
    _validate_fake_inputs(value, shapes, starts, locations, weights, step)
    return value.new_empty((value.shape[0], locations.shape[1], value.shape[2] * value.shape[3]))


backward = _native_op("torch_ms_deform_attn::backward")


@torch.library.register_fake(backward)
def _backward_fake(value, shapes, starts, locations, weights, grad, step):
    _validate_fake_inputs(value, shapes, starts, locations, weights, step, grad)
    return (
        torch.empty_like(value, memory_format=torch.contiguous_format),
        torch.empty_like(locations, memory_format=torch.contiguous_format),
        torch.empty_like(weights, memory_format=torch.contiguous_format),
    )


def _setup_context(ctx, inputs, output):
    value, shapes, starts, locations, weights, step = inputs
    ctx.save_for_backward(value, shapes, starts, locations, weights)
    ctx.step = step


def _autograd_backward(ctx, grad):
    value, shapes, starts, locations, weights = ctx.saved_tensors
    gv, gl, gw = backward(value, shapes, starts, locations, weights, grad, ctx.step)
    return gv, None, None, gl, gw, None


torch.library.register_autograd(forward, _autograd_backward, setup_context=_setup_context)


def _reject_higher_order(ctx, *grads):
    raise RuntimeError("No autograd formula for backward: higher-order gradients are unsupported")


torch.library.register_autograd(backward, _reject_higher_order)

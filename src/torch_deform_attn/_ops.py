"""Dispatcher, FakeTensor, and first-order autograd registrations."""
import torch

from . import _C


@torch.library.custom_op("torch_deform_attn::forward", mutates_args=())
def forward(value: torch.Tensor, shapes: torch.Tensor, starts: torch.Tensor,
            locations: torch.Tensor, weights: torch.Tensor, step: int) -> torch.Tensor:
    if value.is_cuda:
        value, shapes, starts, locations, weights = (
            t.contiguous() for t in (value, shapes, starts, locations, weights))
    return _C.ms_deform_attn_forward(value, shapes, starts, locations, weights, step)


@forward.register_fake
def _forward_fake(value, shapes, starts, locations, weights, step):
    return value.new_empty((value.shape[0], locations.shape[1], value.shape[2] * value.shape[3]))


@torch.library.custom_op("torch_deform_attn::backward", mutates_args=())
def backward(value: torch.Tensor, shapes: torch.Tensor, starts: torch.Tensor,
             locations: torch.Tensor, weights: torch.Tensor, grad: torch.Tensor,
             step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if value.is_cuda:
        value, shapes, starts, locations, weights = (
            t.contiguous() for t in (value, shapes, starts, locations, weights))
    return tuple(_C.ms_deform_attn_backward(value, shapes, starts, locations, weights,
                                          grad.contiguous(), step))


@backward.register_fake
def _backward_fake(value, shapes, starts, locations, weights, grad, step):
    return (torch.empty_like(value, memory_format=torch.contiguous_format),
            torch.empty_like(locations, memory_format=torch.contiguous_format),
            torch.empty_like(weights, memory_format=torch.contiguous_format))


def _setup_context(ctx, inputs, output):
    value, shapes, starts, locations, weights, step = inputs
    ctx.save_for_backward(value, shapes, starts, locations, weights)
    ctx.step = step


def _autograd_backward(ctx, grad):
    value, shapes, starts, locations, weights = ctx.saved_tensors
    gv, gl, gw = backward(value, shapes, starts, locations, weights, grad, ctx.step)
    return gv, None, None, gl, gw, None


forward.register_autograd(_autograd_backward, setup_context=_setup_context)
# Deliberately no autograd formula for backward: higher-order gradients are unsupported.

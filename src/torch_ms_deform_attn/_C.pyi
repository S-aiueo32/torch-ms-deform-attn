"""Types for the native bindings declared in csrc/vision.cpp."""

from collections.abc import Sequence

from jaxtyping import Float, Int64
from torch import Tensor

cpu_parallel_backend: str
with_cuda: bool

def ms_deform_attn_forward(
    value: Float[Tensor, "N S M D"],
    shapes: Int64[Tensor, "L 2"],
    starts: Int64[Tensor, "L"],  # noqa: F821 - jaxtyping dimension
    locations: Float[Tensor, "N Q M L P 2"],
    weights: Float[Tensor, "N Q M L P"],
    step: int,
) -> Float[Tensor, "N Q M*D"]: ...
def ms_deform_attn_backward(
    value: Float[Tensor, "N S M D"],
    shapes: Int64[Tensor, "L 2"],
    starts: Int64[Tensor, "L"],  # noqa: F821 - jaxtyping dimension
    locations: Float[Tensor, "N Q M L P 2"],
    weights: Float[Tensor, "N Q M L P"],
    grad: Float[Tensor, "N Q M*D"],
    step: int,
) -> list[Float[Tensor, "N S M D"] | Float[Tensor, "N Q M L P 2"] | Float[Tensor, "N Q M L P"]]:
    """Return [grad_value, grad_locations, grad_weights], in that order."""
    ...

def _cpu_parallel_worker_count(work_items: int) -> int: ...
def _check_cuda_indexing(dims: Sequence[int], step: int) -> list[int]: ...

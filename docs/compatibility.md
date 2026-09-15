# Upstream module compatibility

The test fixture pins [Deformable-DETR revision 11169a60c33333af00a4849f1808023eba96a931](https://github.com/fundamentalvision/Deformable-DETR/tree/11169a60c33333af00a4849f1808023eba96a931).
`tests/integration/fixtures` preserves the original module and reference source,
including copyright notices, and checks their SHA-256 hashes before execution.
These are test assets, not installed model implementations.

Replace the upstream module's relative function import with:

```python
from torch_ms_deform_attn import MSDeformAttnFunction
```

`MSDeformAttnFunction.apply` accepts the same six positional arguments. Tests run
the unchanged upstream module body, swapping only that dependency. The comparison
uses the pinned upstream **PyTorch reference**, not the original CUDA extension.
Both 2-coordinate reference points and 4-coordinate boxes, padding masks, all
three operator-input gradients, module-input/parameter gradients, and two SGD
steps are compared on CPU and available CUDA, in float32 and fp16/bf16 autocast.
AMP reference inputs are explicitly promoted to float32 to match this package's
operator contract. This is module-level equivalence, not detector accuracy or
full-training reproduction.

Under AMP the operator returns float32; the module's following output projection
is autocast and therefore returns the selected low-precision dtype. Outside
AMP, native kernels require matching float32 or float64 inputs. CPU empty-input
support does not imply the upstream module handles every empty dimension; CUDA
still rejects empty dimensions. The upstream module requires contiguous packed
feature levels even though the standalone operator also accepts arbitrary valid
offsets. See [API contracts](api.md) for the remaining constraints.

Run with `python -m unittest discover -s tests -v`; GPU skips are not GPU evidence.

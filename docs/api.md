# API reference

[Back to README](../README.md)

Use `ms_deform_attn` to sample and combine feature levels with first-order
gradients for values, sampling locations, and attention weights.

## `ms_deform_attn`

```python
ms_deform_attn(value, spatial_shapes, level_start_index, sampling_locations,
               attention_weights, im2col_step=64)
```

See the [README](../README.md#use) for a complete forward/backward example.

| Argument | Shape | Meaning |
| --- | --- | --- |
| `value` | `[N, S, M, D]` | Flattened feature levels; M heads, D channels per head |
| `spatial_shapes` | `[L, 2]` | Height and width per level, int64 |
| `level_start_index` | `[L]` | Each level's starting offset in S, int64 |
| `sampling_locations` | `[N, Q, M, L, P, 2]` | Normalized x/y sampling coordinates |
| `attention_weights` | `[N, Q, M, L, P]` | Weights used directly, without softmax |
| `im2col_step` | Integer, default `64` | Positive maximum CUDA batch chunk size; unused for CPU chunking |
| output | `[N, Q, M * D]` | Weighted sum of sampled features |

`N` is batch size, `S` is the number of flattened feature positions, `M` is the
number of heads, `D` is channels per head, `L` is feature levels, `Q` is queries,
and `P` is sampling points per query/head/level.

### Input constraints and gradients

All inputs must be on the same CPU, CUDA, or MPS device. Outside autocast, floating
tensors must share float16, bfloat16, float32, or float64 dtype.
For CUDA, move every tensor in the example to CUDA,
including `shapes` and `starts`.
Noncontiguous tensors are accepted. Sampling is bilinear, with zero padding and
`align_corners=False`; finite coordinates outside [0, 1] are allowed.
Nonfinite coordinates are not part of the supported input contract.
Gradients are provided for value, locations, and weights. Higher-order gradients
and native float16/bfloat16 kernel arithmetic are unsupported. Explicit low-precision
inputs and AMP use float32 computation; see [AMP and torch.compile](#amp-and-torchcompile).

Each level must have positive height and width and a nonnegative starting offset,
with `start + height * width <= S`. Levels may overlap.

### CUDA constraints and errors

CUDA processes at most `im2col_step` batch elements per launch, including a
smaller final chunk. Empty dimensions are unsupported. Individual dimensions
must fit int32; unsupported products or launch sizes are rejected before
allocation/launch. Reduce `im2col_step` if a batch chunk exceeds the CUDA grid limit.

CUDA backward uses atomic additions and is nondeterministic. With
`torch.use_deterministic_algorithms(True)`, backward raises an error;
`warn_only=True` emits a warning and permits execution.
CUDA validates spatial shapes and level offsets on the device before accessing
features, without a host synchronization. Invalid metadata triggers a device assertion, which may surface
at the next CUDA synchronization. A device assertion invalidates the CUDA context;
restart the process after such an error.

### MPS constraints and errors

The current checkout provides dedicated Metal forward/backward kernels on Apple
Silicon, macOS 13.3+, and PyTorch 2.4+. float32 and float16 inputs are supported;
bfloat16 requires macOS 14+ and a PyTorch build supporting that dtype. float64 and
empty dimensions are rejected. `im2col_step` must be positive but does not control
MPS dispatch sizes. Noncontiguous tensors, storage offsets, and overlapping levels
are supported.

Both kernels compute in float32. Shape/offset metadata is copied to the host for
range validation, which synchronizes pending GPU work. Feature tensors remain on
the GPU. Invalid metadata raises a regular error before kernel execution. The
backend never silently falls back to the CPU or the reference implementation.

Backward uses atomic additions and is nondeterministic. The deterministic-algorithm
error/warning policy matches CUDA. Metal source is compiled lazily on first use;
subsequent calls reuse the pipeline. MPS `torch.compile` support is not claimed.

## AMP and torch.compile

The public function and `MSDeformAttnFunction.apply` use the same dtype rules
on CPU, CUDA, and MPS (except that MPS cannot represent float64):

| Floating input dtype | Context | Computation | Output |
| --- | --- | --- | --- |
| float32 | With or without autocast | float32 | float32 |
| float64 | With or without autocast | float64 | float64 |
| float16 or bfloat16 | Outside autocast; all three dtypes must match | float32 | Input dtype |
| float16 or bfloat16 | Inside autocast | float32 | float32 |

Autograd returns gradients in each original input's dtype. Under autocast,
low-precision tensors are individually promoted to float32; float64 is preserved,
so mixing float64 with other floating dtypes still fails validation. Explicit
low-precision inputs allocate float32 copies and do not provide native
low-precision kernel speed or memory savings.

The examples below use the tensors from the [README example](../README.md#use).

```python
output = ms_deform_attn(value.half(), shapes, starts,
                       locations.half(), weights.half())
assert output.dtype == torch.float16
# Use .bfloat16() on all three floating inputs for bfloat16 output.
output.float().sum().backward()
```

```python
value, shapes, starts, locations, weights = (
    t.to("cuda") for t in (value, shapes, starts, locations, weights)
)
with torch.autocast("cuda", dtype=torch.float16):
    output = ms_deform_attn(value, shapes, starts, locations, weights)
# Backward can run outside the autocast context, including with GradScaler.
output.sum().backward()

compiled_attention = torch.compile(ms_deform_attn, fullgraph=True, dynamic=True)
output = compiled_attention(value.float(), shapes, starts,
                            locations.float(), weights.float())
```

Compilation treats the attention operator as opaque; it does not fuse the kernel
internals. Export/ONNX support is not claimed. See [development and CI](development.md)
for integration test coverage.

MPS autocast dtype availability varies with PyTorch/macOS. The float32 autocast
output rule applies when `torch.is_autocast_enabled("mps")` is true; older PyTorch
versions may disable unsupported autocast dtypes with a warning.

## `MSDeformAttnFunction.apply`

```python
MSDeformAttnFunction.apply(value, shapes, starts, locations, weights, im2col_step)
```

Compatibility entry point for upstream callers. It delegates to `ms_deform_attn`
with the same input contract, output, AMP, and compilation behavior.
`im2col_step` is required at this entry point.

## PyTorch reference

`ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations,
attention_weights)` is also exported. It uses `torch.nn.functional.grid_sample`
and is intended for correctness checks and benchmarks.

Unlike the extension, the reference has no `level_start_index` argument: it
splits `value` into consecutive levels whose areas must sum to `S`. It does not
represent overlapping or arbitrarily offset levels. Use the extension's public
function for the AMP and compilation behavior documented above.

## Type annotations

Shape annotations use jaxtyping names imported only during type checking. Normal
execution does not require jaxtyping; validation comes from the operator, not a
runtime typing decorator. ty checks Tensor types, not shape or dtype compatibility.
If you call `typing.get_type_hints()`, supply the jaxtyping names in its namespace.

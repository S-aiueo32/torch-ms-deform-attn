# API and integration

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

All inputs must be on the same CPU or CUDA device. Floating tensors must share
float32 or float64 dtype. For CUDA, move every tensor in the example to CUDA,
including `shapes` and `starts`.
Noncontiguous tensors are accepted. Sampling is bilinear, with zero padding and
`align_corners=False`; finite coordinates outside [0, 1] are allowed.
Nonfinite coordinates are not part of the supported input contract.
Gradients are provided for value, locations, and weights. Higher-order gradients
and native float16/bfloat16 kernel arithmetic are unsupported. AMP accepts
low-precision inputs by computing this operator in float32; see [AMP and torch.compile](#amp-and-torchcompile).

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
features, without a host
synchronization. Invalid metadata triggers a device assertion, which may surface
at the next CUDA synchronization. A device assertion invalidates the CUDA context;
restart the process after such an error.

## AMP and torch.compile

The public function and compatibility `.apply` entry point both support autocast
on CPU and CUDA. Inside autocast, float16/bfloat16 values, locations, and weights
are promoted to float32. Float32 inputs stay float32; float64 inputs are preserved.
Output is float32 for the low-precision path, and autograd casts gradients back
to the original input dtypes. Outside autocast, low-precision inputs are rejected.

```python
# All input tensors must already be on CUDA.
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

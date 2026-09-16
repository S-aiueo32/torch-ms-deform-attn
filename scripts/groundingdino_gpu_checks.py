"""Additional CUDA integration cases for the exact GroundingDINO revision."""
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import torch

root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('gd_attention', root / 'groundingdino/models/GroundingDINO/ms_deform_attn.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert torch.cuda.is_available()
results = []
for dtype, amp, fp32_refs, ref_dim, batch_first in itertools.product(
    (torch.float16, torch.bfloat16), (False, True), (False, True), (2, 4), (False, True)
):
    torch.manual_seed(42)
    model = m.MultiScaleDeformableAttention(embed_dim=32, num_heads=4, num_levels=2,
        num_points=4, batch_first=batch_first).cuda().to(torch.float32 if amp else dtype)
    qshape = (2, 5, 32) if batch_first else (5, 2, 32)
    vshape = (2, 20, 32) if batch_first else (20, 2, 32)
    query = torch.randn(qshape, device='cuda', dtype=next(model.parameters()).dtype, requires_grad=True)
    value = torch.randn(vshape, device='cuda', dtype=query.dtype, requires_grad=True)
    refs = torch.rand(2, 5, 2, ref_dim, device='cuda', dtype=torch.float32 if fp32_refs else dtype, requires_grad=True)
    with torch.autocast('cuda', dtype=dtype, enabled=amp):
        output = model(query, value=value, reference_points=refs,
            spatial_shapes=torch.tensor([[4, 4], [2, 2]], device='cuda'),
            level_start_index=torch.tensor([0, 16], device='cuda'))
    assert output.dtype == dtype, output.dtype
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    for tensor in (query, value, refs, *model.parameters()):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    results.append(dict(dtype=str(dtype), autocast=amp, fp32_refs=fp32_refs,
        reference_dim=ref_dim, batch_first=batch_first, success=True))
torch.cuda.synchronize()
Path(sys.argv[2]).write_text(json.dumps(dict(success=True, cases=results,
    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
    source_sha=os.environ['GROUNDINGDINO_SHA']), indent=2)+'\n')
print(f'CUDA low-precision output/backward: {len(results)} cases passed')

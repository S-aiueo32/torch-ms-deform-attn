import importlib
import importlib.metadata
from unittest.mock import patch
import json
import os
from pathlib import Path
import sys
import torch
from groundingdino.util.inference import load_model, load_image, predict
root = Path(sys.argv[1])
model = load_model(str(root/'groundingdino/config/GroundingDINO_SwinT_OGC.py'),
    str(root/'weights/groundingdino_swint_ogc.pth'), device='cuda')
_, image = load_image(str(root/'.asset/cat_dog.jpeg'))
boxes, scores, phrases = predict(model, image, 'cat . dog .', 0.3, 0.25, device='cuda')
assert len(boxes) > 0, 'No detections'
assert torch.isfinite(boxes).all() and torch.isfinite(scores).all()
attention = importlib.import_module('groundingdino.models.GroundingDINO.ms_deform_attn')
def reference(value, shapes, starts, locations, weights, step):
    return attention.multi_scale_deformable_attn_pytorch(value, shapes, locations, weights)
with patch.object(attention, 'ms_deform_attn', reference):
    reference_boxes, reference_scores, reference_phrases = predict(
        model, image, 'cat . dog .', 0.3, 0.25, device='cuda')
torch.testing.assert_close(boxes, reference_boxes, rtol=2e-3, atol=2e-4)
torch.testing.assert_close(scores, reference_scores, rtol=2e-3, atol=2e-4)
assert phrases == reference_phrases
torch.cuda.synchronize()
report = dict(success=True, source_sha=os.environ['GROUNDINGDINO_SHA'],
    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
    packages={p:importlib.metadata.version(p) for p in ['torch-ms-deform-attn','torchvision','transformers','timm']},
    reference_parity=True, boxes=boxes.tolist(), scores=scores.tolist(), phrases=phrases)
Path(sys.argv[2]).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))

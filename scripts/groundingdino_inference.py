import importlib.metadata
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
torch.cuda.synchronize()
report = dict(success=True, source_sha=os.environ['GROUNDINGDINO_SHA'],
    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
    packages={p:importlib.metadata.version(p) for p in ['torch-ms-deform-attn','torchvision','transformers','timm']},
    boxes=boxes.tolist(), scores=scores.tolist(), phrases=phrases)
Path(sys.argv[2]).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))

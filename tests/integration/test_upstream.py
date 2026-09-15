"""Execute pinned upstream module with only its operator dependency replaced."""

import ast
import copy
import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from torch_ms_deform_attn import _C, MSDeformAttnFunction

FIXTURES = Path(__file__).parent / "fixtures"


def load_module(operator):
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        if hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Upstream fixture changed: {name}")
    tree = ast.parse((FIXTURES / "ms_deform_attn.py.txt").read_text())
    # Retain the upstream class unchanged, replacing only the relative import.
    tree.body = [
        node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.level)
    ]
    namespace = {"MSDeformAttnFunction": operator}
    exec(compile(tree, "upstream/ms_deform_attn.py", "exec"), namespace)
    return namespace["MSDeformAttn"]


def upstream_reference():
    tree = ast.parse((FIXTURES / "ms_deform_attn_func.py.txt").read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "ms_deform_attn_core_pytorch"
    ]
    namespace = {"torch": torch, "F": F}
    exec(compile(tree, "upstream/ms_deform_attn_func.py", "exec"), namespace)
    return namespace["ms_deform_attn_core_pytorch"]


class UpstreamCases:
    device = "cpu"

    def test_module_optimizer_parity(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for reference_dim in (2, 4):
                with self.subTest(dtype=dtype, reference_dim=reference_dim):
                    self.check_module(dtype, reference_dim)

    def check_module(self, dtype, reference_dim):
        if self.device == "cuda" and dtype == torch.bfloat16:
            self.assertTrue(torch.cuda.is_bf16_supported(), "Requires bf16-capable GPU")
        torch.manual_seed(41)
        captured = [[], []]
        core = upstream_reference()

        def operator(which):
            def apply(v, s, i, loc, w, step):
                for tensor in (v, loc, w):
                    tensor.retain_grad()
                captured[which] = [v, loc, w]
                if which == 0:
                    return MSDeformAttnFunction.apply(v, s, i, loc, w, step)
                # Explicit AMP adapter: upstream grid_sample needs a shared dtype.
                return core(v.float(), s, loc.float(), w.float())

            return SimpleNamespace(apply=apply)

        model = load_module(operator(0))(d_model=8, n_levels=2, n_heads=2, n_points=2).to(
            self.device
        )
        reference = load_module(operator(1))(d_model=8, n_levels=2, n_heads=2, n_points=2).to(
            self.device
        )
        reference.load_state_dict(copy.deepcopy(model.state_dict()))
        # Avoid zero initialization hiding query-gradient propagation.
        with torch.no_grad():
            for name in ("sampling_offsets", "attention_weights"):
                layer = getattr(model, name)
                layer.weight.normal_(std=0.03)
                layer.bias.mul_(0.1)
            reference.load_state_dict(model.state_dict())
        data = [
            torch.randn(2, 3, 8, device=self.device),
            torch.rand(2, 3, 2, reference_dim, device=self.device) * 0.5 + 0.25,
            torch.randn(2, 10, 8, device=self.device),
        ]
        shapes = torch.tensor([[2, 3], [2, 2]], device=self.device)
        starts = torch.tensor([0, 6], device=self.device)
        mask = torch.zeros(2, 10, device=self.device, dtype=torch.bool)
        mask[:, -1] = True
        models = (model, reference)
        optimizers = [torch.optim.SGD(m.parameters(), lr=0.02) for m in models]
        scalers = [torch.amp.GradScaler(self.device, init_scale=8) for _ in models]
        tol = dict(atol=2e-5, rtol=1e-4) if dtype == torch.float32 else dict(atol=2e-3, rtol=2e-2)
        for _ in range(2):
            outputs, input_grads, op_grads = [], [], []
            for index, (target, optimizer, scaler) in enumerate(zip(models, optimizers, scalers)):
                optimizer.zero_grad(set_to_none=True)
                args = [t.detach().clone().requires_grad_() for t in data]
                with torch.autocast(self.device, dtype=dtype, enabled=dtype != torch.float32):
                    output = target(*args, shapes, starts, mask)
                    loss = output.float().square().mean()
                self.assertEqual(output.dtype, dtype)
                scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                outputs.append(output)
                input_grads.append([t.grad / scale for t in args])
                op_grads.append([t.grad / scale for t in captured[index]])
            torch.testing.assert_close(*outputs, **tol)
            for pairs in (input_grads, op_grads):
                for a, e in zip(*pairs):
                    self.assertTrue(torch.isfinite(a).all())
                    torch.testing.assert_close(a, e, **tol)
            for a, e in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, e.grad, **tol)
            before = [p.detach().clone() for p in model.parameters()]
            for scaler, optimizer in zip(scalers, optimizers):
                scaler.step(optimizer)
                scaler.update()
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
            for a, e in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a, e, **tol)


class CPUUpstreamTest(UpstreamCases, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available() and _C.with_cuda, "Requires CUDA extension and GPU")
class CUDAUpstreamTest(UpstreamCases, unittest.TestCase):
    device = "cuda"
